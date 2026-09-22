"""The HTTP surface: POST /v1/systemone (decisions), POST /v1/prewarm, GET /v1/version, GET /healthz, GET /readyz, GET /metrics,
POST /v1/chat/completions (passthrough for clients that also type text). Bearer auth on /v1/* when a token is configured."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .backends.base import BackendError
from .readout import BadQuestion, Readout, ReadoutIncomplete

log = logging.getLogger("openjev")

REQS = Counter("openjev_requests_total", "systemone requests", ["status"])
QUESTIONS = Counter("openjev_questions_total", "questions answered", ["type"])
TOKENS = Counter("openjev_prompt_tokens_total", "prompt tokens sent to the model")
LAT = Histogram("openjev_request_seconds", "systemone request latency", buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13))
INFLIGHT = Gauge("openjev_inflight_requests", "requests in flight")

ANSWER_MODULES = ("prompt.py", "readout.py")  # the code that determines answers; its hash is reported by /v1/version


class Question(BaseModel):
    type: str
    instructions: Any = None
    criteria: Any = None

    @field_validator("type")
    @classmethod
    def _type(cls, v):
        if v not in ("choice", "score", "noul"):
            raise ValueError("type must be one of choice, score, noul")
        return v


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any
    questions: dict[str, Question] = Field(min_length=1)


class PrewarmRequest(BaseModel):
    state: Any


def http_error(status: int, message: str) -> HTTPException:
    return HTTPException(status, {"code": status, "message": message})


def version_info(readout: Readout, *, backend_name: str, model_dir: str, extra: dict | None) -> dict:
    """What GET /v1/version reports: the profile constants, the backend and a hash of the answer-determining code."""
    p = readout.profile
    here = Path(__file__).parent
    code = b"".join((here / m).read_bytes() for m in ANSWER_MODULES)
    return {
        "server": "openjev-server",
        "version": __version__,
        "backend": backend_name,
        "model_dir": os.path.basename(model_dir.rstrip("/")) or model_dir,
        "profile": p.name,
        "T": p.temp,
        "noul_t": p.noul_t,
        "noul_bias": p.noul_bias,
        "flags": {
            "perms": p.perms,
            "stagger": p.stagger,
            "compact": p.compact,
            "compact_cap": p.compact_cap,
            "layout": p.layout,
            "pad": p.pad,
            "targeted": True,
            "instr_style": p.instr_style,
        },
        "readout_sha256": hashlib.sha256(code).hexdigest(),
        **(extra or {}),
    }


def bearer_auth(token: str):
    """A dependency that requires `Authorization: Bearer <token>` when a token is configured (no token: open)."""

    async def auth(request: Request):
        if token and request.headers.get("Authorization", "") != f"Bearer {token}":
            raise http_error(401, "missing or invalid bearer token")

    return auth


def ops_router(readout: Readout, backend_name: str) -> APIRouter:
    """Unauthenticated operations endpoints: liveness, readiness, Prometheus metrics."""
    r = APIRouter()

    @r.get("/healthz")
    async def healthz():
        return {"ok": True}

    @r.get("/readyz")
    async def readyz():
        ok = await readout.backend.healthy()
        return JSONResponse({"ok": ok, "backend": backend_name}, status_code=200 if ok else 503)

    @r.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return r


def api_router(readout: Readout, version: dict, model_string: str, token: str) -> APIRouter:
    """The /v1 API: version, prewarm, systemone, chat passthrough."""
    r = APIRouter(prefix="/v1", dependencies=[Depends(bearer_auth(token))])

    @r.get("/version")
    @r.post("/version")
    async def get_version():
        return version

    @r.post("/prewarm")
    async def prewarm(body: PrewarmRequest):
        t0 = time.perf_counter()
        try:
            n = await readout.prewarm(body.state)
        except (BackendError, ReadoutIncomplete) as e:
            raise http_error(502, str(e)[:200]) from e
        log.info(json.dumps({"prewarm_tokens": n, "ms": round((time.perf_counter() - t0) * 1000)}))
        return {"ok": True, "prompt_tokens": n}

    @r.post("/systemone")
    async def systemone(body: SystemOneRequest):
        rid = f"oj-{uuid.uuid4().hex[:16]}"
        t0 = time.perf_counter()
        INFLIGHT.inc()
        try:
            answers, toks = await readout.answer_all(body.state, {k: q.model_dump() for k, q in body.questions.items()})
        except BadQuestion as e:
            REQS.labels("422").inc()
            raise http_error(422, str(e)) from e
        except ReadoutIncomplete as e:
            REQS.labels("502").inc()
            raise http_error(502, f"readout incomplete: {e}") from e
        except BackendError as e:
            REQS.labels("502").inc()
            raise http_error(502, str(e)[:300]) from e
        finally:
            INFLIGHT.dec()
        record(rid, answers, toks, time.perf_counter() - t0)
        return {"id": rid, "model": model_string, "answers": answers, "usage": {"input_tokens": toks, "output_tokens": 0}}

    @r.post("/chat/completions")
    async def chat(request: Request):
        fn = getattr(readout.backend, "passthrough", None)
        if fn is None:
            raise http_error(501, "this backend has no chat passthrough")
        code, data = await fn(await request.json())
        return Response(data, status_code=code, media_type="application/json")

    return r


def record(rid: str, answers: dict[str, dict], toks: int, dt: float) -> None:
    """Metrics and the one-line JSON log of a served request."""
    LAT.observe(dt)
    REQS.labels("200").inc()
    TOKENS.inc(toks)
    for a in answers.values():
        QUESTIONS.labels(a["type"]).inc()
    ops = {q: a.get("choice") for q, a in answers.items() if "choice" in a}
    log.info(json.dumps({"id": rid, "questions": len(answers), "tokens": toks, "ms": round(dt * 1000), "ops": ops}))


def build_app(readout: Readout, *, token: str = "", model_dir: str = "", backend_name: str = "", extra_version: dict | None = None) -> FastAPI:
    version = version_info(readout, backend_name=backend_name, model_dir=model_dir, extra=extra_version)
    p = readout.profile
    model_string = f"{version['model_dir']} T={p.temp} noul={p.noul_t},{p.noul_bias} profile={p.name} openjev-server@{__version__}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        close = getattr(readout.backend, "aclose", None)
        if close:
            await close()

    app = FastAPI(title="openjev-server", version=__version__, lifespan=lifespan, docs_url="/docs", redoc_url=None)

    @app.exception_handler(HTTPException)
    async def _http_exc(request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": exc.status_code, "message": str(exc.detail)}
        return JSONResponse({"error": detail}, status_code=exc.status_code)

    app.include_router(ops_router(readout, backend_name))
    app.include_router(api_router(readout, version, model_string, token))
    return app
