"""`openjev serve | calibrate | bench | version`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .backends import make_backend
from .config import PROFILES_DIR, Settings, load_profile
from .readout import Readout


def _serve(a):
    s = Settings(**{k: v for k, v in vars(a).items() if v is not None and k in Settings.model_fields})
    logging.basicConfig(level=s.log_level, format="%(message)s")
    profile = load_profile(
        s.profile, {"perms": a.perms, "temp": a.temp, "noul_t": a.noul_t, "assistant_prefix": a.assistant_prefix, "letter_prefix": a.letter_prefix}
    )
    try:
        backend, extra = make_backend(s, profile)
    except ValueError as e:
        sys.exit(str(e))
    readout = Readout(backend, profile)
    from .backends.probe import probe

    found = asyncio.run(backend.start())
    report = asyncio.run(
        probe(readout, backend, letter_prefix_used=found.get("letter_prefix", ""), exact=found.get("exact_readout"), vision=found.get("vision"))
    )
    found.update(report)
    logging.getLogger("openjev").info(json.dumps({"model_probe": found}))
    if "problem" in report and not a.force:
        sys.exit(f"refusing to serve: {report['problem']}  (start with --force to serve anyway)")
    import uvicorn

    from .server import build_app

    app = build_app(readout, token=s.token, model_dir=s.model, backend_name=s.backend, extra_version={**extra, "model_probe": found})
    logging.getLogger("openjev").info(
        json.dumps({"serving": f"http://{s.host}:{s.port}/v1/systemone", "backend": s.backend, "model": s.model, "profile": profile.to_dict()})
    )
    uvicorn.run(app, host=s.host, port=s.port, log_level="warning", access_log=False)


def _calibrate(a):
    from .calibrate import collect, fit, write_profile

    with open(a.dev) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    collected = asyncio.run(collect(rows, a.endpoint, a.token, a.concurrency))
    result = fit(collected)
    print(json.dumps(result, indent=1))
    base = Path(a.base) if Path(a.base).exists() else PROFILES_DIR / f"{a.base}.json"
    out = Path(a.out)
    write_profile(base, result, out.stem, out)
    print(f"profile written: {out}  (serve it with --profile {out})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openjev", description=f"openjev-server {__version__}: a decision API over any open model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve POST /v1/systemone")
    s.add_argument("--backend", choices=["vllm", "mlx"])
    s.add_argument("--model", help="tokenizer / model dir (vllm) or the MLX model dir (mlx)")
    s.add_argument("--vllm-url", dest="vllm_url")
    s.add_argument("--served-model-name", dest="served_model_name")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--token")
    s.add_argument("--profile", help="profile name under profiles/ or a JSON path")
    s.add_argument("--perms", type=int, help="average over N option orders (accuracy mode; costs latency)")
    s.add_argument("--temp", type=float)
    s.add_argument("--noul-t", dest="noul_t", type=float)
    s.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false", default=None)
    s.add_argument(
        "--assistant-prefix",
        dest="assistant_prefix",
        help="text at the start of the assistant turn before the readout (e.g. an empty think block for models that always reason first)",
    )
    s.add_argument("--letter-prefix", dest="letter_prefix", choices=["auto", "", " "], help="label token form: bare 'A', space-prefixed ' A', or auto")
    s.add_argument("--force", action="store_true", help="serve even when the startup probe reports a problem")
    s.set_defaults(fn=_serve)
    c = sub.add_parser("calibrate", help="fit temperature and yes/no scale on development rows against a running server")
    c.add_argument("--dev", required=True, help="JSONL: {state, questions: {id: question}, gold}")
    c.add_argument("--endpoint", default="http://localhost:3000")
    c.add_argument("--token", default="")
    c.add_argument("--base", default="uncalibrated", help="profile the server is running")
    c.add_argument("--out", required=True, help="where to write the fitted profile JSON")
    c.add_argument("--concurrency", type=int, default=8)
    c.set_defaults(fn=_calibrate)
    b = sub.add_parser("bench", help="latency / throughput of a running server")
    b.set_defaults(fn=lambda a: __import__("openjev_server.bench", fromlist=["main"]).main(a.rest))
    b.add_argument("rest", nargs=argparse.REMAINDER)
    v = sub.add_parser("version")
    v.set_defaults(fn=lambda a: print(__version__))
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
