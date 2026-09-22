"""Prefill-only benchmark of a running server: unique prompts (so prefix caching cannot hide the cost), single-stream p50 and
throughput at a concurrency. `openjev bench --endpoint http://localhost:3000 --tokens 1400 --conc 32`."""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time

import httpx


def _request(i: int, tokens: int, rng: random.Random) -> dict:
    words = [
        "the",
        "page",
        "shows",
        "a",
        "form",
        "with",
        "fields",
        "and",
        "buttons",
        "the",
        "user",
        "wants",
        "to",
        "book",
        "a",
        "flight",
        "from",
        "SFO",
        "to",
        "JFK",
        "on",
        "Oct",
        "3",
        "previous",
        "actions",
        "none",
        "candidates",
    ]
    body = " ".join(rng.choice(words) + str(rng.randint(0, 99)) for _ in range(max(8, tokens // 2)))
    return {
        "model": "bench",
        "state": {"page": {"title": f"State {i}", "text": body}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "find flights", "rules": "Which candidate should be acted on next?"},
                "criteria": {"From": "input", "To": "input", "Search": "button", "Hotels": "link"},
            }
        },
    }


async def run(endpoint: str, token: str = "", n: int = 120, conc: int = 32, tokens: int = 1400) -> dict:
    rng = random.Random(0)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=180.0, headers=headers, limits=httpx.Limits(max_connections=conc + 4)) as client:

        async def one(i):
            t = time.perf_counter()
            r = await client.post(endpoint.rstrip("/") + "/v1/systemone", json=_request(i, tokens, rng))
            r.raise_for_status()
            return time.perf_counter() - t, r.json()["usage"]["input_tokens"]

        for i in range(3):
            await one(-1 - i)
        single = [(await one(i))[0] for i in range(10)]
        sem = asyncio.Semaphore(conc)

        async def bounded(i):
            async with sem:
                return await one(i)

        t0 = time.perf_counter()
        res = await asyncio.gather(*(bounded(i) for i in range(n)))
        wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    return {
        "n": n,
        "conc": conc,
        "prompt_tokens_mean": round(statistics.mean(r[1] for r in res)),
        "single_p50_ms": round(statistics.median(single) * 1000),
        "loaded_p50_ms": round(statistics.median(lat) * 1000),
        "loaded_p90_ms": round(lat[int(0.9 * n)] * 1000),
        "req_per_s": round(n / wall, 2),
        "tok_per_s": round(sum(r[1] for r in res) / wall),
    }


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="openjev bench")
    ap.add_argument("--endpoint", default="http://localhost:3000")
    ap.add_argument("--token", default="")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=1400)
    a = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run(a.endpoint, a.token, a.n, a.conc, a.tokens))))
