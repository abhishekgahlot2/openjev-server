"""Fit a profile's calibration constants on DEVELOPMENT rows (never on a test set): the choice temperature and the yes/no scale.

Input: a JSONL of rows {"state": ..., "questions": {"decision": {...}}, "gold": "<option key>" | true | false} and a running server
in the `uncalibrated` profile (temperature 1). The fit minimises negative log-likelihood on a grid; both constants are written to a
new profile JSON. Same procedure that produced OpenJev's constants and the 9B student's (T 1.07, yes/no 1.0748).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx


async def collect(rows: list[dict], endpoint: str, token: str = "", concurrency: int = 8) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=180.0, headers=headers) as client:

        async def one(r):
            async with sem:
                resp = await client.post(endpoint.rstrip("/") + "/v1/systemone", json={"model": "x", "state": r["state"], "questions": r["questions"]})
                resp.raise_for_status()
                qid = next(iter(r["questions"]))
                return {"gold": r["gold"], "answer": resp.json()["answers"][qid]}

        return list(await asyncio.gather(*(one(r) for r in rows)))


def split_rows(collected: list[dict], eps: float) -> tuple[list[tuple[list[float], int]], list[tuple[float, bool]]]:
    """Choice/score rows as (log-probabilities in key order, gold index); yes/no rows as (logit of p_yes, gold is yes)."""
    ch, nl = [], []
    for c in collected:
        a = c["answer"]
        if a["type"] == "noul":
            py = min(max(a["noul"], 1e-4), 1 - 1e-4)
            nl.append((math.log(py / (1 - py)), c["gold"] in (True, "true", "True", "yes", 1)))
        elif a["type"] in ("choice", "score"):
            keys = list(a["probabilities"])
            if str(c["gold"]) in keys:
                ch.append(([math.log(max(a["probabilities"][k], eps)) for k in keys], keys.index(str(c["gold"]))))
    return ch, nl


def fit(collected: list[dict], eps: float = 5e-5) -> dict:
    """Grid-search the choice temperature and the total yes/no logit scale that minimise NLL of the gold answers."""
    ch, nl = split_rows(collected, eps)

    def nll_choice(t):
        s = 0.0
        for lp, gi in ch:
            z = [x / t for x in lp]
            m = max(z)
            s -= z[gi] - (m + math.log(sum(math.exp(v - m) for v in z)))
        return s / len(ch)

    def nll_noul(sc):
        s = 0.0
        for lg, y in nl:
            p = 1 / (1 + math.exp(-lg / sc))
            p = min(max(p, 1e-9), 1 - 1e-9)
            s -= math.log(p if y else 1 - p)
        return s / len(nl)

    grid = [round(0.5 + 0.01 * i, 2) for i in range(251)]
    out = {"n_choice": len(ch), "n_noul": len(nl)}
    if ch:
        t = min(grid, key=nll_choice)
        out.update(
            temp=t,
            choice_nll_before=round(nll_choice(1.0), 4),
            choice_nll_after=round(nll_choice(t), 4),
            choice_acc=round(sum(max(range(len(lp)), key=lp.__getitem__) == gi for lp, gi in ch) / len(ch), 4),
        )
    if nl:
        sc = min(grid, key=nll_noul)
        out.update(
            noul_total_scale=sc,
            noul_nll_before=round(nll_noul(1.0), 4),
            noul_nll_after=round(nll_noul(sc), 4),
            noul_acc=round(sum((lg > 0) == y for lg, y in nl) / len(nl), 4),
        )
        out["noul_t"] = round(sc / out.get("temp", 1.0), 6)  # the server applies temp first, then noul_t on the resulting logit
    return out


def write_profile(base: Path, fit_result: dict, name: str, out: Path):
    d = json.loads(base.read_text())
    d["name"] = name
    d["description"] = f"Fitted by `openjev calibrate` on {fit_result['n_choice']} choice and {fit_result['n_noul']} yes/no development rows."
    if "temp" in fit_result:
        d["temp"] = fit_result["temp"]
    if "noul_t" in fit_result:
        d["noul_t"] = fit_result["noul_t"]
    d["noul_bias"] = 0.0
    d["fit"] = fit_result
    out.write_text(json.dumps(d, indent=2) + "\n")
