"""The one-pass readout: letter scores -> calibrated probabilities -> answers (choice / score / noul).

The calibration math, the two-stage fallback above 52 options and the confidence formulas are the ones of OpenJev's released helper
(helper/shim.py, sha 81a22f1b): with the same profile constants this module produces the same answers, and tests/test_compat.py
holds that equality on fixtures.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .backends.base import Backend
from .backends.letters import LETTERS
from .profile import Profile
from .prompt import State, build_prompt, make_state, prefix_text

MAX_ONE_PASS = len(LETTERS)


class ReadoutIncomplete(RuntimeError):
    """The backend returned no finite score for one or more candidates. Never floored: the answer would be a guess."""


class BadQuestion(ValueError):
    """A question the API cannot answer (wrong type, missing instructions, bad criteria). Reported as HTTP 422."""


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def choice_confidence(p: list[float]) -> float:
    """TypeSafe's official formula (system-one-adapter-python confidence_metrics.py)."""
    if len(p) == 1:
        return 1.0
    u = 1.0 / len(p)
    return max(0.0, (max(p) - u) / (1.0 - u))


def score_confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2
    umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)


def compose_chunks(parts: list[tuple[list[float], int]], final: list[float]) -> list[float]:
    """Two-stage distribution above 52 options: p(i) is proportional to p_final(chunk of i) * p_chunk(i) / p_chunk(winner of that chunk)."""
    raw = [final[c] * pi / p[w] for c, (p, w) in enumerate(parts) for pi in p]
    s = sum(raw)
    return [v / s for v in raw]


@dataclass
class Readout:
    backend: Backend
    profile: Profile
    stats: dict[str, int] = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0})

    async def _once(self, state_text: State, instructions: str, options: list[tuple[str, str]]) -> tuple[list[float], int]:
        """One lettering, one prefill: temperature-scaled probabilities aligned with `options`."""
        ids = self.backend.letter_ids()
        allowed = [ids[LETTERS[i]] for i in range(len(options))]
        if len(set(allowed)) != len(allowed):
            raise ReadoutIncomplete("candidate token ids are not unique for this tokenizer")
        content = build_prompt(state_text, instructions, options, self.profile, self.backend)
        raw, tokens = await self.backend.logprobs(content, allowed)
        if len(raw) != len(allowed) or any(v is None or not math.isfinite(v) for v in raw):
            n = sum(1 for v in raw if v is None or not math.isfinite(v))
            raise ReadoutIncomplete(f"{n} of {len(allowed)} candidate scores missing or not finite")
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += tokens
        return softmax([v / self.profile.temp for v in raw]), tokens

    async def readout(self, state_text: State, instructions: str, options: list[tuple[str, str]]) -> tuple[list[float], int]:
        """Averages over `profile.perms` letterings; probabilities stay aligned with the caller's option order."""
        if self.profile.perms <= 1:
            return await self._once(state_text, instructions, options)
        orders = []
        for j in range(self.profile.perms):
            order = list(range(len(options)))
            random.Random(j).shuffle(order)
            orders.append(order)
        results = await asyncio.gather(*(self._once(state_text, instructions, [options[i] for i in order]) for order in orders))
        acc = [0.0] * len(options)
        tokens = 0
        for order, (p, t) in zip(orders, results, strict=True):
            tokens += t
            for pos, i in enumerate(order):
                acc[i] += p[pos] / self.profile.perms
        return acc, tokens

    async def distribution(self, state_text: State, instructions: str, opts: list[tuple[str, str]]) -> tuple[list[float], int]:
        """Up to 52 options in one pass; above that per-chunk readouts (run concurrently), then one over the chunk winners, composed."""
        if len(opts) <= MAX_ONE_PASS:
            return await self.readout(state_text, instructions, opts)
        k = -(-len(opts) // MAX_ONE_PASS)
        size = -(-len(opts) // k)
        chunks = [opts[i : i + size] for i in range(0, len(opts), size)]
        chunk_results = await asyncio.gather(*(self.readout(state_text, instructions, chunk) for chunk in chunks))
        parts = [(p, max(range(len(p)), key=p.__getitem__)) for p, _ in chunk_results]
        tokens = sum(t for _, t in chunk_results)
        final, t = await self.readout(state_text, instructions, [chunk[w] for chunk, (_, w) in zip(chunks, parts, strict=True)])
        return compose_chunks(parts, final), tokens + t

    # ---- answer types

    def _instr(self, q: dict) -> str:
        i = q.get("instructions")
        if i is None:
            raise BadQuestion("instructions is required")
        if isinstance(i, str):
            return i
        return str(i) if self.profile.instr_style == "pyrepr" else json.dumps(i, ensure_ascii=False)

    @staticmethod
    def _desc(v) -> str:
        return "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))

    async def answer_choice(self, state_text: State, q: dict) -> tuple[dict, int]:
        crit = q.get("criteria")
        if not isinstance(crit, dict) or not crit:
            raise BadQuestion("choice.criteria must be a non-empty map of option -> description|null")
        opts = [(k, self._desc(v)) for k, v in crit.items()]
        p, tokens = await self.distribution(state_text, self._instr(q), opts)
        choice = opts[max(range(len(p)), key=lambda i: p[i])][0]  # winner picked before rounding
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": {k: round(v, 4) for (k, _), v in zip(opts, p, strict=True)},
            "confidence": round(choice_confidence(p), 4),
        }, tokens

    async def answer_score(self, state_text: State, q: dict) -> tuple[dict, int]:
        levels = q.get("criteria")
        if not isinstance(levels, list) or len(levels) < 2:
            raise BadQuestion("score.criteria must be an ordered array of at least two levels")
        opts = [(str(i), self._desc(level)) for i, level in enumerate(levels)]
        p, tokens = await self.distribution(state_text, self._instr(q) + " Rate along the ordered levels below (lowest first).", opts)
        return {
            "type": "score",
            "score": round(sum(i * pi for i, pi in enumerate(p)), 4),
            "legend": {str(i): level for i, level in enumerate(levels)},
            "probabilities": {str(i): round(pi, 4) for i, pi in enumerate(p)},
            "confidence": round(score_confidence(p), 4),
        }, tokens

    async def answer_noul(self, state_text: State, q: dict) -> tuple[dict, int]:
        crit = q.get("criteria") or {}
        yes = self._desc(crit.get("true")) or "The statement is true."
        no = self._desc(crit.get("false")) or "The statement is false."
        p, tokens = await self.distribution(state_text, self._instr(q), [("yes", yes), ("no", no)])
        py = min(max(p[0], 1e-4), 1 - 1e-4)
        z = math.log(py / (1 - py)) / self.profile.noul_t + self.profile.noul_bias
        return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}, tokens

    ANSWER: ClassVar[dict[str, str]] = {"choice": "answer_choice", "score": "answer_score", "noul": "answer_noul"}

    async def answer(self, state_text: State, q: dict) -> tuple[dict, int]:
        kind = q.get("type") if isinstance(q, dict) else None
        if kind not in self.ANSWER:
            raise BadQuestion("type must be one of choice, score, noul")
        return await getattr(self, self.ANSWER[kind])(state_text, q)

    async def answer_all(self, state: Any, questions: dict[str, dict]) -> tuple[dict[str, dict], int]:
        """Every question against one state. Questions run concurrently; on a long state the first runs alone so its prefix is cached first."""
        state_text = make_state(state, self.profile)
        items = list(questions.items())
        for qid, q in items:
            if not isinstance(q, dict) or q.get("type") not in self.ANSWER:
                raise BadQuestion(f"questions.{qid}.type must be one of choice, score, noul")
        results: list[tuple[dict, int]] = []
        if self.profile.stagger and len(state_text) >= self.profile.stagger_min_chars and len(items) > 1:
            results.append(await self.answer(state_text, items[0][1]))
            results += list(await asyncio.gather(*(self.answer(state_text, q) for _, q in items[1:])))
        else:
            results = list(await asyncio.gather(*(self.answer(state_text, q) for _, q in items)))
        return {qid: a for (qid, _), (a, _) in zip(items, results, strict=True)}, int(sum(t for _, t in results))

    async def prewarm(self, state: Any) -> int:
        return await self.backend.prewarm(prefix_text(make_state(state, self.profile), self.profile, self.backend))
