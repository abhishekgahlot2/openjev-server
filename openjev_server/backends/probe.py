"""Startup probe: what this model can do for a letter readout. Runs one tiny question and reports letter mass at the readout
position, so an incompatible setup (a model that always thinks first, a template that swallows the turn) fails loudly at start
instead of quietly returning noise."""

from __future__ import annotations

import math

PROBE_STATE = "The customer writes: my parcel never arrived and I want a refund."
PROBE_Q = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": None, "shipping": None, "technical": None}}
MIN_LETTER_MASS = 1e-3


async def probe(readout, backend, *, letter_prefix_used: str, exact: bool | None, vision: bool | None) -> dict:
    from ..prompt import build_prompt, make_state
    from .letters import LETTERS

    st = make_state(PROBE_STATE, readout.profile)
    opts = [(k, "") for k in PROBE_Q["criteria"]]
    content = build_prompt(st, PROBE_Q["instructions"], opts, readout.profile, backend)
    ids = backend.letter_ids()
    allowed = [ids[LETTERS[i]] for i in range(len(opts))]
    raw, tokens = await backend.logprobs(content, allowed)
    finite = [v for v in raw if v is not None and math.isfinite(v)]
    mass = sum(math.exp(v) for v in finite) if finite else 0.0
    report = {
        "letter_prefix": letter_prefix_used,
        "exact_readout": exact,
        "vision": vision,
        "probe_prompt_tokens": tokens,
        "letter_mass_at_readout_position": round(mass, 6),
        "scores_finite": len(finite) == len(allowed),
    }
    if len(finite) != len(allowed):
        report["problem"] = (
            "the backend returned no score for some candidate letters (does the server allow logprobs for requested token ids? vLLM needs --max-logprobs >= 52)"
        )
    elif mass < MIN_LETTER_MASS:
        report["problem"] = (
            f"the model puts only {mass:.2e} of its probability on the option letters at the readout position: it is probably writing something "
            "else first (a think block, a preamble). Try --assistant-prefix (e.g. an empty think block) or a template that answers directly."
        )
    return report
