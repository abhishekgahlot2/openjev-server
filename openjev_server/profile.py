"""A profile is everything that turns raw label scores into answers: calibration constants and prompt layout switches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Profile:
    """Everything that turns raw scores into answers. `openjev` is the profile of the published numbers; a new model gets its own
    constants from `openjev calibrate` (fit on development rows, never on a test set)."""

    name: str = "openjev"
    temp: float = 0.85  # choice / score: softmax over letter scores divided by this
    noul_t: float = 1.829074  # yes/no: logit(p_yes) / noul_t + noul_bias
    noul_bias: float = 0.0
    perms: int = 1  # >1 averages over that many option orderings (about 2.6x the latency at 4 on short prompts)
    instr_style: str = "pyrepr"  # how dict instructions are rendered: pyrepr (str(dict)) or json
    stagger: bool = True  # long states: run the first question alone so its prefix is cached before the others
    stagger_min_chars: int = 16000
    compact: bool = False  # drop geometry / id fields and duplicate elements from dict states (lossless for the model's evidence)
    compact_cap: int = 0  # >0 also truncates strings (lossy; off by default)
    layout: str = ""  # "page_first": stable page / elements / goal first, mutable history last (longer shared prefixes across steps)
    pad: int = 0  # >0 pads the shared prefix to a multiple of this many tokens (cache-block alignment)
    letter_prefix: str = "auto"  # which single-token form the labels use: "" (bare "A"), " " (space-prefixed), or auto (bare if the tokenizer has it)
    assistant_prefix: str = ""  # text placed at the start of the assistant turn before the readout position (e.g. an empty think block for reasoning models)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
