"""What a backend must provide. Backends only answer one question: given a prompt (text, or image + text) and a list of candidate
token ids, what are the model's log-probabilities of exactly those tokens at the first output position."""

from __future__ import annotations

from typing import Any, Protocol


class BackendError(RuntimeError):
    """The model server failed or refused a request. Reported as HTTP 502."""


class Backend(Protocol):
    async def start(self) -> dict:
        """Connect / load and report what was found (served model, exact readout, chat template, letter prefix, vision)."""

    async def logprobs(self, content: str | list[dict[str, Any]], token_ids: list[int]) -> tuple[list[float], int]:
        """Log-probabilities of exactly `token_ids` at the first output position, and the prompt token count."""

    async def prewarm(self, text: str) -> int:
        """One-token prefill of a shared prefix so following calls only pay for their tails. Returns the prompt token count."""

    def letter_ids(self) -> dict[str, int]:
        """Token id of each single-letter option label for this model's tokenizer."""

    def header_tokens(self, text: str) -> int:
        """Token count of chat header + text, used only for prefix padding."""

    async def healthy(self) -> bool:
        """Liveness of the model server, for GET /readyz."""
