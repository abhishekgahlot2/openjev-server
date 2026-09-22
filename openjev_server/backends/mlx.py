"""MLX backend for Apple silicon: the same readout from an mlx-lm model on the Mac. One forward pass per question; with
`prefix_cache=True` the shared 'State:' prefix of a request is prefilled once and copied for the other questions of that step.
MLX is not thread-safe: all model calls go through one lock in a worker thread, so the async server stays responsive.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from typing import Any


class MlxBackend:
    def __init__(self, model_dir: str, *, prefix_cache: bool = True, min_prefix_tokens: int = 64, letter_prefix: str = "auto", assistant_prefix: str = ""):
        import mlx.core as mx
        from mlx_lm import load
        from transformers import AutoTokenizer

        self.mx = mx
        self.model, _ = load(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.prefix_cache = prefix_cache
        self.min_prefix_tokens = min_prefix_tokens
        self._lock = threading.Lock()
        self._prefix: tuple[list[int], Any] | None = None
        from .letters import header_counter, letter_ids, render

        self._ids, self.letter_prefix = letter_ids(self.tok, letter_prefix)
        self.assistant_prefix = assistant_prefix
        self._render = lambda content: render(self.tok, content, assistant_prefix)
        self._header = header_counter(self.tok, assistant_prefix)
        self.vision = False  # mlx-lm text models only

    async def start(self) -> dict:
        return {
            "served_model": "mlx",
            "exact_readout": True,
            "chat_template": bool(getattr(self.tok, "chat_template", None)),
            "letter_prefix": self.letter_prefix,
            "vision": False,
            "prefix_cache": self.prefix_cache,
        }

    def letter_ids(self) -> dict[str, int]:
        return self._ids

    def header_tokens(self, text: str) -> int:
        return self._header(text)

    async def healthy(self) -> bool:
        return True

    async def aclose(self):
        pass

    def _ids_for(self, content) -> list[int]:
        if not isinstance(content, str):
            raise TypeError("the MLX build is text-only: screenshot states are not supported")
        return list(self.tok.encode(self._render(content), add_special_tokens=False))

    def _feed(self, ids: list[int], cache, step: int = 2048):
        mx = self.mx
        while len(ids) > step:
            self.model(mx.array(ids[:step])[None], cache=cache)
            mx.eval([c.state for c in cache])
            ids = ids[step:]
        return self.model(mx.array(ids)[None], cache=cache)[0, -1]

    def _last_logits(self, ids: list[int], content: str):
        mx = self.mx
        cut = content.rfind("\n\nQuestion: ") if self.prefix_cache else -1
        if cut < 0:
            return self.model(mx.array(ids)[None])[0, -1]
        from mlx_lm.models.cache import make_prompt_cache

        shared = self._prefix[0] if self._prefix else None
        if not (shared and ids[: len(shared)] == shared and len(ids) > len(shared)):
            head = self._ids_for(content[:cut])
            n = next((i for i, (x, y) in enumerate(zip(ids, head, strict=False)) if x != y), min(len(ids), len(head)))
            n = min(n, len(ids) - 1)
            if n < self.min_prefix_tokens:
                return self.model(mx.array(ids)[None])[0, -1]
            cache = make_prompt_cache(self.model)
            self._feed(ids[:n], cache)
            mx.eval([c.state for c in cache])
            self._prefix = (ids[:n], cache)
            shared = ids[:n]
        return self._feed(ids[len(shared) :], copy.deepcopy(self._prefix[1]))

    def _logprobs_sync(self, content: str, token_ids: list[int]) -> tuple[list[float], int]:
        mx = self.mx
        ids = self._ids_for(content)
        with self._lock:
            logits = self._last_logits(ids, content).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1)
            vals = [float(lp[i]) for i in token_ids]
            mx.eval(logits)
        return vals, len(ids)

    async def logprobs(self, content: str | list[dict[str, Any]], token_ids: list[int]) -> tuple[list[float], int]:
        return await asyncio.to_thread(self._logprobs_sync, content, token_ids)

    async def prewarm(self, text: str) -> int:
        def run():
            ids = self._ids_for(text)
            with self._lock:
                if self.prefix_cache:
                    from mlx_lm.models.cache import make_prompt_cache

                    cache = make_prompt_cache(self.model)
                    self._feed(ids, cache)
                    self.mx.eval([c.state for c in cache])
                    self._prefix = (ids, cache)
            return len(ids)

        return await asyncio.to_thread(run)
