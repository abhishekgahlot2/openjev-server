"""vLLM (or any OpenAI-compatible server that returns logprobs for requested token ids) over one persistent HTTP connection pool.

Exact readout: the request asks for `logprob_token_ids` = the candidate letters and reads them back BY ID, so a candidate is never
confused with a look-alike token and a candidate outside the top-K is never floored. Servers without that extension fall back to
top_logprobs matched by text (the old protocol), where a missing label is floored at -30.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import httpx

from .base import BackendError
from .letters import has_chat_template, header_counter, letter_ids

MISSING = -30.0  # text-matched fallback only: a label outside the top-K


@dataclass(frozen=True)
class VllmOptions:
    model: str | None = None  # None: discovered from GET /models at startup
    api_key: str = "x"
    timeout_s: float = 120.0
    max_connections: int = 64
    retries: int = 2  # on 429 / 503 / transport errors, exponential backoff
    exact: bool | None = None  # None: detected at startup (vLLM's logprob_token_ids extension), else top_logprobs by text
    letter_prefix: str = "auto"
    assistant_prefix: str = ""


def detect_vision(tokenizer: str) -> bool | None:
    """True/False from the model config's vision_config; None when the architecture is unknown to transformers."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(tokenizer, trust_remote_code=False)
        return hasattr(cfg, "vision_config") or "vision_config" in getattr(cfg, "to_dict", dict)()
    except (OSError, ValueError, KeyError) as e:
        logging.getLogger("openjev").info(json.dumps({"vision_detection": f"unknown ({type(e).__name__})"}))
        return None


class VllmBackend:
    def __init__(self, base_url: str, tokenizer: str, options: VllmOptions | None = None):
        from transformers import AutoTokenizer

        options = options or VllmOptions()
        self.base_url = base_url.rstrip("/")
        self.options = options
        self.model = options.model
        self.exact = options.exact
        self.assistant_prefix = options.assistant_prefix
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(options.timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=options.max_connections, max_keepalive_connections=options.max_connections),
            headers={"Authorization": f"Bearer {options.api_key}", "Content-Type": "application/json"},
        )
        self._tok = AutoTokenizer.from_pretrained(tokenizer)
        self._ids, self.letter_prefix = letter_ids(self._tok, options.letter_prefix)
        self._header = header_counter(self._tok, options.assistant_prefix)
        self.chat = has_chat_template(self._tok)
        self.vision = detect_vision(tokenizer)

    # ---- contract

    async def start(self) -> dict:
        """Discover the served model name and whether the server supports the exact readout. Returns what it found."""
        if self.model is None:
            r = await self._client.get("/models")
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            if not models:
                raise BackendError("the backend lists no models")
            self.model = models[0]
        if self.exact is None:
            ids = list(self._ids.values())[:3]
            try:
                top = self._chat_top(await self._post(self._chat_body("Say A, B or C.", ids, exact=True)))
                self.exact = all(f"token_id:{i}" in top for i in ids)
            except (BackendError, KeyError, IndexError, TypeError):
                self.exact = False
        return {"served_model": self.model, "exact_readout": self.exact, "chat_template": self.chat, "letter_prefix": self.letter_prefix, "vision": self.vision}

    def letter_ids(self) -> dict[str, int]:
        return self._ids

    def header_tokens(self, text: str) -> int:
        return self._header(text)

    async def logprobs(self, content: str | list[dict[str, Any]], token_ids: list[int]) -> tuple[list[float], int]:
        if not isinstance(content, str) and self.vision is False:
            raise BackendError("this model has no vision tower: send text, JSON or DOM states, not screenshots")
        if not self.chat and not isinstance(content, str):
            raise BackendError("a model without a chat template cannot take image content")
        if self.chat:
            r = await self._post(self._chat_body(content, token_ids, self.exact))
            top = self._chat_top(r)
        else:  # raw completions: the text goes in as is
            r = await self._post(self._completion_body(content + self.assistant_prefix, token_ids), path="/completions")
            top = self._completion_top(r)
        return self._scores(top, token_ids), self._prompt_tokens(r)

    async def prewarm(self, text: str) -> int:
        if self.chat:
            body = {
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [{"role": "user", "content": text}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            r = await self._post(body)
        else:
            r = await self._post({"model": self.model, "prompt": text, "max_tokens": 1, "temperature": 0}, path="/completions")
        return self._prompt_tokens(r)

    async def passthrough(self, body: dict) -> tuple[int, bytes]:
        """Forward a plain chat completion (thinking forced off). Used by clients that type text through the same endpoint."""
        for k in ("reasoning", "thinking"):
            body.pop(k, None)
        body["chat_template_kwargs"] = {"enable_thinking": False}
        r = await self._client.post("/chat/completions", json=body)
        return r.status_code, r.content

    async def healthy(self) -> bool:
        try:
            r = await self._client.get("/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self):
        await self._client.aclose()

    # ---- request bodies

    def _chat_body(self, content, token_ids: list[int], exact: bool | None) -> dict:
        msgs = [{"role": "user", "content": content}]
        if self.assistant_prefix:
            msgs.append({"role": "assistant", "content": self.assistant_prefix})
        body = {"model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}
        if self.assistant_prefix:
            body.update(add_generation_prompt=False, continue_final_message=True)
        if exact:
            body.update(allowed_token_ids=token_ids, logprob_token_ids=token_ids, return_tokens_as_token_ids=True)
        else:
            body.update(top_logprobs=min(max(20, len(token_ids)), 64), allowed_token_ids=token_ids)
        return body

    def _completion_body(self, prompt: str, token_ids: list[int]) -> dict:
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 1 if self.exact else max(20, len(token_ids))}
        if self.exact:
            body.update(allowed_token_ids=token_ids, logprob_token_ids=token_ids, return_tokens_as_token_ids=True)
        return body

    # ---- responses

    @staticmethod
    def _chat_top(r: dict) -> dict[str, float]:
        """token -> logprob at the first output position of a chat completion."""
        return {t["token"]: t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}

    @staticmethod
    def _completion_top(r: dict) -> dict[str, float]:
        lp = r["choices"][0]["logprobs"]
        return dict(lp["top_logprobs"][0]) if lp.get("top_logprobs") else {}

    @staticmethod
    def _prompt_tokens(r: dict) -> int:
        return int(r.get("usage", {}).get("prompt_tokens", 0))

    def _scores(self, top: dict[str, float], token_ids: list[int]) -> list[float | None]:
        """The candidates' log-probabilities in `token_ids` order: by id when exact, else by label text (bare form preferred over a
        whitespace-padded look-alike). None marks a candidate the server did not score."""
        if self.exact:
            raw = [top.get(f"token_id:{i}") for i in token_ids]
        else:
            inv = {v: k for k, v in self._ids.items()}
            by_text: dict[str, float] = {}
            for tok, lp in top.items():
                k = tok if tok in inv.values() else tok.strip()
                if k not in by_text or tok in inv.values():
                    by_text[k] = lp
            raw = [by_text.get(inv[i], MISSING) for i in token_ids]
        return [None if v is None or not math.isfinite(v) else float(v) for v in raw]

    # ---- transport

    async def _post(self, body: dict, path: str = "/chat/completions") -> dict:
        retries = self.options.retries
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = await self._client.post(path, json=body)
                if r.status_code in (429, 503) and attempt < retries:
                    await asyncio.sleep(0.3 * 2**attempt + random.random() * 0.1)
                    continue
                if r.status_code != 200:
                    raise BackendError(f"backend HTTP {r.status_code}: {r.text[:200]}")
                return r.json()
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = e
                if attempt < retries:
                    await asyncio.sleep(0.3 * 2**attempt)
                    continue
                raise BackendError(f"backend unreachable: {type(e).__name__}: {e}") from None
        raise BackendError(f"backend unreachable: {last}")
