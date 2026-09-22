"""The vLLM backend against a fake OpenAI-compatible server (httpx.MockTransport): exact readout by id, text fallback, raw completions
for models without a chat template, retries."""

import asyncio
import json

import httpx
import pytest

from openjev_server.backends.base import BackendError
from openjev_server.backends.vllm import MISSING, VllmBackend, VllmOptions


def make_backend(tokenizer_path, handler, **opts):
    b = VllmBackend("http://fake/v1", tokenizer_path, VllmOptions(model="m", **opts))
    b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake/v1")
    return b


def chat_response(top, prompt_tokens=7):
    return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}], "usage": {"prompt_tokens": prompt_tokens}})


def test_exact_readout_by_id(tokenizer_path):
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen.update(body)
        return chat_response([{"token": f"token_id:{i}", "logprob": -0.5 * n} for n, i in enumerate(body["logprob_token_ids"])])

    b = make_backend(tokenizer_path, handler, exact=True)
    ids = [b.letter_ids()[c] for c in "ABC"]
    scores, tokens = asyncio.run(b.logprobs("hello", ids))
    assert scores == [0.0, -0.5, -1.0] and tokens == 7
    assert seen["allowed_token_ids"] == ids and seen["max_tokens"] == 1 and seen["messages"][0]["content"] == "hello"


def test_text_fallback_prefers_bare_label_and_floors_missing(tokenizer_path):
    def handler(request):
        return chat_response([{"token": " A", "logprob": -9.0}, {"token": "A", "logprob": -0.1}, {"token": "B\n", "logprob": -1.0}])

    b = make_backend(tokenizer_path, handler, exact=False)
    ids = [b.letter_ids()[c] for c in "ABC"]
    scores, _ = asyncio.run(b.logprobs("hello", ids))
    assert scores == [-0.1, -1.0, MISSING]


def test_raw_completions_without_chat_template(tokenizer_path):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"logprobs": {"top_logprobs": [{f"token_id:{i}": -0.2 for i in seen["logprob_token_ids"]}]}}], "usage": {"prompt_tokens": 3}}
        )

    b = make_backend(tokenizer_path, handler, exact=True, assistant_prefix="<think></think>")
    b.chat = False
    ids = [b.letter_ids()[c] for c in "AB"]
    scores, tokens = asyncio.run(b.logprobs("raw text", ids))
    assert scores == [-0.2, -0.2] and tokens == 3
    assert seen["path"].endswith("/completions") and seen["prompt"] == "raw text<think></think>"
    with pytest.raises(BackendError):
        asyncio.run(b.logprobs([{"type": "text", "text": "x"}], ids))


def test_retries_then_fails(tokenizer_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="busy")

    b = make_backend(tokenizer_path, handler, exact=True, retries=1)
    with pytest.raises(BackendError, match="503"):
        asyncio.run(b.logprobs("x", [1]))
    assert len(calls) == 2


def test_start_discovers_model_and_exact_support(tokenizer_path):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "served-name"}]})
        body = json.loads(request.content)
        return chat_response([{"token": f"token_id:{i}", "logprob": -1.0} for i in body["logprob_token_ids"]])

    b = make_backend(tokenizer_path, handler)
    b.model = None
    found = asyncio.run(b.start())
    assert found["served_model"] == "served-name" and found["exact_readout"] is True and found["letter_prefix"] == b.letter_prefix
