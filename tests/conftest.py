import hashlib
import os

import pytest


def _tokenizer_dir():
    """OPENJEV_TEST_TOKENIZER, else the published OpenJev tokenizer from the Hub (a few MB); None when neither is reachable."""
    if os.environ.get("OPENJEV_TEST_TOKENIZER"):
        return os.environ["OPENJEV_TEST_TOKENIZER"]
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download("openjev/openjev-MLX-4bit", allow_patterns=["tokenizer*.json", "chat_template*", "config.json"])
    except (OSError, ValueError):  # no network, no Hub access
        return None


def _shim_path():
    """OPENJEV_TEST_SHIM, else the released helper/shim.py from the OpenJev model repo; None when neither is reachable."""
    if os.environ.get("OPENJEV_TEST_SHIM"):
        return os.environ["OPENJEV_TEST_SHIM"]
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download("openjev/openjev", "helper/shim.py")
    except (OSError, ValueError):  # no network, no Hub access
        return None


TOKENIZER = _tokenizer_dir()
SHIM = _shim_path()


def fake_scores(content, token_ids):
    """Deterministic pseudo log-probabilities from the prompt text and the candidate ids: the same prompt gives the same scores."""
    text = content if isinstance(content, str) else "".join(c.get("text", "") or c.get("image_url", {}).get("url", "")[:64] for c in content)
    out = []
    for i in token_ids:
        h = int(hashlib.sha256(f"{text}|{i}".encode()).hexdigest()[:8], 16)
        out.append(-0.05 - 6.0 * (h / 0xFFFFFFFF))
    return out


class FakeBackend:
    """Answers the readout's one question from fake_scores; records every prompt it saw."""

    def __init__(self):
        from transformers import AutoTokenizer

        from openjev_server.backends.letters import header_counter, letter_ids

        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self._ids, self.letter_prefix = letter_ids(self.tok)
        self._header = header_counter(self.tok)
        self.prompts = []
        self.vision = True

    def letter_ids(self):
        return self._ids

    def header_tokens(self, text):
        return self._header(text)

    async def healthy(self):
        return True

    async def logprobs(self, content, token_ids):
        self.prompts.append(content)
        return fake_scores(content, token_ids), 100 + len(str(content)) // 4

    async def prewarm(self, text):
        return len(text) // 4

    async def start(self):
        return {"served_model": "fake", "exact_readout": True, "chat_template": True, "letter_prefix": self.letter_prefix, "vision": True}


@pytest.fixture(scope="session")
def tokenizer_path():
    if not TOKENIZER or not os.path.exists(os.path.join(TOKENIZER, "tokenizer.json")):
        pytest.skip("no tokenizer for the compat tests (set OPENJEV_TEST_TOKENIZER or allow Hub downloads)")
    return TOKENIZER


@pytest.fixture
def fake_backend(tokenizer_path):
    return FakeBackend()


@pytest.fixture(scope="session")
def shim(tokenizer_path):
    """The released helper (shim.py 81a22f1b), imported with the published profile and its model client replaced by fake_scores."""
    if not SHIM or not os.path.exists(SHIM):
        pytest.skip("released shim.py not available (set OPENJEV_TEST_SHIM or allow Hub downloads)")
    import importlib.util
    import types

    for k, v in {
        "READOUT_T": "0.85",
        "READOUT_NOUL_T": "1.829074",
        "READOUT_NOUL_BIAS": "0",
        "READOUT_TARGETED": "1",
        "READOUT_INSTR_STYLE": "pyrepr",
        "SHIM_STAGGER": "1",
        "TOKENIZER": TOKENIZER,
        "READOUT_PERMS": os.environ.get("SHIM_TEST_PERMS", "1"),
    }.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("released_shim", SHIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = []

    def create(model=None, max_tokens=1, temperature=0, logprobs=False, messages=None, extra_body=None, **kw):
        content = messages[0]["content"]
        want = (extra_body or {}).get("logprob_token_ids") or []
        seen.append(content)
        vals = fake_scores(content, want)
        top = [types.SimpleNamespace(token=f"token_id:{i}", logprob=v) for i, v in zip(want, vals, strict=True)]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(logprobs=types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)]))],
            usage=types.SimpleNamespace(prompt_tokens=100 + len(str(content)) // 4),
        )

    mod.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)), base_url="fake://")
    mod._seen_prompts = seen
    return mod
