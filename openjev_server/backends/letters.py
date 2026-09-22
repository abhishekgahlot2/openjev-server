"""Tokenizer helpers shared by the backends: the token id of each single-letter option label, and a chat-header-aware token counter.

Any model: the 52 labels must be single tokens. Most tokenizers have bare "A".."z" as single tokens; SentencePiece-style ones only
have the space-prefixed form (" A"). `letter_ids` tries the bare form first, then the space-prefixed one, and reports which it used.
"""

from __future__ import annotations

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]  # the 52 single-letter option labels
PREFIXES = ("", " ")


def letter_variants(tok) -> dict[str, dict[str, int]]:
    """{prefix: {letter: token id}} for every prefix whose 52 labels are single, unique tokens."""
    out = {}
    for pre in PREFIXES:
        ids = {}
        for L in LETTERS:
            enc = tok.encode(pre + L, add_special_tokens=False)
            if len(enc) != 1:
                ids = None
                break
            ids[L] = enc[0]
        if ids and len(set(ids.values())) == len(ids):
            out[pre] = ids
    return out


def letter_ids(tok, prefer: str = "auto") -> tuple[dict[str, int], str]:
    """(ids, prefix used). prefer: "" or " " forces a variant; "auto" takes the bare form when it exists, else the space form."""
    variants = letter_variants(tok)
    if not variants:
        raise ValueError("neither 'A' nor ' A' are single tokens for this tokenizer: this model cannot serve a letter readout")
    if prefer != "auto":
        if prefer not in variants:
            raise ValueError(f"letter prefix {prefer!r} is not a single token for this tokenizer; available: {sorted(variants)}")
        return variants[prefer], prefer
    pre = "" if "" in variants else " "
    return variants[pre], pre


def has_chat_template(tok) -> bool:
    return bool(getattr(tok, "chat_template", None))


def render(tok, content: str, assistant_prefix: str = "") -> str:
    """The exact text a chat backend prefills: user turn + generation prompt (+ an optional assistant prefix continued in place).
    Tokenizers without a chat template get the raw content."""
    if not has_chat_template(tok):
        return content + assistant_prefix
    msgs = [{"role": "user", "content": content}]
    if assistant_prefix:
        msgs.append({"role": "assistant", "content": assistant_prefix})
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def header_counter(tok, assistant_prefix: str = ""):
    """Returns f(text) = token count of header + text (+ assistant prefix), what the backend actually prefills."""
    marker = "\u0007X\u0007"
    rendered = render(tok, marker, assistant_prefix)
    head, _, tail = rendered.partition(marker)
    return lambda text: len(tok.encode(head + text + tail, add_special_tokens=False))
