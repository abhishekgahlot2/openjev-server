"""State text and prompt composition: what the model reads for one lettering of one question.

The layouts are the ones of OpenJev's released helper (helper/shim.py, sha 81a22f1b) and must stay byte-identical: the compat tests
compare the prompts of this module with the helper's on fixtures.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from .backends.base import Backend
from .backends.letters import LETTERS
from .profile import Profile


class State(str):
    """The state as text, plus an optional screenshot (data URL) that rides along to the model as an image."""

    image: str | None = None
    fields: dict[str, Any] | None = None


DROP_KEYS = {
    "box",
    "rect",
    "bbox",
    "bounds",
    "xpath",
    "selector",
    "nonce",
    "n",
    "fingerprint",
    "marker",
    "page_key",
    "elapsed_ms",
    "started_at",
    "latency_ms",
    "executed_ms",
    "usage",
}
INTERACTIVE = {"a", "button", "input", "select", "textarea", "option", "label", "summary"}
ID_KEYS = ("index", "id", "idx", "n", "marker")
STABLE_FIRST = ("page", "elements", "goal", "task", "plan")
HIST_KEYS = ("previous_actions", "history", "recent_actions")


def compact(state: dict, cap: int = 0) -> dict:
    """Drop geometry / id fields; dedupe elements that repeat the same (tag, text, id); keep interactive or labelled elements."""

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in DROP_KEYS}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v[:cap] if cap and isinstance(v, str) and len(v) > cap else v

    st = clean(state)
    els = state.get("elements")
    if isinstance(els, list):
        seen, out = set(), []
        for e in els:
            if not isinstance(e, dict):
                out.append(e)
                continue
            ident = next((str(e[k]) for k in ID_KEYS if e.get(k) is not None), None)
            key = (e.get("tag"), (e.get("text") or "").strip(), ident)
            if ident is not None and key in seen:
                continue
            if e.get("tag") not in INTERACTIVE and not (e.get("text") or e.get("label") or e.get("aria") or e.get("role")):
                continue
            seen.add(key)
            out.append(clean(e))
        st["elements"] = out
    return st


def page_first(state: dict) -> dict:
    return {**{k: state[k] for k in STABLE_FIRST if k in state}, **{k: v for k, v in state.items() if k not in STABLE_FIRST}}


def make_state(state: Any, profile: Profile) -> State:
    """state.screenshot / state.image (data URL or raw base64) becomes the image; the rest is the text state."""
    if profile.compact and isinstance(state, dict):
        state = compact(state, profile.compact_cap)
    if profile.layout == "page_first" and isinstance(state, dict):
        state = page_first(state)
    if isinstance(state, dict):
        for key in ("screenshot", "image"):
            v = state.get(key)
            if isinstance(v, str) and (v.startswith("data:image") or len(v) > 2000):
                rest = {k: x for k, x in state.items() if k != key}
                st = State(json.dumps(rest, ensure_ascii=False) if rest else "(see screenshot)")
                st.image = v if v.startswith("data:image") else "data:image/png;base64," + v
                st.fields = rest
                return st
        return State(json.dumps(state, ensure_ascii=False))
    return State(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------------------------------- prompts


def pad_prefix(prefix: str, pad: int, header_tokens) -> str:
    """Append filler so chat header + prefix is 2 tokens past a multiple of `pad` tokens (the boundary stays inside the shared text)."""
    if not pad:
        return prefix
    n = header_tokens(prefix)
    k = (2 - n) % pad
    if k < 2:
        k += pad
    out = prefix + "\n" + " pad" * k
    for _ in range(3):  # BPE can merge fillers; correct the count
        d = (header_tokens(out) - 2) % pad
        if d == 0:
            break
        out = out + " pad" * (pad - d) if d > pad // 2 else out[: len(out) - 4 * d]
    return out


def prefix_text(state_text: State, profile: Profile, backend: Backend) -> str:
    """The shared part of a text-mode prompt (everything before the question)."""
    return pad_prefix(f"State:\n{state_text}", profile.pad, backend.header_tokens)


def _task(fields: dict, instructions: str) -> str | None:
    """Task line of the screenshot layout: state.task, or instructions.goal read from the structure (JSON or Python literal)."""
    if fields.get("task"):
        return str(fields["task"])
    if instructions.startswith("{"):
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(instructions)
            except (ValueError, SyntaxError):
                continue
            return v.get("goal") if isinstance(v, dict) else None
    return None


def build_prompt(state_text: State, instructions: str, options: list[tuple[str, str]], profile: Profile, backend: Backend):
    """The exact prompt for one lettering. Returns text, or [image, text] content for a screenshot state."""
    lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    image = getattr(state_text, "image", None)
    fields = getattr(state_text, "fields", None) or {}
    task = _task(fields, instructions) if image else None
    if task:
        hist = next((fields[k] for k in HIST_KEYS if fields.get(k)), [])
        hist = (
            "\n".join(f"- {h if isinstance(h, str) else json.dumps(h, ensure_ascii=False)}" for h in (hist if isinstance(hist, list) else [hist])) or "- (none)"
        )
        extra = {k: v for k, v in fields.items() if k not in ("task", *HIST_KEYS)}
        marks = "\n".join(f"[{LETTERS[i]}] {d if d else k}" for i, (k, d) in enumerate(options))
        shown = (
            "The screenshot shows the current page with candidate elements marked by red letters."
            if "task" in fields
            else "The screenshot shows the current page; candidate elements:"
        )
        page = f"{shown}\n{marks}\n"
        task_block = f"Task: {task}\nPrevious actions:\n{hist}\n" + (f"Context: {json.dumps(extra, ensure_ascii=False)}\n" if extra else "")
        prompt = (page + "\n" + task_block if profile.layout == "page_first" else task_block + "\n" + page) + f"\n{instructions} Answer with the letter only."
    else:
        head = f"The screenshot shows the current screen.\nState:\n{state_text}" if image else prefix_text(state_text, profile, backend)
        prompt = head + f"\n\nQuestion: {instructions}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only."
    if image:
        return [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": prompt}]
    return prompt
