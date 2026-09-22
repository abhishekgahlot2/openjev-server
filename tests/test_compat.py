"""Byte-for-byte compatibility with the released helper: same prompts, same probabilities, on the published profile."""

import asyncio
import json

import pytest

from openjev_server.profile import Profile
from openjev_server.readout import Readout

STATES = [
    "Customer message: I was charged twice for my order last week and nobody has replied.",
    {
        "page": {"url": "https://example.com/checkout", "title": "Checkout", "text": "Shipping address " * 40},
        "elements": [f"[{i}] button Option {i}" for i in range(12)],
        "recent_actions": [{"action": "CLICK", "text": "Cart"}],
    },
    {"task": "Book a flight", "previous_actions": ["click Flights"], "screenshot": "data:image/png;base64," + "A" * 3000},
]
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "shipping": "Delivery problems", "technical": None},
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today", "right now"]},
    "dict_instr": {"type": "choice", "instructions": {"goal": "find flights", "rules": "pick one"}, "criteria": {str(i): f"opt {i}" for i in range(7)}},
    "many": {"type": "choice", "instructions": "Pick the element.", "criteria": {f"e{i}": f"element {i}" for i in range(120)}},
}


def shim_answers(shim, state, questions):
    st = shim.with_image(state)
    return {qid: shim.ANSWER[q["type"]](st, q)[0] for qid, q in questions.items()}


@pytest.mark.parametrize("state", STATES, ids=["text", "dict", "screenshot"])
def test_answers_equal_released_helper(shim, fake_backend, state):
    ro = Readout(fake_backend, Profile())
    ours, _ = asyncio.run(ro.answer_all(state, QUESTIONS))
    theirs = shim_answers(shim, state, QUESTIONS)
    assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True)


def test_prompts_equal_released_helper(shim, fake_backend):
    ro = Readout(fake_backend, Profile())
    asyncio.run(ro.answer_all(STATES[1], {"route": QUESTIONS["route"], "urgency": QUESTIONS["urgency"]}))
    shim._seen_prompts.clear()
    shim_answers(shim, STATES[1], {"route": QUESTIONS["route"], "urgency": QUESTIONS["urgency"]})
    assert sorted(map(json.dumps, fake_backend.prompts)) == sorted(map(json.dumps, shim._seen_prompts))


def test_perms_average_matches(shim, fake_backend, monkeypatch):
    monkeypatch.setattr(shim, "PERMS", 4)
    ro = Readout(fake_backend, Profile(perms=4))
    ours, _ = asyncio.run(ro.answer_all(STATES[0], {"route": QUESTIONS["route"], "dict_instr": QUESTIONS["dict_instr"]}))
    theirs = shim_answers(shim, STATES[0], {"route": QUESTIONS["route"], "dict_instr": QUESTIONS["dict_instr"]})
    assert json.dumps(ours, sort_keys=True) == json.dumps(theirs, sort_keys=True)
