import asyncio

import pytest

from openjev_server.profile import Profile
from openjev_server.readout import BadQuestion, Readout, choice_confidence, compose_chunks, score_confidence, softmax


def test_confidence_formulas():
    assert choice_confidence([1.0]) == 1.0
    assert choice_confidence([0.25] * 4) == 0.0
    assert abs(choice_confidence([0.7, 0.1, 0.1, 0.1]) - (0.7 - 0.25) / 0.75) < 1e-12
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    assert 0.0 <= score_confidence([0.5, 0.0, 0.5]) <= 1.0


def test_compose_chunks_is_a_distribution_and_order_free():
    parts = [([0.7, 0.2, 0.1], 0), ([0.1, 0.6, 0.3], 1)]
    p = compose_chunks(parts, [0.4, 0.6])
    assert abs(sum(p) - 1) < 1e-12 and all(x > 0 for x in p)
    assert abs(p[0] / p[4] - (0.4 * 0.7 / 0.7) / (0.6 * 0.6 / 0.6)) < 1e-12  # winners keep the final ratio


def test_softmax():
    p = softmax([0.0, 0.0])
    assert abs(p[0] - 0.5) < 1e-12


def test_bad_questions(fake_backend):
    ro = Readout(fake_backend, Profile())
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "choice", "instructions": "x", "criteria": {}}}))
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}))
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "choice", "criteria": {"a": None}}}))
    with pytest.raises(BadQuestion):
        asyncio.run(ro.answer_all("s", {"q": {"type": "essay"}}))


def test_many_options_sum_to_one_and_use_two_stages(fake_backend):
    ro = Readout(fake_backend, Profile())
    ans, _ = asyncio.run(ro.answer_all("s", {"q": {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(130)}}}))
    p = ans["q"]["probabilities"]
    assert len(p) == 130 and abs(sum(p.values()) - 1) < 0.01 and ans["q"]["choice"] in p
    assert len(fake_backend.prompts) == 4  # 3 chunks of <= 52 + 1 over the winners


def test_stagger_runs_first_question_alone(fake_backend):
    ro = Readout(fake_backend, Profile(stagger=True, stagger_min_chars=10))
    ans, toks = asyncio.run(ro.answer_all("x" * 100, {"a": {"type": "noul", "instructions": "yes?"}, "b": {"type": "noul", "instructions": "no?"}}))
    assert set(ans) == {"a", "b"} and toks > 0 and 0 <= ans["a"]["noul"] <= 1


def test_noul_calibration_direction(fake_backend):
    ro = Readout(fake_backend, Profile(noul_t=1.0, noul_bias=0.0))
    ans, _ = asyncio.run(ro.answer_all("s", {"q": {"type": "noul", "instructions": "?"}}))
    ro2 = Readout(fake_backend, Profile(noul_t=1.0, noul_bias=3.0))
    ans2, _ = asyncio.run(ro2.answer_all("s", {"q": {"type": "noul", "instructions": "?"}}))
    assert ans2["q"]["noul"] > ans["q"]["noul"]
