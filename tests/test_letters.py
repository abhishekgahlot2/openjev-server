import asyncio

from openjev_server.backends.letters import has_chat_template, header_counter, letter_ids, letter_variants, render
from openjev_server.backends.probe import probe
from openjev_server.profile import Profile
from openjev_server.readout import Readout


def test_letter_variants_and_choice(fake_backend):
    v = letter_variants(fake_backend.tok)
    assert "" in v and len(v[""]) == 52
    ids, pre = letter_ids(fake_backend.tok, "auto")
    assert pre == "" and len(set(ids.values())) == 52


def test_render_with_and_without_assistant_prefix(fake_backend):
    tok = fake_backend.tok
    assert has_chat_template(tok)
    plain = render(tok, "hello")
    pref = render(tok, "hello", "<think>\n\n</think>\n\n")
    assert "hello" in plain and "hello" in pref and pref.rstrip().endswith("</think>") and pref.index("hello") < pref.rstrip().rindex("</think>")
    assert header_counter(tok, "")("abc") < header_counter(tok, "<think>\n\n</think>\n\n")("abc")


def test_probe_reports_letter_mass(fake_backend):
    ro = Readout(fake_backend, Profile())
    rep = asyncio.run(probe(ro, fake_backend, letter_prefix_used="", exact=True, vision=True))
    assert rep["scores_finite"] is True and rep["letter_mass_at_readout_position"] > 0 and "problem" not in rep
