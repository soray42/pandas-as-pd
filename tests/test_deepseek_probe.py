"""DeepSeek probe: pure-function correctness (parsing, prompt construction, method selection).

No network is touched; the HTTP client itself is exercised only against the live API in
scripts/run_deepseek.py. These tests cover the deterministic glue that turns a stimulus into a
task prompt and a response into a labelled record.
"""

from __future__ import annotations

import random

import pytest

from alias_inertia import deepseek_probe as dp
from alias_inertia.lexicons import SWAP_PAIRS
from alias_inertia.stimuli import build_stimulus

PAIR = SWAP_PAIRS[0]  # numpy <-> pandas


def _meta(condition, alias="zz", count_tokens=None):
    ct = count_tokens or (lambda s: len(s.split()))
    return build_stimulus(
        pair=PAIR, condition=condition, depth_tokens=0, template_id="t1", rep=0,
        count_tokens=ct, seed=1, non_canonical_alias=alias,
    ).meta


def test_context_code_strips_use_site():
    meta = _meta("swapped")
    code = dp.context_code(meta)
    assert code.endswith("import pandas as np")  # ends at the import, not the np. use site
    assert not code.rstrip().endswith("np.")


def test_parse_attribute():
    assert dp.parse_attribute("array(5)") == "array"
    assert dp.parse_attribute("  DataFrame()") == "DataFrame"
    assert dp.parse_attribute("`arange`(3)") == "arange"
    assert dp.parse_attribute("") is None
    assert dp.parse_attribute("== 3") is None
    # chat-model wrappers: code fence and a re-echoed "alias." prefix
    assert dp.parse_attribute("```python\nDataFrame()") == "DataFrame"
    assert dp.parse_attribute("zz.DataFrame()", alias="zz") == "DataFrame"
    assert dp.parse_attribute("np.arange(3)", alias="np") == "arange"


def test_match_library():
    cands = {"numpy": "numpy", "pandas": "pandas"}
    assert dp.match_library("it is pandas", cands) == "pandas"
    assert dp.match_library("numpy, definitely", cands) == "numpy"
    assert dp.match_library("no idea", cands) is None


def test_parse_choice_from_content_and_token():
    mk = lambda content, ft=None: dp.ChatResult(
        content=content, reasoning_content="", finish_reason="stop",
        first_token=ft, first_token_logprob=None, top_logprobs=[], usage={},
    )
    assert dp.parse_choice(mk("A")) == "A"
    assert dp.parse_choice(mk("b")) == "B"
    assert dp.parse_choice(mk("The answer is B.")) == "B"
    assert dp.parse_choice(mk("", ft="A")) == "A"  # fall back to first logprob token
    assert dp.parse_choice(mk("no letter here")) is None


def test_forced_choice_messages_mapping_and_options():
    meta = _meta("swapped")
    rng = random.Random(0)
    msgs, info = dp.forced_choice_messages(meta, "arange", "DataFrame", rng)
    assert info["distractor_letter"] != info["bound_letter"]
    assert {info["distractor_letter"], info["bound_letter"]} == {"A", "B"}
    content = msgs[0]["content"]
    assert "np.arange" in content and "np.DataFrame" in content
    # the labelled options match the letters
    assert info["option_a"].endswith("arange") or info["option_a"].endswith("DataFrame")


def test_salience_note_states_binding():
    meta = _meta("swapped")
    note = dp.salience_note(meta)
    assert "np" in note and "pandas" in note
    # the note is prepended only when passed
    plain = dp.generation_messages(meta)[0]["content"]
    cued = dp.generation_messages(meta, note=note)[0]["content"]
    assert cued.endswith(plain)
    assert cued.startswith(note)


def test_pick_methods_well_posed():
    # numpy <-> pandas are installed in the test env; distractor must be absent on bound lib.
    from alias_inertia.validity import library_available

    if not (library_available("numpy") and library_available("pandas")):
        pytest.skip("numpy/pandas not importable")
    rng = random.Random(0)
    picked = dp.pick_methods("numpy", "pandas", rng)  # distractor numpy, bound pandas
    assert picked is not None
    distractor, bound = picked
    from alias_inertia.validity import resolves_on
    assert resolves_on(bound, "pandas")["exists"] is True
    assert resolves_on(distractor, "pandas")["exists"] is False
