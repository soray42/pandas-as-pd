"""Generation arm: attribute extraction + completion classification."""

from __future__ import annotations

from alias_inertia.generation import classify_generation, extract_first_attribute
from alias_inertia.lexicons import LEXICONS

LEX = {"numpy": LEXICONS["numpy"], "pandas": LEXICONS["pandas"]}


def test_extract_first_attribute():
    assert extract_first_attribute("array([1,2,3])") == "array"
    assert extract_first_attribute("DataFrame({'a':[1]})") == "DataFrame"
    assert extract_first_attribute("nn.Linear(3,3)") == "nn"
    assert extract_first_attribute("") is None
    assert extract_first_attribute("   \n") is None


def test_classify_prior_style():
    c = classify_generation("array([1,2,3])", prior_lib="numpy", bound_lib="pandas", lexicons=LEX)
    assert c["klass"] == "prior_style" and c["attribute"] == "array" and c["prior_match"]


def test_classify_bound_style():
    c = classify_generation("DataFrame({'a':[1]})", prior_lib="numpy", bound_lib="pandas", lexicons=LEX)
    assert c["klass"] == "bound_style" and c["bound_match"]


def test_classify_other():
    c = classify_generation("foobar(1) + 2", prior_lib="numpy", bound_lib="pandas", lexicons=LEX)
    assert c["klass"] == "other"


def test_classify_empty():
    c = classify_generation("", prior_lib="numpy", bound_lib="pandas", lexicons=LEX)
    assert c["klass"] == "empty"


def test_conventional_match_counts_as_bound():
    # conventional: prior == bound; a lexicon match is the correct library, reported as bound_style
    c = classify_generation("arange(10)", prior_lib="numpy", bound_lib="numpy", lexicons=LEX)
    assert c["klass"] == "bound_style"
