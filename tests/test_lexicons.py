"""Lexicon integrity: discriminative, disjoint, blocklist-free, aliases present."""

from __future__ import annotations

from alias_inertia.lexicons import (
    CANONICAL_ALIASES,
    IMPORT_NAMES,
    LEXICONS,
    SHARED_BLOCKLIST,
    SWAP_PAIRS,
    normalize_member,
    verify_lexicons,
)


def test_verify_lexicons_passes():
    verify_lexicons()  # raises on any violation


def test_no_shared_blocklist_tokens():
    for lib, members in LEXICONS.items():
        for m in members:
            assert normalize_member(m) not in SHARED_BLOCKLIST, f"{lib}.{m} is shared/non-discriminative"


def test_continuations_disjoint_across_libs():
    seen = {}
    for lib, members in LEXICONS.items():
        for m in members:
            assert m not in seen, f"{m!r} shared by {seen.get(m)} and {lib}"
            seen[m] = lib


def test_every_lib_has_alias_and_import_name():
    for lib in LEXICONS:
        assert lib in CANONICAL_ALIASES
        assert lib in IMPORT_NAMES


def test_canonical_aliases():
    assert CANONICAL_ALIASES["numpy"] == "np"
    assert CANONICAL_ALIASES["pandas"] == "pd"
    assert CANONICAL_ALIASES["matplotlib.pyplot"] == "plt"
    assert CANONICAL_ALIASES["sklearn"] == "sk"
    assert CANONICAL_ALIASES["xgboost"] == "xgb"
    assert CANONICAL_ALIASES["torch"] == "torch"


def test_first_pair_is_numpy_pandas():
    assert SWAP_PAIRS[0].prior_lib == "numpy"
    assert SWAP_PAIRS[0].other_lib == "pandas"
    assert SWAP_PAIRS[0].treatment_alias == "np"


def test_lexicon_sizes_reasonable():
    for lib, members in LEXICONS.items():
        assert len(members) >= 5, f"{lib} lexicon too small"
        assert len(set(members)) == len(members), f"{lib} has duplicate continuations"
