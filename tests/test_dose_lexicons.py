"""Dose-axis lexicon/pair integrity for the full run."""

from __future__ import annotations

from alias_inertia.lexicons import (
    ALIAS_PRIOR_TIER,
    SWAP_PAIRS,
    TIER_RANK,
    get_pair,
    verify_lexicons,
)


def test_pairs_span_all_tiers():
    assert len(SWAP_PAIRS) >= 4
    assert {p.tier for p in SWAP_PAIRS} == {"very_common", "common", "rare"}


def test_each_pair_tier_matches_treatment_alias():
    for p in SWAP_PAIRS:
        assert p.tier == ALIAS_PRIOR_TIER[p.treatment_alias]
        assert p.tier_rank == TIER_RANK[p.tier]


def test_all_six_aliases_used_as_treatment():
    aliases = {p.treatment_alias for p in SWAP_PAIRS}
    assert {"np", "pd", "torch", "plt", "sk", "xgb"} <= aliases


def test_get_pair_by_name():
    p = get_pair("torch__sklearn")
    assert p.prior_lib == "torch" and p.other_lib == "sklearn" and p.treatment_alias == "torch"


def test_verify_lexicons_still_passes_after_extension():
    verify_lexicons()
