"""Tests for alias_inertia.mech.patching: GROUPS, positions_for, patch_run, run_m2.

CPU tests (GROUPS tuple, positions_for with synthetic stimuli, hook logic via fake model)
run without model download or GPU. Integration tests with a real HookedTransformer are
skipped unless model weights can be loaded.
"""

from __future__ import annotations

import math

import pytest
import torch

try:
    from alias_inertia.mech.patching import (
        GROUPS,
        _SRC_NAMES_FILTER,
        _cache_source_resids,
        patch_run,
        positions_for,
        run_m2,
    )
    from alias_inertia.mech.proxy import ProxyLexicon
    from alias_inertia.mech.stimuli_mech import MechStimulus
except ImportError as _err:
    pytest.skip(f"alias_inertia.mech.patching not importable: {_err}", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lex(prior_ids=(0,), bound_ids=(1,)):
    return ProxyLexicon(
        prior_lib="numpy",
        bound_lib="pandas",
        prior_ids=list(prior_ids),
        bound_ids=list(bound_ids),
        prior_strings=[],
        bound_strings=[],
        dropped=[],
        sha256="",
    )


def _stim(
    condition="swapped",
    token_ids=None,
    *,
    import_span=(0, 4),
    filler_span=(4, 4),
    use_alias_pos=4,
    final_pos=5,
    depth=0,
    base_id="b0",
):
    if token_ids is None:
        token_ids = list(range(6))
    return MechStimulus(
        base_id=base_id,
        stimulus_id=f"{base_id}_{condition}",
        pair_name="numpy__pandas",
        condition=condition,
        alias="np",
        depth=depth,
        template_id="t1",
        rep=0,
        use_var="x",
        prompt="import pandas as np\nnp.",
        token_ids=token_ids,
        import_span=import_span,
        filler_span=filler_span,
        use_alias_pos=use_alias_pos,
        final_pos=final_pos,
        prompt_sha256="",
    )


# ---------------------------------------------------------------------------
# GROUPS constant
# ---------------------------------------------------------------------------

def test_groups_is_tuple():
    assert isinstance(GROUPS, tuple)


def test_groups_contains_required_names():
    assert "import_span" in GROUPS
    assert "use_alias" in GROUPS
    assert "final_pos" in GROUPS
    assert "filler_span" in GROUPS


def test_groups_no_duplicates():
    assert len(GROUPS) == len(set(GROUPS))


# ---------------------------------------------------------------------------
# _SRC_NAMES_FILTER (pure Python, no model)
# ---------------------------------------------------------------------------

def test_src_names_filter_accepts_resid_post():
    assert _SRC_NAMES_FILTER("blocks.0.hook_resid_post")
    assert _SRC_NAMES_FILTER("blocks.7.hook_resid_post")


def test_src_names_filter_accepts_ln_final_scale():
    assert _SRC_NAMES_FILTER("ln_final.hook_scale")


def test_src_names_filter_rejects_resid_pre():
    assert not _SRC_NAMES_FILTER("blocks.0.hook_resid_pre")


def test_src_names_filter_rejects_hook_z():
    assert not _SRC_NAMES_FILTER("blocks.0.attn.hook_z")


# ---------------------------------------------------------------------------
# positions_for with synthetic MechStimulus
# ---------------------------------------------------------------------------

def test_positions_for_import_span():
    stim = _stim(import_span=(0, 4))
    assert positions_for(stim, "import_span") == [0, 1, 2, 3]


def test_positions_for_import_span_empty_when_zero_length():
    stim = _stim(import_span=(2, 2))
    assert positions_for(stim, "import_span") == []


def test_positions_for_use_alias():
    stim = _stim(use_alias_pos=4)
    assert positions_for(stim, "use_alias") == [4]


def test_positions_for_final_pos():
    stim = _stim(final_pos=5)
    assert positions_for(stim, "final_pos") == [5]


def test_positions_for_filler_span_nonempty():
    stim = _stim(
        token_ids=list(range(12)),
        import_span=(0, 4),
        filler_span=(4, 9),
        use_alias_pos=9,
        final_pos=10,
    )
    assert positions_for(stim, "filler_span") == [4, 5, 6, 7, 8]


def test_positions_for_filler_span_empty_at_depth_zero():
    stim = _stim(filler_span=(4, 4))
    assert positions_for(stim, "filler_span") == []


def test_positions_for_unknown_group_raises():
    stim = _stim()
    with pytest.raises(ValueError, match="unknown group"):
        positions_for(stim, "nonexistent_group")


def test_positions_for_all_groups_return_lists():
    stim = _stim(
        token_ids=list(range(12)),
        import_span=(0, 4),
        filler_span=(4, 9),
        use_alias_pos=9,
        final_pos=10,
    )
    for group in GROUPS:
        result = positions_for(stim, group)
        assert isinstance(result, list)
        assert all(isinstance(p, int) for p in result)


def test_positions_for_within_token_ids_bounds():
    n = 10
    stim = _stim(
        token_ids=list(range(n)),
        import_span=(0, 3),
        filler_span=(3, 7),
        use_alias_pos=7,
        final_pos=8,
    )
    for group in GROUPS:
        for pos in positions_for(stim, group):
            assert 0 <= pos < n


# ---------------------------------------------------------------------------
# Fake model for patch_run / run_m2 tests (no download)
# ---------------------------------------------------------------------------

def _make_fake_model(d_vocab=50, d_model=16, n_layers=2):
    """Fake HookedTransformer-shaped object for patching tests - CPU, no download."""

    class _FakeCfg:
        pass

    cfg = _FakeCfg()
    cfg.n_layers = n_layers

    class _FakeModel:
        def __init__(self):
            self.cfg = cfg

        def parameters(self):
            yield torch.zeros(1)

        def run_with_cache(self, ids, prepend_bos, names_filter):
            seq = ids.shape[1]
            cache: dict = {}
            for layer in range(self.cfg.n_layers):
                key = f"blocks.{layer}.hook_resid_post"
                cache[key] = torch.zeros(1, seq, d_model)
            return None, cache

        def run_with_hooks(self, ids, prepend_bos, fwd_hooks):
            seq = ids.shape[1]
            return torch.randn(1, seq, d_vocab)

        def __call__(self, ids, prepend_bos):
            seq = ids.shape[1]
            return torch.randn(1, seq, d_vocab)

    return _FakeModel()


# ---------------------------------------------------------------------------
# patch_run (fake model, CPU)
# ---------------------------------------------------------------------------

def test_patch_run_returns_float():
    model = _make_fake_model()
    src = _stim("conventional", list(range(6)))
    dst = _stim("swapped", list(range(6)))
    src_resids = _cache_source_resids(model, src)
    result = patch_run(model, dst, src_resids, layer=0, group="import_span", lex=_lex())
    assert isinstance(result, float)
    assert math.isfinite(result)


def test_patch_run_empty_filler_returns_unpatched():
    # filler_span empty -> patch_run returns unpatched pull without error
    model = _make_fake_model()
    src = _stim("conventional", list(range(6)), filler_span=(4, 4))
    dst = _stim("swapped", list(range(6)), filler_span=(4, 4))
    src_resids = _cache_source_resids(model, src)
    result = patch_run(model, dst, src_resids, layer=0, group="filler_span", lex=_lex())
    assert math.isfinite(result)


def test_patch_run_all_groups():
    model = _make_fake_model()
    src = _stim("conventional", list(range(12)), import_span=(0, 4), filler_span=(4, 9), use_alias_pos=9, final_pos=10)
    dst = _stim("swapped", list(range(12)), import_span=(0, 4), filler_span=(4, 9), use_alias_pos=9, final_pos=10)
    src_resids = _cache_source_resids(model, src)
    for group in GROUPS:
        result = patch_run(model, dst, src_resids, layer=0, group=group, lex=_lex())
        assert math.isfinite(result), f"group={group} returned non-finite"


# ---------------------------------------------------------------------------
# run_m2 output contract (fake model)
# ---------------------------------------------------------------------------

_EXPECTED_M2_KEYS = {
    "base_id",
    "direction",
    "layer",
    "group",
    "pair_name",
    "depth",
    "template_id",
    "rep",
    "src_stimulus_id",
    "dst_stimulus_id",
    "pull_src",
    "pull_dst",
    "pull_patched",
    "fraction_restored",
}


def test_run_m2_empty_when_no_pairs():
    model = _make_fake_model()
    stimuli = [_stim("swapped", list(range(6)))]
    records = run_m2(model, stimuli, _lex())
    assert records == []


def test_run_m2_produces_records_for_complete_pair():
    n_layers = 2
    model = _make_fake_model(n_layers=n_layers)
    swap = _stim("swapped", list(range(6)), base_id="b1")
    nopr = _stim("no_prior", list(range(6)), base_id="b1")
    records = run_m2(model, [swap, nopr], _lex())
    # n_layers * n_groups * n_directions = 2 * 4 * 2 = 16
    assert len(records) == n_layers * len(GROUPS) * 2


def test_run_m2_record_keys():
    model = _make_fake_model(n_layers=2)
    swap = _stim("swapped", list(range(6)), base_id="b2")
    nopr = _stim("no_prior", list(range(6)), base_id="b2")
    records = run_m2(model, [swap, nopr], _lex())
    for rec in records:
        assert set(rec.keys()) >= _EXPECTED_M2_KEYS


def test_run_m2_directions_both_present():
    model = _make_fake_model(n_layers=2)
    swap = _stim("swapped", list(range(6)), base_id="b3")
    nopr = _stim("no_prior", list(range(6)), base_id="b3")
    records = run_m2(model, [swap, nopr], _lex())
    directions = {r["direction"] for r in records}
    assert directions == {"noprior_to_swapped", "swapped_to_noprior"}


def test_run_m2_all_groups_covered():
    model = _make_fake_model(n_layers=2)
    swap = _stim("swapped", list(range(6)), base_id="b4")
    nopr = _stim("no_prior", list(range(6)), base_id="b4")
    records = run_m2(model, [swap, nopr], _lex())
    observed_groups = {r["group"] for r in records}
    assert observed_groups == set(GROUPS)


def test_run_m2_pull_patched_is_finite():
    model = _make_fake_model(n_layers=2)
    swap = _stim("swapped", list(range(6)), base_id="b5")
    nopr = _stim("no_prior", list(range(6)), base_id="b5")
    records = run_m2(model, [swap, nopr], _lex())
    for rec in records:
        assert math.isfinite(rec["pull_patched"])


def test_run_m2_single_direction():
    model = _make_fake_model(n_layers=2)
    swap = _stim("swapped", list(range(6)), base_id="b6")
    nopr = _stim("no_prior", list(range(6)), base_id="b6")
    records = run_m2(model, [swap, nopr], _lex(), directions=("noprior_to_swapped",))
    assert all(r["direction"] == "noprior_to_swapped" for r in records)
    assert len(records) == _make_fake_model(n_layers=2).cfg.n_layers * len(GROUPS)


def test_run_m2_conventional_ignored():
    # conventional stimuli not in (swapped, no_prior) -> not paired -> no records
    model = _make_fake_model(n_layers=2)
    conv = _stim("conventional", list(range(6)), base_id="b7")
    swap = _stim("swapped", list(range(6)), base_id="b7")
    records = run_m2(model, [conv, swap], _lex())
    assert records == []
