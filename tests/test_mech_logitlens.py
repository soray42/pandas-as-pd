"""Tests for alias_inertia.mech.logitlens: trajectory and run_m1.

CPU tests (names_filter, record schema via fake model) run without model download.
Integration tests with a real HookedTransformer are skipped unless the model
can be loaded (they require GPU and model weights).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

try:
    from alias_inertia.mech.logitlens import _NAMES_FILTER, run_m1, trajectory
    from alias_inertia.mech.proxy import ProxyLexicon
    from alias_inertia.mech.stimuli_mech import MechStimulus
except ImportError as _err:
    pytest.skip(f"alias_inertia.mech.logitlens not importable: {_err}", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lex(prior_ids=(10, 11), bound_ids=(20, 21)):
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


def _stim(token_ids, *, final_pos=None, depth=0):
    n = len(token_ids)
    fp = final_pos if final_pos is not None else n - 1
    return MechStimulus(
        base_id="b0",
        stimulus_id="b0_swapped",
        pair_name="numpy__pandas",
        condition="swapped",
        alias="np",
        depth=depth,
        template_id="t1",
        rep=0,
        use_var="x",
        prompt="import pandas as np\nnp.",
        token_ids=token_ids,
        import_span=(0, 4),
        filler_span=(4, 4),
        use_alias_pos=fp - 1 if fp > 0 else 0,
        final_pos=fp,
        prompt_sha256="",
    )


# ---------------------------------------------------------------------------
# _NAMES_FILTER (pure Python, no model)
# ---------------------------------------------------------------------------

def test_names_filter_accepts_resid_pre():
    assert _NAMES_FILTER("blocks.0.hook_resid_pre")
    assert _NAMES_FILTER("blocks.11.hook_resid_pre")


def test_names_filter_accepts_resid_post():
    assert _NAMES_FILTER("blocks.0.hook_resid_post")
    assert _NAMES_FILTER("blocks.5.hook_resid_post")


def test_names_filter_accepts_ln_final_scale():
    assert _NAMES_FILTER("ln_final.hook_scale")


def test_names_filter_rejects_hook_z():
    assert not _NAMES_FILTER("blocks.0.attn.hook_z")


def test_names_filter_rejects_hook_pattern():
    assert not _NAMES_FILTER("blocks.0.attn.hook_pattern")


def test_names_filter_rejects_mlp_out():
    assert not _NAMES_FILTER("blocks.0.hook_mlp_out")
    assert not _NAMES_FILTER("blocks.0.mlp.hook_post")


# ---------------------------------------------------------------------------
# run_m1 record schema (fake model, no download)
# ---------------------------------------------------------------------------

_EXPECTED_M1_KEYS = {
    "stimulus_id",
    "base_id",
    "pair_name",
    "condition",
    "alias",
    "depth",
    "template_id",
    "rep",
    "prompt_sha256",
    "trajectory",
    "final_proxy_pull",
    "n_checkpoints",
}


def _make_fake_model(n_layers=4, d_model=32, d_vocab=100):
    """Fake HookedTransformer-shaped object for CPU tests - no download required."""

    class _FakeCache:
        def __init__(self, nl, dm):
            self._data = torch.zeros(nl + 1, 1, dm)

        def accumulated_resid(self, layer, apply_ln, pos_slice):
            return self._data

    class _FakeModel:
        class _unembed:
            def __init__(self, dm, dv):
                self.W_U = torch.randn(dm, dv)
                self.b_U = torch.zeros(dv)

        def __init__(self, nl, dm, dv):
            self._nl = nl
            self._dm = dm
            self._dv = dv
            self.unembed = self._unembed(dm, dv)

        def parameters(self):
            yield self.unembed.W_U

        def run_with_cache(self, ids, prepend_bos, names_filter):
            return None, _FakeCache(self._nl, self._dm)

    return _FakeModel(n_layers, d_model, d_vocab)


def test_run_m1_record_keys():
    model = _make_fake_model(n_layers=2)
    stim = _stim(list(range(6)))
    records = run_m1(model, [stim], _lex(prior_ids=[0], bound_ids=[1]))
    assert len(records) == 1
    rec = records[0]
    assert set(rec.keys()) >= _EXPECTED_M1_KEYS


def test_run_m1_trajectory_is_list():
    model = _make_fake_model(n_layers=2)
    stim = _stim(list(range(6)))
    records = run_m1(model, [stim], _lex(prior_ids=[0], bound_ids=[1]))
    assert isinstance(records[0]["trajectory"], list)


def test_run_m1_n_checkpoints_equals_n_layers_plus_one():
    n_layers = 3
    model = _make_fake_model(n_layers=n_layers)
    stim = _stim(list(range(6)))
    records = run_m1(model, [stim], _lex(prior_ids=[0], bound_ids=[1]))
    assert records[0]["n_checkpoints"] == n_layers + 1


def test_run_m1_final_proxy_pull_is_finite():
    model = _make_fake_model(n_layers=2)
    stim = _stim(list(range(6)))
    records = run_m1(model, [stim], _lex(prior_ids=[0, 2], bound_ids=[1, 3]))
    assert math.isfinite(records[0]["final_proxy_pull"])


def test_run_m1_empty_stimuli():
    model = _make_fake_model(n_layers=2)
    records = run_m1(model, [], _lex(prior_ids=[0], bound_ids=[1]))
    assert records == []


def test_run_m1_metadata_forwarded():
    model = _make_fake_model(n_layers=2)
    stim = MechStimulus(
        base_id="bx", stimulus_id="bx_conventional", pair_name="torch__sklearn",
        condition="conventional", alias="torch", depth=512, template_id="t1", rep=1,
        use_var="x", prompt="import torch as torch\ntorch.", token_ids=list(range(6)),
        import_span=(0, 4), filler_span=(4, 4), use_alias_pos=4, final_pos=5,
        prompt_sha256="abc",
    )
    records = run_m1(model, [stim], _lex(prior_ids=[0], bound_ids=[1]))
    rec = records[0]
    assert rec["condition"] == "conventional"
    assert rec["pair_name"] == "torch__sklearn"
    assert rec["depth"] == 512
    assert rec["rep"] == 1


def test_run_m1_multiple_stimuli():
    model = _make_fake_model(n_layers=2)
    stimuli = [_stim(list(range(6 + i))) for i in range(4)]
    records = run_m1(model, stimuli, _lex(prior_ids=[0], bound_ids=[1]))
    assert len(records) == 4


# ---------------------------------------------------------------------------
# trajectory: shape and type (fake model, no download)
# ---------------------------------------------------------------------------

def test_trajectory_returns_numpy_array():
    model = _make_fake_model(n_layers=3)
    stim = _stim(list(range(6)))
    result = trajectory(model, stim, _lex(prior_ids=[0], bound_ids=[1]))
    assert isinstance(result, np.ndarray)


def test_trajectory_length_is_n_layers_plus_one():
    n_layers = 6
    model = _make_fake_model(n_layers=n_layers)
    stim = _stim(list(range(8)))
    result = trajectory(model, stim, _lex(prior_ids=[0], bound_ids=[1]))
    assert len(result) == n_layers + 1


def test_trajectory_values_are_finite():
    model = _make_fake_model(n_layers=4)
    stim = _stim(list(range(6)))
    result = trajectory(model, stim, _lex(prior_ids=[0], bound_ids=[1]))
    assert all(math.isfinite(float(v)) for v in result)


def test_trajectory_final_value_matches_run_m1():
    # trajectory[-1] should equal run_m1's final_proxy_pull for the same input.
    model = _make_fake_model(n_layers=2)
    stim = _stim(list(range(6)))
    lex = _lex(prior_ids=[0], bound_ids=[1])
    traj = trajectory(model, stim, lex)
    records = run_m1(model, [stim], lex)
    assert abs(float(traj[-1]) - records[0]["final_proxy_pull"]) < 1e-5
