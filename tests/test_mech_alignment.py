"""Tests for MechStimulus triple alignment (alias_inertia.mech.stimuli_mech).

"Alignment" = the three conditions (conventional, swapped, no_prior) sharing
the same token positions for filler and usage site so that activation patching
is position-equivalent across conditions.

CPU-only tests (synthetic MechStimulus objects) run without model download.
Tests that require build_mech_stimuli or a real tokenizer are marked xfail
pending full implementation.
"""

from __future__ import annotations

import pytest

try:
    from alias_inertia.mech.stimuli_mech import AlignmentError, MechStimulus, build_mech_stimuli
    from alias_inertia.mech.patching import positions_for
except ImportError as _err:
    pytest.skip(f"mech.stimuli_mech not importable: {_err}", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stim(
    *,
    condition="swapped",
    import_span=(0, 4),
    filler_span=(4, 8),
    use_alias_pos=8,
    final_pos=9,
    token_ids=None,
    base_id="b0",
    alias="np",
    template_id="t1",
):
    if token_ids is None:
        token_ids = list(range(10))
    return MechStimulus(
        base_id=base_id,
        stimulus_id=f"{base_id}_{condition}",
        pair_name="numpy__pandas",
        condition=condition,
        alias=alias,
        depth=512,
        template_id=template_id,
        rep=0,
        use_var="x",
        prompt=f"import pandas as {alias}\n<filler>\n{alias}.",
        token_ids=token_ids,
        import_span=import_span,
        filler_span=filler_span,
        use_alias_pos=use_alias_pos,
        final_pos=final_pos,
        prompt_sha256="deadbeef",
    )


# ---------------------------------------------------------------------------
# AlignmentError
# ---------------------------------------------------------------------------

def test_alignment_error_is_exception():
    assert issubclass(AlignmentError, Exception)


def test_alignment_error_can_be_raised_and_caught():
    with pytest.raises(AlignmentError):
        raise AlignmentError("token_ids lengths differ at filler positions")


def test_alignment_error_message_preserved():
    msg = "filler_span mismatch: (4, 8) vs (4, 9)"
    try:
        raise AlignmentError(msg)
    except AlignmentError as exc:
        assert msg in str(exc)


# ---------------------------------------------------------------------------
# MechStimulus dataclass structure
# ---------------------------------------------------------------------------

def test_mechstimulus_fields_accessible():
    s = _stim()
    assert s.base_id == "b0"
    assert s.stimulus_id == "b0_swapped"
    assert s.condition == "swapped"
    assert s.alias == "np"
    assert s.pair_name == "numpy__pandas"
    assert isinstance(s.token_ids, list)
    assert isinstance(s.import_span, tuple) and len(s.import_span) == 2
    assert isinstance(s.filler_span, tuple) and len(s.filler_span) == 2
    assert isinstance(s.use_alias_pos, int)
    assert isinstance(s.final_pos, int)
    assert isinstance(s.prompt_sha256, str)


def test_mechstimulus_template_id_is_string():
    # template_id is a str (e.g. "t1"), not an int.
    s = _stim(template_id="t1")
    assert isinstance(s.template_id, str)
    assert s.template_id == "t1"


def test_mechstimulus_final_pos_is_last_token():
    ids = list(range(12))
    s = _stim(token_ids=ids, final_pos=11)
    assert s.final_pos == len(s.token_ids) - 1


def test_mechstimulus_import_span_within_token_ids():
    s = _stim(import_span=(0, 4), token_ids=list(range(10)))
    start, end = s.import_span
    assert 0 <= start < end <= len(s.token_ids)


def test_mechstimulus_filler_span_within_token_ids():
    s = _stim(filler_span=(4, 8), token_ids=list(range(10)))
    start, end = s.filler_span
    assert 0 <= start <= end <= len(s.token_ids)


def test_mechstimulus_depth_zero_empty_filler():
    # At depth 0 there is no filler; filler_span is a zero-length interval.
    s = _stim(filler_span=(4, 4), use_alias_pos=4, final_pos=5, token_ids=list(range(6)))
    start, end = s.filler_span
    assert start == end


# ---------------------------------------------------------------------------
# Triple alignment invariants (checked via positions_for)
# ---------------------------------------------------------------------------

def test_triple_same_filler_span_gives_same_positions():
    conv = _stim(condition="conventional", filler_span=(4, 8))
    swap = _stim(condition="swapped", filler_span=(4, 8))
    nopr = _stim(condition="no_prior", filler_span=(4, 8))
    assert positions_for(conv, "filler_span") == positions_for(swap, "filler_span")
    assert positions_for(swap, "filler_span") == positions_for(nopr, "filler_span")


def test_triple_same_final_pos():
    conv = _stim(condition="conventional", final_pos=9)
    swap = _stim(condition="swapped", final_pos=9)
    nopr = _stim(condition="no_prior", final_pos=9)
    assert positions_for(conv, "final_pos") == positions_for(swap, "final_pos") == positions_for(nopr, "final_pos") == [9]


def test_triple_aligned_import_span_same_length():
    # Import spans may differ in CONTENT (numpy vs pandas) but must share the same length
    # so that patching at relative positions within the import is position-equivalent.
    conv = _stim(condition="conventional", import_span=(0, 4))
    swap = _stim(condition="swapped", import_span=(0, 4))
    nopr = _stim(condition="no_prior", import_span=(0, 4))
    assert len(positions_for(conv, "import_span")) == len(positions_for(swap, "import_span"))
    assert len(positions_for(swap, "import_span")) == len(positions_for(nopr, "import_span"))


def test_depth_zero_filler_span_empty_for_all_conditions():
    conv = _stim(condition="conventional", filler_span=(4, 4), token_ids=list(range(6)))
    swap = _stim(condition="swapped", filler_span=(4, 4), token_ids=list(range(6)))
    assert positions_for(conv, "filler_span") == []
    assert positions_for(swap, "filler_span") == []


# ---------------------------------------------------------------------------
# build_mech_stimuli: requires a real tokenizer with offset_mapping support.
# Tests below are skipped unless a compatible tokenizer can be loaded.
# ---------------------------------------------------------------------------

try:
    from transformers import AutoTokenizer as _AutoTokenizer
    _tok = _AutoTokenizer.from_pretrained("gpt2", add_prefix_space=False)
    _HAS_TOKENIZER = True
except Exception:
    _tok = None
    _HAS_TOKENIZER = False


@pytest.mark.skipif(not _HAS_TOKENIZER, reason="gpt2 tokenizer not available")
def test_build_mech_stimuli_returns_list():
    stimuli = build_mech_stimuli(_tok, depths=(0,), n_per_cell=1)
    assert isinstance(stimuli, list)


@pytest.mark.skipif(not _HAS_TOKENIZER, reason="gpt2 tokenizer not available")
def test_build_mech_stimuli_produces_mechstimulus_objects():
    stimuli = build_mech_stimuli(_tok, depths=(0,), n_per_cell=1)
    for s in stimuli:
        assert isinstance(s, MechStimulus)


@pytest.mark.skipif(not _HAS_TOKENIZER, reason="gpt2 tokenizer not available")
def test_build_mech_stimuli_conditions_are_conventional_swapped_noprior():
    stimuli = build_mech_stimuli(_tok, depths=(0,), n_per_cell=1)
    observed = {s.condition for s in stimuli}
    assert observed <= {"conventional", "swapped", "no_prior"}


# ---------------------------------------------------------------------------
# Optional: assert_triple_aligned (may be added as part of full implementation)
# ---------------------------------------------------------------------------

try:
    from alias_inertia.mech.stimuli_mech import assert_triple_aligned as _assert_triple_aligned
    _HAS_ASSERT_ALIGNED = True
except ImportError:
    _HAS_ASSERT_ALIGNED = False


@pytest.mark.skipif(not _HAS_ASSERT_ALIGNED, reason="assert_triple_aligned not yet implemented")
def test_assert_triple_aligned_passes_for_valid_triple():
    conv = _stim(condition="conventional", filler_span=(4, 8), final_pos=9, token_ids=list(range(10)))
    swap = _stim(condition="swapped", filler_span=(4, 8), final_pos=9, token_ids=list(range(10)))
    nopr = _stim(condition="no_prior", filler_span=(4, 8), final_pos=9, token_ids=list(range(10)))
    _assert_triple_aligned(conv, swap, nopr)  # should not raise


@pytest.mark.skipif(not _HAS_ASSERT_ALIGNED, reason="assert_triple_aligned not yet implemented")
def test_assert_triple_aligned_raises_on_filler_mismatch():
    conv = _stim(condition="conventional", filler_span=(4, 8))
    swap = _stim(condition="swapped", filler_span=(4, 9))  # mismatched
    nopr = _stim(condition="no_prior", filler_span=(4, 8))
    with pytest.raises(AlignmentError):
        _assert_triple_aligned(conv, swap, nopr)


@pytest.mark.skipif(not _HAS_ASSERT_ALIGNED, reason="assert_triple_aligned not yet implemented")
def test_assert_triple_aligned_raises_on_length_mismatch():
    conv = _stim(condition="conventional", token_ids=list(range(10)))
    swap = _stim(condition="swapped", token_ids=list(range(11)))  # extra token
    nopr = _stim(condition="no_prior", token_ids=list(range(10)))
    with pytest.raises(AlignmentError):
        _assert_triple_aligned(conv, swap, nopr)
