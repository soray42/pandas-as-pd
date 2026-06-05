"""Stimulus generator correctness: filler never mentions the alias, depth lands on target,
each condition builds the right import line, prompt ends exactly at the usage site."""

from __future__ import annotations

import pytest

from alias_inertia.lexicons import SWAP_PAIRS
from alias_inertia.stimuli import CONDITIONS, TEMPLATES, build_stimulus, generate_grid, reps_for_depth

PAIR = SWAP_PAIRS[0]  # numpy <-> pandas
SEED = 12345


def _build(condition, depth, template="t1", rep=0, count_tokens=None):
    return build_stimulus(
        pair=PAIR,
        condition=condition,
        depth_tokens=depth,
        template_id=template,
        rep=rep,
        count_tokens=count_tokens,
        seed=SEED,
        non_canonical_alias="zz",
        depth_tolerance_tokens=24,
    )


def test_import_lines_per_condition(fake_tokenizer):
    conv = _build("conventional", 0, count_tokens=fake_tokenizer)
    swp = _build("swapped", 0, count_tokens=fake_tokenizer)
    npr = _build("no_prior", 0, count_tokens=fake_tokenizer)
    assert conv.meta["import_line"] == "import numpy as np"
    assert swp.meta["import_line"] == "import pandas as np"
    assert npr.meta["import_line"] == "import pandas as zz"
    assert conv.prompt.endswith("np.")
    assert swp.prompt.endswith("np.")
    assert npr.prompt.endswith("zz.")


def test_prior_and_bound_targets(fake_tokenizer):
    conv = _build("conventional", 0, count_tokens=fake_tokenizer).meta
    swp = _build("swapped", 0, count_tokens=fake_tokenizer).meta
    npr = _build("no_prior", 0, count_tokens=fake_tokenizer).meta
    assert (conv["prior_target"], conv["bound_target"]) == ("numpy", "numpy")
    assert (swp["prior_target"], swp["bound_target"]) == ("numpy", "pandas")
    assert (npr["prior_target"], npr["bound_target"]) == ("numpy", "pandas")
    assert conv["alias_has_canonical_prior"] is True
    assert swp["alias_has_canonical_prior"] is True
    assert npr["alias_has_canonical_prior"] is False


def test_prompt_ends_exactly_at_usage(fake_tokenizer):
    for cond in CONDITIONS:
        for tmpl in TEMPLATES:
            for depth in (0, 64, 256):
                s = _build(cond, depth, template=tmpl, count_tokens=fake_tokenizer)
                alias = s.meta["alias"]
                assert s.prompt.endswith(f"{alias}.")
                # nothing after the dot (no trailing whitespace/newline)
                assert s.prompt == s.prompt.rstrip()


def test_filler_never_mentions_alias(fake_tokenizer):
    for cond in CONDITIONS:
        for tmpl in TEMPLATES:
            for depth in (128, 512):
                for rep in range(3):
                    s = _build(cond, depth, template=tmpl, rep=rep, count_tokens=fake_tokenizer)
                    alias = s.meta["alias"]
                    # reconstruct the filler region (between import line and usage)
                    body = s.prompt
                    # the alias appears only in the import line and the final usage token
                    occurrences = body.count(f"{alias}")
                    # import line has the alias once; usage has it once -> exactly 2 in t1,
                    # (header has none). So filler contributed zero alias mentions.
                    assert occurrences == 2, f"{cond}/{tmpl}/d{depth}: alias {alias!r} leaked ({occurrences})"


def test_depth_zero_has_no_filler(fake_tokenizer):
    for cond in CONDITIONS:
        s = _build(cond, 0, count_tokens=fake_tokenizer)
        assert s.meta["depth_tokens_actual"] == 0
        # exactly import line + usage line (plus optional header)
        lines = [ln for ln in s.prompt.splitlines() if ln and not ln.startswith("#")]
        assert len(lines) == 2


def test_depth_lands_within_tolerance(fake_tokenizer):
    tol = 24
    for depth in (256, 512, 1024):
        for cond in CONDITIONS:
            s = _build(cond, depth, count_tokens=fake_tokenizer)
            actual = s.meta["depth_tokens_actual"]
            assert depth - tol <= actual <= depth + tol, f"{cond} d={depth}: actual={actual}"


def test_generation_is_deterministic(fake_tokenizer):
    a = _build("swapped", 512, rep=1, count_tokens=fake_tokenizer)
    b = _build("swapped", 512, rep=1, count_tokens=fake_tokenizer)
    assert a.prompt == b.prompt
    assert a.meta["prompt_sha256"] == b.meta["prompt_sha256"]


def test_reps_differ(fake_tokenizer):
    a = _build("swapped", 512, rep=0, count_tokens=fake_tokenizer)
    b = _build("swapped", 512, rep=1, count_tokens=fake_tokenizer)
    assert a.prompt != b.prompt  # different filler per rep -> within-cell variance


def test_reps_for_depth_collapses_depth0():
    # depth-0 degeneracy fix: no filler to vary -> a single rep
    assert reps_for_depth(0, 3) == 1
    assert reps_for_depth(512, 3) == 3


def test_deep_bins_policy():
    from alias_inertia.stimuli import grid_plan_for_depth
    # depth 0 -> all templates, 1 rep
    assert grid_plan_for_depth(0, ["t1", "t2"], 3, 8192, ["t1"], 1) == (["t1", "t2"], 1)
    # shallow -> all templates, full reps
    assert grid_plan_for_depth(2048, ["t1", "t2"], 3, 8192, ["t1"], 1) == (["t1", "t2"], 3)
    # deep -> deep templates, deep reps
    assert grid_plan_for_depth(32768, ["t1", "t2"], 3, 8192, ["t1"], 1) == (["t1"], 1)
    # no threshold -> never deep
    assert grid_plan_for_depth(32768, ["t1", "t2"], 3, None, ["t1"], 1) == (["t1", "t2"], 3)


def test_grid_cardinality(fake_tokenizer):
    stimuli = list(
        generate_grid(
            pair=PAIR,
            conditions=["conventional", "swapped", "no_prior"],
            depths_tokens=[0, 128, 256],
            templates=["t1", "t2"],
            repetitions=3,
            count_tokens=fake_tokenizer,
            seed=SEED,
        )
    )
    # depth 0 -> 1 rep; depths 128/256 -> 3 reps each. 3 conds x 2 templates x (1 + 3 + 3).
    assert len(stimuli) == 3 * 2 * (1 + 3 + 3)
    ids = {s.meta["stimulus_id"] for s in stimuli}
    assert len(ids) == len(stimuli)  # all unique


def test_long_filler_efficient_and_on_target(fake_tokenizer):
    # large depth must still land within tolerance (exercises the chunked filler builder)
    s = build_stimulus(
        pair=PAIR, condition="swapped", depth_tokens=2000, template_id="t1", rep=0,
        count_tokens=fake_tokenizer, seed=SEED, depth_tolerance_tokens=48,
    )
    assert abs(s.meta["depth_tokens_actual"] - 2000) <= 48
    assert "np" not in "\n".join(  # alias still never leaks into filler
        ln for ln in s.prompt.splitlines() if ln not in (s.meta["import_line"],) and not ln.endswith("np.")
    )


def test_rejects_canonical_non_canonical_alias(fake_tokenizer):
    with pytest.raises(ValueError):
        build_stimulus(
            pair=PAIR, condition="no_prior", depth_tokens=0, template_id="t1", rep=0,
            count_tokens=fake_tokenizer, seed=SEED, non_canonical_alias="pd",  # pd IS canonical
        )
