"""Metric correctness: logsumexp aggregation and prior_pull sign logic (model-free stub)."""

from __future__ import annotations

import math

from alias_inertia.metrics import compute_metric_row, logsumexp, score_lexicon


def test_logsumexp_known_values():
    assert math.isclose(logsumexp([0.0, 0.0]), math.log(2.0), rel_tol=1e-12)
    assert math.isclose(logsumexp([math.log(2), math.log(3)]), math.log(5.0), rel_tol=1e-12)
    assert logsumexp([]) == float("-inf")
    # numerically stable for large negatives
    assert math.isclose(logsumexp([-1000.0, -1000.0]), -1000.0 + math.log(2), rel_tol=1e-12)


def test_score_lexicon_logsumexp(stub_backend_factory):
    be = stub_backend_factory({"a(": math.log(0.1), "b(": math.log(0.2)})
    out = score_lexicon(be, "prompt", ["a(", "b("])
    assert math.isclose(out["logsumexp"], math.log(0.3), rel_tol=1e-9)
    assert set(out["per_continuation"]) == {"a(", "b("}


def test_prior_pull_positive_when_prior_higher(stub_backend_factory):
    # prior (numpy) continuations score higher than bound (pandas) -> prior_pull > 0
    table = {"array(": math.log(0.4), "arange(": math.log(0.4), "DataFrame(": math.log(0.01), "read_csv(": math.log(0.01)}
    be = stub_backend_factory(table)
    row = compute_metric_row(
        be, "import pandas as np\nnp.",
        prior_lib="numpy", bound_lib="pandas",
        lexicons={"numpy": ["array(", "arange("], "pandas": ["DataFrame(", "read_csv("]},
    )
    assert row["prior_pull"] > 0
    assert math.isclose(row["bound_mass"], row["logsumexp_by_lib"]["pandas"], rel_tol=1e-12)


def test_prior_pull_negative_when_bound_higher(stub_backend_factory):
    table = {"array(": math.log(0.01), "arange(": math.log(0.01), "DataFrame(": math.log(0.4), "read_csv(": math.log(0.4)}
    be = stub_backend_factory(table)
    row = compute_metric_row(
        be, "import pandas as zz\nzz.",
        prior_lib="numpy", bound_lib="pandas",
        lexicons={"numpy": ["array(", "arange("], "pandas": ["DataFrame(", "read_csv("]},
    )
    assert row["prior_pull"] < 0


def test_conventional_bound_mass_is_prior_lexicon(stub_backend_factory):
    # conventional: prior_lib == bound_lib -> prior_pull degenerate (0), bound_mass = L(numpy)
    table = {"array(": math.log(0.4), "arange(": math.log(0.4)}
    be = stub_backend_factory(table)
    row = compute_metric_row(
        be, "import numpy as np\nnp.",
        prior_lib="numpy", bound_lib="numpy",
        lexicons={"numpy": ["array(", "arange("]},
    )
    assert math.isclose(row["prior_pull"], 0.0, abs_tol=1e-12)
    assert math.isclose(row["bound_mass"], logsumexp([math.log(0.4), math.log(0.4)]), rel_tol=1e-9)
