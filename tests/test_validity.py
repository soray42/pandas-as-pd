"""Consequence/validity arm: static attribute resolution against the real bound library."""

from __future__ import annotations

from alias_inertia.validity import library_available, resolves_on


def test_numpy_array_resolves_on_numpy():
    r = resolves_on("array", "numpy")
    assert r["status"] == "resolves" and r["exists"] is True


def test_numpy_only_method_broken_on_pandas():
    # pandas has no `arange` / `ndarray` -> the swapped (np->pandas) call would AttributeError
    assert resolves_on("arange", "pandas")["status"] == "broken"
    assert resolves_on("ndarray", "pandas")["status"] == "broken"


def test_pandas_only_method_broken_on_numpy():
    assert resolves_on("DataFrame", "numpy")["status"] == "broken"
    assert resolves_on("read_csv", "numpy")["status"] == "broken"


def test_unknown_attr_when_none():
    assert resolves_on(None, "numpy")["status"] == "unknown_attr"


def test_libraries_available():
    # numpy/pandas always; xgboost installed for the rare-tier pairs' validity coverage
    assert library_available("numpy") and library_available("pandas")
    assert library_available("xgboost")
