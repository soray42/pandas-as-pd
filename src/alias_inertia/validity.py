"""Consequence / validity arm: would the generated call actually resolve under the BOUND library?

For each generation we extract the attribute accessed on the alias and check whether it exists
on the REAL bound library (e.g. swapped ``np`` is bound to pandas -> does ``pandas.array`` exist?).
This is done by **static attribute lookup** (``hasattr``) against the actually-installed library -
NOT by executing any model-generated code. A heavily-guarded subprocess is provided as an opt-in
fallback but is unnecessary for attribute existence and is OFF by default.

``broken`` = the attribute does not exist on the bound library (would raise AttributeError).
Note: existence does not guarantee correct *behaviour* (e.g. ``pandas.array`` exists but differs
from ``numpy.array``); this arm reports the conservative, unambiguous attribute-resolution rate.
"""

from __future__ import annotations

import importlib
import os

VALIDITY_VERSION = "1.0"

# Library key -> importable module path.
_IMPORT_PATH = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "matplotlib.pyplot": "matplotlib.pyplot",
}

_MODULE_CACHE: dict[str, object] = {}
_IMPORT_FAILED: set[str] = set()


def _get_module(lib: str):
    if lib in _MODULE_CACHE:
        return _MODULE_CACHE[lib]
    if lib in _IMPORT_FAILED:
        return None
    path = _IMPORT_PATH.get(lib)
    if path is None:
        _IMPORT_FAILED.add(lib)
        return None
    try:
        if path == "matplotlib.pyplot":
            os.environ.setdefault("MPLBACKEND", "Agg")  # headless; no display needed
        mod = importlib.import_module(path)
        _MODULE_CACHE[lib] = mod
        return mod
    except Exception:
        _IMPORT_FAILED.add(lib)
        return None


def library_available(lib: str) -> bool:
    return _get_module(lib) is not None


def resolves_on(attribute: str | None, bound_lib: str) -> dict:
    """Static attribute resolution check.

    Returns {status, exists, library_available} with status in
    {resolves, broken, unknown_attr, unknown_lib}.
    """
    mod = _get_module(bound_lib)
    if mod is None:
        return {"status": "unknown_lib", "exists": None, "library_available": False}
    if not attribute:
        return {"status": "unknown_attr", "exists": None, "library_available": True}
    exists = hasattr(mod, attribute)
    return {
        "status": "resolves" if exists else "broken",
        "exists": bool(exists),
        "library_available": True,
    }
