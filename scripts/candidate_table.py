#!/usr/bin/env python
"""Version-pinned candidate-continuation table (appendix rigor).

For every scored continuation we record: the library, its canonical alias, the bare attribute,
whether it resolves on the library at the pinned package version, and its token length under a
reference tokenizer. This documents that the continuations are real attributes (not invented),
ties them to package versions, and lets the reader check the per-library candidate sets are
comparable in token length (the matched-candidate concern). Output: results/candidate_table.json
plus a compact per-library LaTeX summary on stdout.

  python scripts/candidate_table.py
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alias_inertia.lexicons import CANONICAL_ALIASES, IMPORT_NAMES, LEXICONS, normalize_member  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# top-level distribution name for each importable module, for version pinning
_DIST = {
    "numpy": "numpy", "pandas": "pandas", "torch": "torch", "sklearn": "scikit-learn",
    "xgboost": "xgboost", "matplotlib.pyplot": "matplotlib", "seaborn": "seaborn",
    "networkx": "networkx", "statsmodels.api": "statsmodels", "sqlalchemy": "SQLAlchemy",
    "sympy": "sympy", "plotly.express": "plotly",
}


def _version(lib):
    try:
        return md.version(_DIST.get(lib, lib))
    except Exception:
        return "unknown"


def main() -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    rows = []
    per_lib = {}
    for lib, members in LEXICONS.items():
        alias = CANONICAL_ALIASES.get(lib, "?")
        ver = _version(lib)
        tlens = []
        import importlib.util as _ilu
        for m in members:
            bare = normalize_member(m)
            exists = resolves_on(bare, lib)["exists"]
            # A submodule continuation (e.g. sklearn ``linear_model.``) is a real, importable
            # subpackage even though it is not a top-level attribute before import.
            if not exists and m.endswith("."):
                try:
                    exists = _ilu.find_spec(f"{IMPORT_NAMES[lib]}.{bare}") is not None
                except Exception:
                    exists = False
            tlen = len(enc.encode("." + bare))  # use-site form ".method"
            tlens.append(tlen)
            rows.append({"library": lib, "alias": alias, "module": IMPORT_NAMES[lib],
                         "continuation": m, "attribute": bare, "resolves": exists,
                         "token_len": tlen, "version": ver})
        per_lib[lib] = {
            "alias": alias, "version": ver, "n": len(members),
            "mean_token_len": round(sum(tlens) / len(tlens), 2),
            "all_resolve": all(r["resolves"] for r in rows if r["library"] == lib),
        }

    out = {"n_continuations": len(rows), "all_resolve": all(r["resolves"] for r in rows),
           "tokenizer": "tiktoken/cl100k_base", "per_library": per_lib, "rows": rows}
    with open(os.path.join(REPO_ROOT, "results", "candidate_table.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{len(rows)} continuations across {len(LEXICONS)} libraries; "
          f"all resolve at pinned versions: {out['all_resolve']}")
    print("\n% per-library summary (LaTeX rows: library & alias & n & mean token len & version)")
    for lib, d in per_lib.items():
        print(f"  {lib} & \\texttt{{{d['alias']}}} & {d['n']} & {d['mean_token_len']} & {d['version']} \\\\")
    print("\nsaved -> results/candidate_table.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
