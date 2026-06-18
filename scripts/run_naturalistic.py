#!/usr/bin/env python
"""Naturalistic arm: does a pandas-demanding task context rescue the binding?

The controlled arm uses neutral filler, so the only cue to the library is the import line. Here we
ask the harder, more practical question: when the surrounding code UNAMBIGUOUSLY calls for a pandas
operation, does the model track the binding, or does the np-alias prior override even the task
semantics? The realistic task contexts are model-generated (scripts/workflow), but the experimental
manipulation stays fully controlled here: we attach the three binding conditions, score the same
prior-pull metric as the main arm (L(numpy) - L(pandas) over the fixed discriminative lexicons), and
hash every prompt. Setups are rejected if they leak np/pd/numpy/pandas, and a scenario is dropped
unless its target pandas op actually resolves on pandas at the pinned version.

  python scripts/run_naturalistic.py --models qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia import deepseek_probe as dp  # parse_attribute  # noqa: E402
from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, sha256_text, stable_hash, utc_now_iso  # noqa: E402
from alias_inertia.lexicons import LEXICONS  # noqa: E402
from alias_inertia.metrics import compute_metric_row  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEX = {"numpy": LEXICONS["numpy"], "pandas": LEXICONS["pandas"]}
FORBIDDEN = ("np", "pd", "numpy", "pandas")  # a setup line must not leak the library or alias
# prior_lib=numpy (the alias np's conventional library), bound_lib=pandas (what np is rebound to)
CONDITIONS = [
    ("swapped", "import pandas as np", "np"),
    ("no_prior", "import pandas as zz", "zz"),
    ("no_prior", "import pandas as qx", "qx"),
    ("no_prior", "import pandas as vv", "vv"),
    ("correct", "import pandas as pd", "pd"),
]


# A clean setup defines only literal data/paths/vars: no alias leak, and no call to a library
# function (a bare ``DataFrame(...)`` etc. would prime pandas and break the neutrality of the setup).
_SETUP_CALL = re.compile(
    r"\b(DataFrame|Series|Index|read_csv|read_excel|read_parquet|read_json|concat|merge|join|"
    r"pivot_table|pivot|melt|crosstab|get_dummies|to_datetime|date_range|factorize|cut|qcut|"
    r"array|arange|zeros|linspace)\s*\(")


def setup_is_clean(setup: str) -> bool:
    low = setup.lower()
    if any(tok in low for tok in FORBIDDEN):
        return False
    return _SETUP_CALL.search(setup) is None


def valid_scenario(s: dict) -> bool:
    op = s.get("pandas_op", "")
    return (
        bool(s.get("setup")) and bool(s.get("use_var")) and bool(op)
        and setup_is_clean(s["setup"])
        and resolves_on(op, "pandas").get("exists") is True  # the target op is a real pandas attr
    )


def build_prompt(import_line: str, setup: str, use_var: str, alias: str) -> str:
    return f"{import_line}\n{setup.rstrip()}\n{use_var} = {alias}."


def boot_ci(vals, rng, n=5000):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return None, None, None
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def boot_did(swap, npr, rng, n=5000):
    a, b = np.asarray(swap, float), np.asarray(npr, float)
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    boots = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
             for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios.json"))
    ap.add_argument("--models", default="qwen2.5-coder:7b", help="comma-separated Ollama GGUF model names")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "naturalistic"))
    args = ap.parse_args()

    with open(args.scenarios, encoding="utf-8") as f:
        raw = json.load(f)
    scenarios = raw["scenarios"] if isinstance(raw, dict) else raw
    kept, dropped = [], []
    for s in scenarios:
        (kept if valid_scenario(s) else dropped).append(s)
    for s in kept:
        s["scenario_id"] = stable_hash([s["intent"], s["setup"], s["use_var"], s["pandas_op"]], length=12)
    print(f"scenarios: {len(scenarios)} -> {len(kept)} valid ({len(dropped)} dropped: leak/op-missing)")

    models = [m for m in args.models.split(",") if m]
    records = []
    for model in models:
        print(f"\n=== model: {model} ===")
        be = build_backend("llamacpp", {"ollama_model": model, "n_ctx": args.n_ctx,
                                        "n_threads": 8, "n_batch": 512, "seed": 12345})
        for i, s in enumerate(kept):
            for cond, import_line, alias in CONDITIONS:
                prompt = build_prompt(import_line, s["setup"], s["use_var"], alias)
                m = compute_metric_row(be, prompt, prior_lib="numpy", bound_lib="pandas", lexicons=LEX)
                try:
                    gen = be.generate(prompt, max_new_tokens=args.max_new_tokens)
                except Exception:
                    gen = ""
                attr = dp.parse_attribute(gen, alias=alias)
                rb = resolves_on(attr, "pandas") if attr else {"exists": None}
                records.append({
                    "model": model, "scenario_id": s["scenario_id"], "domain": s.get("domain", ""),
                    "pandas_op": s["pandas_op"], "condition": cond, "alias": alias,
                    "prior_pull": m["prior_pull"], "L_numpy": m["logsumexp_by_lib"]["numpy"],
                    "L_pandas": m["logsumexp_by_lib"]["pandas"],
                    "gen_attr": attr, "gen_text": gen.strip()[:60],
                    "broken_on_pandas": (rb.get("exists") is False) if attr else None,
                    "wrote_target_op": (attr == s["pandas_op"]) if attr else False,
                    "prompt_sha256": sha256_text(prompt),
                })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(kept)} scenarios scored")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.scenarios, "w", encoding="utf-8") as f:  # persist validated ids
        json.dump({"scenarios": kept, "dropped": dropped}, f, ensure_ascii=False, indent=2)

    # Analysis per model: prior_pull by condition, DiD (swapped - no_prior), broken/target rates.
    rng = np.random.default_rng(20260618)
    summary = {"models": models, "n_scenarios": len(kept), "n_dropped": len(dropped), "per_model": {}}
    for model in models:
        mr = [r for r in records if r["model"] == model]
        by = {c: [r for r in mr if r["condition"] == c] for c in ("swapped", "no_prior", "correct")}
        block = {}
        for c, rs in by.items():
            mean, lo, hi = boot_ci([r["prior_pull"] for r in rs], rng)
            broken = np.mean([r["broken_on_pandas"] for r in rs if r["broken_on_pandas"] is not None]) if rs else None
            target = np.mean([r["wrote_target_op"] for r in rs]) if rs else None
            block[c] = {"prior_pull": mean, "ci": [lo, hi], "n": len(rs),
                        "broken_on_pandas_rate": (float(broken) if broken is not None else None),
                        "wrote_target_op_rate": (float(target) if target is not None else None)}
        did, dlo, dhi = boot_did([r["prior_pull"] for r in by["swapped"]],
                                 [r["prior_pull"] for r in by["no_prior"]], rng)
        block["did_swapped_minus_no_prior"] = {"value": did, "ci": [dlo, dhi]}
        summary["per_model"][model] = block
        print(f"\n[{model}] prior-pull (L_numpy - L_pandas), + = leans numpy prior:")
        for c in ("correct", "no_prior", "swapped"):
            b = block[c]
            print(f"  {c:9} pp={b['prior_pull']:+.2f} [{b['ci'][0]:+.2f},{b['ci'][1]:+.2f}] "
                  f"broken={b['broken_on_pandas_rate']} wrote_target={b['wrote_target_op_rate']} n={b['n']}")
        print(f"  DiD swapped-noprior = {did:+.2f} [{dlo:+.2f},{dhi:+.2f}]")

    summary["environment"] = environment_fingerprint()
    summary["timestamp_utc"] = utc_now_iso()
    with open(args.out + "_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.relpath(args.out + '_results.json', REPO_ROOT)} "
          f"(+ _records.jsonl, prompts hashed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
