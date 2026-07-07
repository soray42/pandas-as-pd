#!/usr/bin/env python
"""Naturalistic task-context arm across ALL alias conventions (the 12 EXTENDED_PAIRS).

Generalises scripts/run_naturalistic.py from the numpy/pandas case to every pair: for a pair
(prior_lib, other_lib) the canonical alias of prior_lib is rebound to other_lib in a snippet whose
surrounding task plainly needs other_lib. We measure the same prior-pull metric, L(prior_lib) minus
L(other_lib) over the fixed discriminative lexicons, plus free generation. Conditions per scenario:
swapped (other_lib aliased as prior_lib's alias), no-prior (other_lib aliased to nonces), correct
(other_lib under its own canonical alias). Realistic contexts are model-generated; the binding
manipulation is applied programmatically and every prompt is hashed.

  python scripts/run_naturalistic_all.py --models qwen2.5-coder:7b,qwen2.5:0.5b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia import deepseek_probe as dp  # noqa: E402
from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, sha256_text, stable_hash, utc_now_iso  # noqa: E402
from alias_inertia.lexicons import CANONICAL_ALIASES, EXTENDED_PAIRS, IMPORT_NAMES, LEXICONS  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NONCES = ["zz", "qx", "vv"]
_ALIAS_TOKENS = set(CANONICAL_ALIASES.values()) | set(NONCES)
_LIB_SUBSTR = ("numpy", "pandas", "sklearn", "xgboost", "matplotlib", "pyplot", "seaborn",
               "networkx", "statsmodels", "sqlalchemy", "sympy", "plotly", "scipy")
_SETUP_CALL = re.compile(
    r"\b(DataFrame|Series|Index|read_csv|read_excel|read_parquet|read_json|concat|merge|join|"
    r"pivot_table|pivot|melt|crosstab|get_dummies|to_datetime|date_range|factorize|cut|qcut|"
    r"array|arange|zeros|ones|linspace|dot|ndarray|eye|full|identity|reshape|concatenate|"
    r"XGBClassifier|XGBRegressor|DMatrix|Booster|linear_model|datasets|preprocessing|"
    r"model_selection|cluster|ensemble)\s*[\(.]")


def setup_is_clean(setup: str) -> bool:
    low = setup.lower()
    if any(tok in low for tok in _LIB_SUBSTR):
        return False
    if _SETUP_CALL.search(setup):
        return False
    # no canonical alias / nonce as a standalone token (substring is fine, e.g. 'sa' in 'sales')
    toks = set(re.findall(r"[A-Za-z_]\w*", setup))
    return not (toks & _ALIAS_TOKENS)


def op_ok(op: str, lib: str) -> bool:
    if not op:
        return False
    r = resolves_on(op, lib)
    if r.get("exists") is True:
        return True
    # sklearn discriminative ops are submodules: real, importable, but not top-level attrs
    import importlib.util as ilu
    try:
        return ilu.find_spec(f"{IMPORT_NAMES[lib]}.{op}") is not None
    except Exception:
        return False


def load_scenarios_by_lib(pandas_path, more_path):
    by_lib = {}
    # pandas set (existing): key pandas_op -> target_op
    if os.path.exists(pandas_path):
        d = json.load(open(pandas_path, encoding="utf-8"))
        rows = d["scenarios"] if isinstance(d, dict) else d
        by_lib["pandas"] = [{"intent": s["intent"], "setup": s["setup"], "use_var": s["use_var"],
                             "target_op": s.get("pandas_op") or s.get("target_op")} for s in rows]
    # numpy / sklearn / xgboost set (new): grouped by "lib"
    if os.path.exists(more_path):
        d = json.load(open(more_path, encoding="utf-8"))
        rows = d["scenarios"] if isinstance(d, dict) else d
        for s in rows:
            by_lib.setdefault(s["lib"], []).append(
                {"intent": s["intent"], "setup": s["setup"], "use_var": s["use_var"], "target_op": s["target_op"]})
    # validate + id
    clean = {}
    for lib, scens in by_lib.items():
        keep = []
        for s in scens:
            if s["setup"] and s["use_var"] and setup_is_clean(s["setup"]) and op_ok(s["target_op"], lib):
                s["scenario_id"] = stable_hash([lib, s["setup"], s["use_var"], s["target_op"]], length=12)
                keep.append(s)
        clean[lib] = keep
    return clean


def prompt_for(bound_lib, alias, setup, use_var):
    return f"import {IMPORT_NAMES[bound_lib]} as {alias}\n{setup.rstrip()}\n{use_var} = {alias}."


def boot_did(swap, npr, rng, n=5000):
    a, b = np.asarray(swap, float), np.asarray(npr, float)
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    boots = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandas", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios.json"))
    ap.add_argument("--more", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios_more.json"))
    ap.add_argument("--models", default="qwen2.5-coder:7b,qwen2.5:0.5b")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "naturalistic_all"))
    args = ap.parse_args()

    by_lib = load_scenarios_by_lib(args.pandas, args.more)
    print("scenarios per bound lib:", {k: len(v) for k, v in by_lib.items()})
    pairs = [p for p in EXTENDED_PAIRS if by_lib.get(p.other_lib)]
    print(f"pairs with scenarios: {len(pairs)}/{len(EXTENDED_PAIRS)}")

    models = [m for m in args.models.split(",") if m]
    lex_cache = {}
    records = []
    for model in models:
        print(f"\n=== model: {model} ===")
        be = build_backend("llamacpp", {"ollama_model": model, "n_ctx": args.n_ctx,
                                        "n_threads": 8, "n_batch": 512, "seed": 12345})
        for pi, pair in enumerate(pairs):
            prior, bound = pair.prior_lib, pair.other_lib
            talias, balias = pair.treatment_alias, CANONICAL_ALIASES[bound]
            lex = {prior: LEXICONS[prior], bound: LEXICONS[bound]}
            conds = ([("swapped", talias), ("correct", balias)] + [("no_prior", nc) for nc in NONCES])
            for s in by_lib[bound]:
                for cond, alias in conds:
                    prompt = prompt_for(bound, alias, s["setup"], s["use_var"])
                    from alias_inertia.metrics import compute_metric_row
                    m = compute_metric_row(be, prompt, prior_lib=prior, bound_lib=bound, lexicons=lex)
                    try:
                        gen = be.generate(prompt, max_new_tokens=args.max_new_tokens)
                    except Exception:
                        gen = ""
                    attr = dp.parse_attribute(gen, alias=alias)
                    rb = resolves_on(attr, bound) if attr else {"exists": None}
                    records.append({
                        "model": model, "pair": pair.name, "prior_lib": prior, "bound_lib": bound,
                        "alias_tier": pair.tier, "scenario_id": s["scenario_id"], "target_op": s["target_op"],
                        "condition": cond, "alias": alias, "prior_pull": m["prior_pull"],
                        "gen_attr": attr, "gen_text": gen.strip()[:50],
                        "broken_on_bound": (rb.get("exists") is False) if attr else None,
                        "wrote_target_op": (attr == s["target_op"]) if attr else False,
                        "prompt_sha256": sha256_text(prompt)})
            print(f"  [{pi+1}/{len(pairs)}] {pair.name} ({len(by_lib[bound])} scen) done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rng = np.random.default_rng(20260618)
    summary = {"models": models, "n_pairs": len(pairs), "per_model": {}}
    for model in models:
        mr = [r for r in records if r["model"] == model]
        per_pair = {}
        for pair in pairs:
            pr = [r for r in mr if r["pair"] == pair.name]
            sw = [r["prior_pull"] for r in pr if r["condition"] == "swapped"]
            npr = [r["prior_pull"] for r in pr if r["condition"] == "no_prior"]
            cor = [r["prior_pull"] for r in pr if r["condition"] == "correct"]
            brk = [r["broken_on_bound"] for r in pr if r["condition"] == "swapped" and r["broken_on_bound"] is not None]
            did, lo, hi = boot_did(sw, npr, rng)
            per_pair[pair.name] = {
                "tier": pair.tier, "pp_swapped": float(np.mean(sw)) if sw else None,
                "pp_no_prior": float(np.mean(npr)) if npr else None,
                "pp_correct": float(np.mean(cor)) if cor else None,
                "did": did, "did_ci": [lo, hi],
                "swapped_broken_rate": float(np.mean(brk)) if brk else None, "n_scen": len(sw)}
        # overall DiD across all pairs (pooled rows)
        allsw = [r["prior_pull"] for r in mr if r["condition"] == "swapped"]
        allnp = [r["prior_pull"] for r in mr if r["condition"] == "no_prior"]
        od, olo, ohi = boot_did(allsw, allnp, rng)
        summary["per_model"][model] = {"overall_did": od, "overall_did_ci": [olo, ohi], "per_pair": per_pair}
        print(f"\n[{model}] overall naturalistic DiD (swapped-no_prior, nats) = {od:+.2f} [{olo:+.2f},{ohi:+.2f}]")
        for name, b in sorted(per_pair.items(), key=lambda kv: -(kv[1]["did"] or -99)):
            if b["did"] is not None:
                print(f"   {name:26} ({b['tier']:11}) DiD={b['did']:+6.2f} swap_pp={b['pp_swapped']:+6.2f} "
                      f"noprior_pp={b['pp_no_prior']:+6.2f} broken={b['swapped_broken_rate']}")

    summary["environment"] = environment_fingerprint(); summary["timestamp_utc"] = utc_now_iso()
    with open(args.out + "_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.relpath(args.out + '_results.json', REPO_ROOT)} (+ _records.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
