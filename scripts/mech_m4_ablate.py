#!/usr/bin/env python
"""M4 (stretch): top-k head ablation effect and collateral check.

Reads the m3 DLA ranking from mech/results/m3_summary.json (or --m3-json).
Ablates the top-k prior-promoting heads (by DLA on swapped items) with scale
factors {0, 0.25, 0.5}. Measures proxy_pull on:
  - swapped items: the effect (does prior-pull drop?)
  - conventional items: the collateral check (do we also break correct behaviour?)

Summary table: mean proxy_pull per (k, factor, condition) with bootstrap 95% CI
vs unablated baseline.

  python scripts/mech_m4_ablate.py [--top-k 3] [--factors 0,0.25,0.5]
      [--m3-json PATH] [--models M] [--depths D] [--items N] [--seed S] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=3,
                    help="ablate top-k heads by |DLA| on swapped (default: 3, max: 5)")
    ap.add_argument("--factors", default="0,0.25,0.5",
                    help="comma-separated ablation scale factors (default: 0,0.25,0.5)")
    ap.add_argument("--m3-json", default=None,
                    help="path to m3_summary.json (default: mech/results/m3_summary.json)")
    ap.add_argument("--models", default=None,
                    help="comma-separated HF model names (default: MECH_MODELS)")
    ap.add_argument("--depths", default="0,512",
                    help="comma-separated filler depths (default: 0,512)")
    ap.add_argument("--items", type=int, default=None,
                    help="cap items per condition per depth")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mech", "results"),
                    help="output directory (default: mech/results/)")
    return ap.parse_args()


def _model_short(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _bootstrap_ci(vals, rng, n=4000):
    import numpy as np
    v = np.asarray(vals, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.top_k > 5:
        print(f"[m4] --top-k must be in 1..5, got {args.top_k}")
        return 1
    os.makedirs(args.out, exist_ok=True)

    import numpy as np
    import torch

    from alias_inertia.determinism import environment_fingerprint, utc_now_iso
    from alias_inertia.mech.ablate import ablate_heads
    from alias_inertia.mech.env import MECH_MODELS, load_model
    from alias_inertia.mech.manifest import update_manifest
    from alias_inertia.mech.proxy import build_proxy_lexicon
    from alias_inertia.mech.stimuli_mech import build_mech_stimuli
    from transformers import AutoTokenizer

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MECH_MODELS)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    factors = [float(f) for f in args.factors.split(",") if f.strip()]
    rng = np.random.default_rng(args.seed)

    m3_json = args.m3_json or os.path.join(args.out, "m3_summary.json")
    if not os.path.exists(m3_json):
        print(f"[m4] m3_summary.json not found at {m3_json}; run mech_m3_heads.py first.")
        return 1
    with open(m3_json, encoding="utf-8") as fh:
        m3 = json.load(fh)

    tokenizer = AutoTokenizer.from_pretrained(models[0])
    stimuli_all = build_mech_stimuli(
        tokenizer, depths=tuple(depths), n_per_cell=30, seed=args.seed,
        pair_names=("numpy__pandas",),
    )
    lex = build_proxy_lexicon(tokenizer)

    all_records = []
    all_summaries = {}

    for model_name in models:
        short = _model_short(model_name)
        print(f"\n[m4] model: {model_name}")

        # Extract top-k heads from m3 ranking for this model.
        m3_model = m3.get("models", {}).get(model_name, {})
        top_heads_all = m3_model.get("top_heads", [])
        # Filter to prior-promoting (class == prior_promoting) and sort by abs_dla_swapped.
        prior_heads = [h for h in top_heads_all if h.get("class") == "prior_promoting"]
        prior_heads.sort(key=lambda x: -abs(x.get("abs_dla_swapped", 0)))
        top_k_heads = [(int(h["layer"]), int(h["head"])) for h in prior_heads[: args.top_k]]
        if not top_k_heads:
            print(f"  [m4] no prior-promoting heads found for {model_name}; skipping")
            continue
        print(f"  top-{args.top_k} prior-promoting heads: {top_k_heads}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tl = load_model(model_name, device=device)
        model_tl.eval()

        # Cap stimuli.
        stims = stimuli_all
        if args.items:
            cap: list = []
            cnt: dict[str, int] = {}
            for s in stims:
                if cnt.get(s.condition, 0) < args.items:
                    cnt[s.condition] = cnt.get(s.condition, 0) + 1
                    cap.append(s)
            stims = cap

        # Conditions of interest: swapped (effect) + conventional (collateral).
        swapped_stims = [s for s in stims if s.condition == "swapped"]
        conv_stims = [s for s in stims if s.condition == "conventional"]

        # Baseline (no ablation, factor=1.0).
        print("  baseline (factor=1.0) ...")
        baseline_sw = ablate_heads(model_tl, swapped_stims, top_k_heads, 1.0, lex)
        baseline_cv = ablate_heads(model_tl, conv_stims, top_k_heads, 1.0, lex)
        base_sw_pp = [r["proxy_pull"] for r in baseline_sw]
        base_cv_pp = [r["proxy_pull"] for r in baseline_cv]

        model_records = []
        model_table = {}
        for rec_list, label, base_pp in [
            (baseline_sw, "swapped_baseline", base_sw_pp),
            (baseline_cv, "conventional_baseline", base_cv_pp),
        ]:
            m, lo, hi = _bootstrap_ci(base_pp, rng)
            model_table[label] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(base_pp)}
            for r in rec_list:
                model_records.append({**r, "model": model_name, "factor": 1.0,
                                       "k": args.top_k, "ablated_heads": top_k_heads})

        # Ablation sweep.
        for factor in factors:
            print(f"  factor={factor} ...")
            sw_recs = ablate_heads(model_tl, swapped_stims, top_k_heads, factor, lex)
            cv_recs = ablate_heads(model_tl, conv_stims, top_k_heads, factor, lex)
            sw_pp = [r["proxy_pull"] for r in sw_recs]
            cv_pp = [r["proxy_pull"] for r in cv_recs]
            for cond, pp_list, rec_list in [("swapped", sw_pp, sw_recs),
                                             ("conventional", cv_pp, cv_recs)]:
                m, lo, hi = _bootstrap_ci(pp_list, rng)
                key = f"k{args.top_k}_f{factor}_{cond}"
                model_table[key] = {
                    "mean": m, "ci_low": lo, "ci_high": hi, "n": len(pp_list),
                    "factor": factor, "k": args.top_k, "condition": cond,
                }
                for r in rec_list:
                    model_records.append({**r, "model": model_name, "factor": factor,
                                           "k": args.top_k, "ablated_heads": top_k_heads})
            n_done = len(model_records)
            if n_done % 20 == 0:
                print(f"  {n_done} records")

        all_records.extend(model_records)
        all_summaries[model_name] = {
            "top_k": args.top_k,
            "ablated_heads": top_k_heads,
            "table": model_table,
            "n_records": len(model_records),
        }

        # Print table.
        print(f"\n  [m4 {short}] ablation summary (proxy_pull mean [95% CI]):")
        print(f"  {'key':40s} {'mean':>8} {'ci_low':>8} {'ci_high':>8} {'n':>5}")
        for k, v in model_table.items():
            print(f"  {k:40s} {v['mean']:>8.3f} {v['ci_low']:>8.3f} {v['ci_high']:>8.3f} {v['n']:>5}")

        del model_tl
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Write outputs (merge with prior per-model runs so split processes don't clobber) ----
    rec_path = os.path.join(args.out, "m4_records.jsonl")
    if os.path.exists(rec_path):
        kept = []
        with open(rec_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("model") not in models:
                    kept.append(row)
        all_records = kept + all_records
    with open(rec_path, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    sum_path = os.path.join(args.out, "m4_summary.json")
    if os.path.exists(sum_path):
        try:
            with open(sum_path, encoding="utf-8") as fh:
                prior = json.load(fh)
            for k, v in prior.get("models", {}).items():
                all_summaries.setdefault(k, v)
            # Config must describe ALL data in the file, not the last invocation.
            models = sorted(set(prior.get("config", {}).get("models", [])) | set(models))
        except (json.JSONDecodeError, KeyError):
            pass

    summary = {
        "timestamp_utc": utc_now_iso(),
        "config": {"models": models, "depths": depths, "top_k": args.top_k,
                   "factors": factors, "items_cap": args.items, "seed": args.seed},
        "n_records": len(all_records),
        "models": all_summaries,
        "environment": environment_fingerprint(),
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)

    print(f"\n[m4] {len(all_records)} records -> {os.path.relpath(rec_path, REPO_ROOT)}")
    print(f"[m4] summary -> {os.path.relpath(sum_path, REPO_ROOT)}")

    update_manifest("m4_ablate", {
        "models": models, "top_k": args.top_k, "factors": factors,
        "n_records": len(all_records),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
