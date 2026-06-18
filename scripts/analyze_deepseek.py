#!/usr/bin/env python
"""Analyze the DeepSeek API probe into the numbers cited in the paper.

Reads results/deepseek_raw.jsonl and reports, per mode (non-thinking / thinking):
  forced_choice : rate of choosing the prior library's (non-existent) method, swapped vs
                  no-prior, with the difference (the prior-pull DiD) and a pair-clustered
                  bootstrap CI; plus overall accuracy.
  generation    : broken-on-bound rate and prior-only rate by condition.
  verbal        : recognition accuracy and the rate of naming the prior library, by condition.
  depth curve   : the swapped-condition forced_choice and generation rates by in-context depth,
                  up to the deepest (128k) bin, to show whether the effect survives long context.

  python scripts/analyze_deepseek.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_GLOB = os.path.join(REPO_ROOT, "results", "deepseek_raw*.jsonl")
OUT = os.path.join(REPO_ROOT, "results", "deepseek_analysis.json")
CONDS = ["conventional", "swapped", "no_prior"]


def load() -> pd.DataFrame:
    """Load and concatenate every deepseek_raw*.jsonl (core + salience + nonce passes), then
    de-duplicate identical probes. salience defaults to False for the original schema."""
    rows = []
    for path in sorted(glob.glob(RAW_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "salience" not in df.columns:
        df["salience"] = False
    df["salience"] = df["salience"].fillna(False).astype(bool)
    # A probe is uniquely identified by these; later files do not override earlier ones.
    df = df.drop_duplicates(subset=["stimulus_id", "mode", "task", "salience"], keep="first")
    return df.reset_index(drop=True)


def boot_rate(df, col, rng, n_boot=5000):
    """Pair-clustered bootstrap mean of a 0/1 column (drops NaN), returns (mean, lo, hi, n)."""
    d = df[["pair", col]].dropna(subset=[col])
    if len(d) == 0:
        return None, None, None, 0
    pairs = d["pair"].unique()
    by = {p: d[d["pair"] == p][col].astype(float).values for p in pairs}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        vals = np.concatenate([by[p] for p in chosen])
        boots.append(vals.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d[col].astype(float).mean()), float(lo), float(hi), int(len(d))


def boot_did(df_a, df_b, col, rng, n_boot=5000):
    """Pair-clustered bootstrap of rate(A) - rate(B) for a 0/1 column."""
    a = df_a[["pair", col]].dropna(subset=[col])
    b = df_b[["pair", col]].dropna(subset=[col])
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    pairs = sorted(set(a["pair"]) | set(b["pair"]))
    aby = {p: a[a["pair"] == p][col].astype(float).values for p in pairs}
    bby = {p: b[b["pair"] == p][col].astype(float).values for p in pairs}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        av = np.concatenate([aby[p] for p in chosen if len(aby[p])])
        bv = np.concatenate([bby[p] for p in chosen if len(bby[p])])
        if len(av) and len(bv):
            boots.append(av.mean() - bv.mean())
    point = float(a[col].astype(float).mean() - b[col].astype(float).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def rate_block(df, col, rng):
    out = {}
    for c in CONDS:
        m, lo, hi, n = boot_rate(df[df["condition"] == c], col, rng)
        out[c] = {"rate": m, "ci_lo": lo, "ci_hi": hi, "n": n}
    # DiD swapped - no_prior (the prior effect, controlling for base rate)
    p, lo, hi = boot_did(df[df["condition"] == "swapped"], df[df["condition"] == "no_prior"], col, rng)
    out["did_swapped_minus_no_prior"] = {"value": p, "ci_lo": lo, "ci_hi": hi}
    return out


def main() -> int:
    if not glob.glob(RAW_GLOB):
        print("missing results/deepseek_raw*.jsonl; run scripts/run_deepseek.py first")
        return 1
    df_all = load()
    df = df_all[~df_all["salience"]].reset_index(drop=True)  # main blocks: no intervention
    rng = np.random.default_rng(20260617)
    modes = sorted(df["mode"].unique())
    out = {"n_records": int(len(df_all)), "n_main": int(len(df)),
           "modes": modes, "model": df["model"].iloc[0]}

    # finish/parse health (thinking mode can truncate before answering)
    health = {}
    for mode in modes:
        dm = df[df["mode"] == mode]
        health[mode] = {
            "n": int(len(dm)),
            "fc_unparsed": int(((dm["task"] == "forced_choice") & dm["choice"].isna()).sum()),
            "gen_no_attr": int(((dm["task"] == "generation") & dm["attribute"].isna()).sum()),
            "len_truncated": int((dm["finish_reason"] == "length").sum()),
        }
    out["health"] = health

    out["forced_choice"] = {}
    out["generation"] = {}
    out["verbal"] = {}
    for mode in modes:
        dm = df[df["mode"] == mode]
        fc = dm[dm["task"] == "forced_choice"]
        out["forced_choice"][mode] = {
            "chose_prior_lib": rate_block(fc, "chose_prior_lib", rng),
            "correct": rate_block(fc, "correct", rng),
        }
        gen = dm[dm["task"] == "generation"]
        out["generation"][mode] = {
            "broken_on_bound": rate_block(gen, "broken_on_bound", rng),
            "prior_only": rate_block(gen, "prior_only", rng),
        }
        vb = dm[dm["task"] == "verbal"]
        out["verbal"][mode] = {
            "correct": rate_block(vb, "correct", rng),
            "named_prior": rate_block(vb, "named_prior", rng),
        }

    # Depth curve: swapped condition, by depth target, per mode.
    depth_curve = {}
    for mode in modes:
        dm = df[(df["mode"] == mode) & (df["condition"] == "swapped")]
        per_depth = {}
        for depth in sorted(dm["depth_target"].unique()):
            dd = dm[dm["depth_target"] == depth]
            fc = dd[dd["task"] == "forced_choice"]
            gen = dd[dd["task"] == "generation"]
            fcm, fclo, fchi, fcn = boot_rate(fc, "chose_prior_lib", rng)
            gm, glo, ghi, gn = boot_rate(gen, "broken_on_bound", rng)
            # actual in-context length seen by the API at this bin (median)
            api_tok = dd["prompt_tokens_api"]
            per_depth[str(int(depth))] = {
                "fc_chose_prior": fcm, "fc_ci": [fclo, fchi], "fc_n": fcn,
                "gen_broken": gm, "gen_ci": [glo, ghi], "gen_n": gn,
                "median_prompt_tokens_api": int(api_tok.median()) if len(api_tok) else None,
            }
        depth_curve[mode] = per_depth
    out["depth_curve_swapped"] = depth_curve

    # Salience intervention: swapped, explicit binding cue (True) vs none (False), per mode.
    salience = {}
    for mode in modes:
        sw = df_all[(df_all["mode"] == mode) & (df_all["condition"] == "swapped")]
        block = {}
        for label, task, col in [("forced_choice", "forced_choice", "chose_prior_lib"),
                                 ("generation", "generation", "broken_on_bound"),
                                 ("verbal", "verbal", "correct")]:
            ns = sw[(sw["task"] == task) & (~sw["salience"])]
            sa = sw[(sw["task"] == task) & (sw["salience"])]
            if len(sa) == 0:
                continue
            m_ns, lo_ns, hi_ns, n_ns = boot_rate(ns, col, rng)
            m_sa, lo_sa, hi_sa, n_sa = boot_rate(sa, col, rng)
            block[label] = {
                "no_cue": {"rate": m_ns, "ci": [lo_ns, hi_ns], "n": n_ns},
                "with_cue": {"rate": m_sa, "ci": [lo_sa, hi_sa], "n": n_sa},
            }
        if block:
            salience[mode] = block
    if salience:
        out["salience_intervention_swapped"] = salience

    # Nonce-alias robustness: no-prior control rates per nonce alias (non-salience).
    nonce = {}
    npr = df[df["condition"] == "no_prior"]
    aliases = sorted(npr["alias"].unique())
    if len(aliases) > 1:
        for mode in modes:
            per_alias = {}
            for al in aliases:
                a = npr[(npr["mode"] == mode) & (npr["alias"] == al)]
                fcm, _, _, fcn = boot_rate(a[a["task"] == "forced_choice"], "chose_prior_lib", rng)
                gm, _, _, gn = boot_rate(a[a["task"] == "generation"], "broken_on_bound", rng)
                per_alias[al] = {"fc_chose_prior": fcm, "fc_n": fcn, "gen_broken": gm, "gen_n": gn}
            nonce[mode] = per_alias
        out["nonce_alias_robustness_no_prior"] = nonce

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"records={len(df)}  model={out['model']}  modes={modes}")
    for mode in modes:
        print(f"\n=== mode={mode} ===")
        h = health[mode]
        print(f"  health: fc_unparsed={h['fc_unparsed']} gen_no_attr={h['gen_no_attr']} truncated={h['len_truncated']}")
        fc = out["forced_choice"][mode]["chose_prior_lib"]
        print("  forced_choice chose-prior-method rate:")
        for c in CONDS:
            b = fc[c]
            if b["rate"] is not None:
                print(f"    {c:13} {b['rate']:.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] n={b['n']}")
        did = fc["did_swapped_minus_no_prior"]
        if did["value"] is not None:
            print(f"    DiD swapped-noprior = {did['value']:+.3f} [{did['ci_lo']:+.3f},{did['ci_hi']:+.3f}]")
        gb = out["generation"][mode]["broken_on_bound"]
        print("  generation broken-on-bound rate:")
        for c in CONDS:
            b = gb[c]
            if b["rate"] is not None:
                print(f"    {c:13} {b['rate']:.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] n={b['n']}")
        vb = out["verbal"][mode]["correct"]
        print("  verbal accuracy:")
        for c in CONDS:
            b = vb[c]
            if b["rate"] is not None:
                print(f"    {c:13} {b['rate']:.3f} n={b['n']}")
    print("\n  depth curve (swapped):")
    for mode in modes:
        print(f"    mode={mode}")
        for depth, v in out["depth_curve_swapped"][mode].items():
            print(f"      depth~{depth:>7} (api~{v['median_prompt_tokens_api']}): "
                  f"fc_chose_prior={v['fc_chose_prior']} gen_broken={v['gen_broken']} "
                  f"(n_fc={v['fc_n']}, n_gen={v['gen_n']})")
    if "salience_intervention_swapped" in out:
        print("\n  salience intervention (swapped, with-cue vs no-cue):")
        for mode, block in out["salience_intervention_swapped"].items():
            for task, b in block.items():
                print(f"    {mode:8} {task:13} no_cue={b['no_cue']['rate']} -> with_cue={b['with_cue']['rate']}")
    if "nonce_alias_robustness_no_prior" in out:
        print("\n  nonce-alias robustness (no_prior, per alias):")
        for mode, per in out["nonce_alias_robustness_no_prior"].items():
            for al, v in per.items():
                print(f"    {mode:8} {al}: fc_chose_prior={v['fc_chose_prior']} gen_broken={v['gen_broken']}")
    print(f"\nsaved -> {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
