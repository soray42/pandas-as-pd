#!/usr/bin/env python
"""FULL-RUN decision/analysis: DiD gap across pairs x depths x models, the dose curve (gap vs
prior-strength tier), generation + broken-call rates per condition, base-vs-instruct & size
trends, with clustered bootstrap CIs. Writes figures + an honest verdict.

  python scripts/analyze.py --config configs/full.yaml

Framing pivot honoured: we CHARACTERISE the distance dependence (does growth appear at long
range, or is the effect flat?), we do NOT assert distance-growth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIER_ORDER = ["very_common", "common", "rare"]


def prior_minus_other(df: pd.DataFrame) -> pd.Series:
    """Row-wise L(prior_lib) - L(other_lib). = prior_pull for swapped/no_prior; ceiling margin
    for conventional. Uses each row's own pair libs (columns differ across pairs)."""
    def one(r):
        cp, co = f"logsumexp_{r['prior_lib']}", f"logsumexp_{r['other_lib']}"
        return float(r[cp]) - float(r[co])
    return df.apply(one, axis=1)


def _gap(d: pd.DataFrame) -> float:
    sw = d.loc[d.condition == "swapped", "pmo"]
    npr = d.loc[d.condition == "no_prior", "pmo"]
    if len(sw) == 0 or len(npr) == 0:
        return float("nan")
    return float(sw.mean() - npr.mean())


def clustered_gap_ci(d: pd.DataFrame, n_boot: int, ci: float, rng) -> tuple:
    """Gap = mean(swapped pmo) - mean(no_prior pmo), with a bootstrap CI clustered on `pair`
    (random effect), resampling rows within each chosen pair."""
    pairs = d["pair"].unique()
    if len(pairs) == 0:
        return float("nan"), float("nan"), float("nan")
    point = _gap(d)
    by_pair = {p: d[d["pair"] == p] for p in pairs}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        parts = [by_pair[p].iloc[rng.integers(0, len(by_pair[p]), size=len(by_pair[p]))] for p in chosen]
        boots.append(_gap(pd.concat(parts)))
    boots = np.array([b for b in boots if not np.isnan(b)])
    if len(boots) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return point, float(lo), float(hi)


def dose_regression(df, n_boot=2000, seed=20260604):
    """Dose effect on ALL swapped+no_prior rows (uses magnitude, not a rank corr on 6 collapsed
    points). The is_swapped:tier interaction = the DiD gap's dose slope. Reports the crossed-RE
    mixed model (1|pair)+(1|model) Wald CI AND a pair-clustered bootstrap CI (honest at 6 clusters)."""
    import warnings

    import statsmodels.formula.api as smf

    d = df[df["condition"].isin(["swapped", "no_prior"])].copy()
    if d["tier"].nunique() < 2 or d["pair"].nunique() < 2:
        return {"error": "insufficient tiers/pairs"}
    d["is_swapped"] = (d["condition"] == "swapped").astype(float)
    d["tier_c"] = d["tier_rank"].astype(float) - 2.0  # rare=-1, common=0, very_common=+1
    d["depth_f"] = d["depth_tokens_target"].astype(str)
    name = "is_swapped:tier_c"
    out = {"n_rows": int(len(d)), "n_pairs": int(d["pair"].nunique())}

    try:
        d["grp"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("pmo ~ is_swapped * tier_c + C(depth_f)", d, groups="grp",
                             vc_formula={"pair": "0 + C(pair)", "model": "0 + C(model)"},
                             re_formula="0").fit(method="lbfgs", maxiter=2000)
        coef, se, b_sw = float(md.params[name]), float(md.bse[name]), float(md.params["is_swapped"])
        out["mixed"] = {"slope": coef, "se": se, "p": float(md.pvalues[name]),
                        "wald_ci": [coef - 1.96 * se, coef + 1.96 * se],
                        "gap_rare": b_sw - coef, "gap_common": b_sw, "gap_very_common": b_sw + coef}
    except Exception as e:  # pragma: no cover
        out["mixed"] = {"error": str(e)}

    def ols_slope(fr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(smf.ols("pmo ~ is_swapped * tier_c + C(model) + C(depth_f)", fr).fit().params[name])

    try:
        rng = np.random.default_rng(seed)
        pairs = d["pair"].unique()
        point = ols_slope(d)
        boots = []
        for _ in range(n_boot):
            fr = pd.concat([d[d["pair"] == p] for p in rng.choice(pairs, size=len(pairs), replace=True)],
                           ignore_index=True)
            if fr["tier_rank"].nunique() < 2:
                continue
            try:
                boots.append(ols_slope(fr))
            except Exception:
                continue
        boots = np.array(boots)
        out["cluster_bootstrap"] = {"slope": point, "ci": list(np.percentile(boots, [2.5, 97.5])),
                                    "n_boot": int(len(boots)), "frac_positive": float((boots > 0).mean()),
                                    "n_pair_clusters": int(len(pairs))}
    except Exception as e:  # pragma: no cover
        out["cluster_bootstrap"] = {"error": str(e)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    acfg = cfg["analysis"]
    n_boot, ci = int(acfg["bootstrap_n"]), float(acfg["ci"])
    rng = np.random.default_rng(int(acfg["bootstrap_seed"]))
    figdir = os.path.join(REPO_ROOT, acfg["figure_dir"])
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_parquet(os.path.join(REPO_ROOT, cfg["output"]["results"]))
    df["pmo"] = prior_minus_other(df)
    gens = _load_jsonl(os.path.join(REPO_ROOT, cfg["output"]["generations"]))

    report = {"n_rows": int(len(df)), "models": sorted(df["model"].unique().tolist()),
              "pairs": sorted(df["pair"].unique().tolist()), "depths": sorted(int(x) for x in df["depth_tokens_target"].unique())}
    print("=" * 80)
    print(f"alias-inertia FULL analysis | rows={len(df)} models={len(report['models'])} "
          f"pairs={len(report['pairs'])} depths={report['depths']}")
    print("=" * 80)

    # ---- (A) overall + per-model DiD gap ----------------------------------------------
    g, lo, hi = clustered_gap_ci(df, n_boot, ci, rng)
    report["gap_overall"] = {"gap": g, "ci_lo": lo, "ci_hi": hi}
    print(f"\n(A) OVERALL DiD gap (Swapped-No_prior), clustered on pair: {g:+.3f} [{lo:+.3f},{hi:+.3f}]")
    report["gap_by_model"] = {}
    print("    per model:")
    for m in report["models"]:
        gm, lm, hm = clustered_gap_ci(df[df.model == m], n_boot, ci, rng)
        sz = float(df[df.model == m]["size_b"].iloc[0]); var = df[df.model == m]["variant"].iloc[0]
        report["gap_by_model"][m] = {"gap": gm, "ci_lo": lm, "ci_hi": hm, "size_b": sz, "variant": var}
        print(f"      {m:24} ({sz:>4}B {var:8}): {gm:+.3f} [{lm:+.3f},{hm:+.3f}]")

    # ---- (B) DOSE curve: gap vs prior-strength tier -----------------------------------
    print("\n(B) DOSE curve -- DiD gap by prior-strength tier (pooled over models, pairs, depths):")
    report["dose"] = {}
    for tier in TIER_ORDER:
        dt = df[df.tier == tier]
        if len(dt) == 0:
            continue
        gt, lt, ht = clustered_gap_ci(dt, n_boot, ci, rng)
        report["dose"][tier] = {"gap": gt, "ci_lo": lt, "ci_hi": ht, "pairs": sorted(dt["pair"].unique().tolist())}
        print(f"      {tier:12}: gap={gt:+.3f} [{lt:+.3f},{ht:+.3f}]  pairs={report['dose'][tier]['pairs']}")
    # ordinal correlation: per-pair gap vs tier_rank
    pair_gap = []
    for p in report["pairs"]:
        dp = df[df.pair == p]
        pair_gap.append((p, _gap(dp), int(dp["tier_rank"].iloc[0]), dp["tier"].iloc[0]))
    report["pair_gaps"] = [{"pair": p, "gap": g_, "tier_rank": tr, "tier": t} for p, g_, tr, t in pair_gap]
    # Dose effect tested properly: regression on ALL rows (magnitude + crossed REs), NOT a rank
    # correlation on 6 collapsed points.
    report["dose_regression"] = dose_regression(df, n_boot=int(acfg.get("dose_boot", 2000)),
                                                seed=int(acfg["bootstrap_seed"]))
    dr = report["dose_regression"]
    cb, mm = dr.get("cluster_bootstrap", {}), dr.get("mixed", {})
    if cb.get("ci"):
        print(f"    DOSE SLOPE (is_swapped:tier) = {cb['slope']:+.3f} nats/tier-step | "
              f"pair-clustered bootstrap 95% CI [{cb['ci'][0]:+.3f},{cb['ci'][1]:+.3f}] "
              f"({100*cb['frac_positive']:.0f}% boots>0) | mixed-model Wald "
              f"[{mm.get('wald_ci', [float('nan'), float('nan')])[0]:+.3f},"
              f"{mm.get('wald_ci', [float('nan'), float('nan')])[1]:+.3f}]")

    # ---- distance characterisation: gap vs depth, per model ---------------------------
    print("\n    distance: DiD gap vs depth (per model):")
    report["gap_by_model_depth"] = {}
    for m in report["models"]:
        dm = df[df.model == m]
        per = {}
        for d_ in sorted(int(x) for x in dm["depth_tokens_target"].unique()):
            gd, ld, hd = clustered_gap_ci(dm[dm.depth_tokens_target == d_], n_boot, ci, rng)
            per[d_] = {"gap": gd, "ci_lo": ld, "ci_hi": hd}
        report["gap_by_model_depth"][m] = per

    # ---- (C) generation + broken-call arms --------------------------------------------
    report["generation"] = {}
    if gens is not None and len(gens):
        gdf = pd.DataFrame(gens)
        print("\n(C) GENERATION arm -- completion class rate per condition (pooled over models/pairs):")
        for cond in ["conventional", "swapped", "no_prior"]:
            sub = gdf[gdf.condition == cond]
            if len(sub) == 0:
                continue
            rates = sub["gen_klass"].value_counts(normalize=True).to_dict()
            broke = sub[sub.validity_status.isin(["resolves", "broken"])]
            brate = float((broke.validity_status == "broken").mean()) if len(broke) else float("nan")
            report["generation"][cond] = {"n": int(len(sub)),
                                          "klass_rates": {k: float(v) for k, v in rates.items()},
                                          "broken_call_rate": brate, "n_resolvable": int(len(broke))}
            pr = rates.get("prior_style", 0.0); bo = rates.get("bound_style", 0.0); ot = rates.get("other", 0.0)
            print(f"      {cond:12}: prior={pr:.2f} bound={bo:.2f} other={ot:.2f} | "
                  f"broken-call={brate:.2f} (n={len(sub)}, resolvable={len(broke)})")
    else:
        print("\n(C) GENERATION arm: no generations found.")

    # ---- (D) base vs instruct + size trend --------------------------------------------
    print("\n(D) base-vs-instruct (matched family+size) & size trend:")
    report["base_vs_instruct"] = []
    md = df.groupby(["family", "size_b", "variant"]).size().reset_index()[["family", "size_b", "variant"]]
    for (fam, sz), grp in md.groupby(["family", "size_b"]):
        variants = set(grp["variant"])
        if {"base", "instruct"} <= variants:
            gb, _, _ = clustered_gap_ci(df[(df.family == fam) & (df.size_b == sz) & (df.variant == "base")], n_boot, ci, rng)
            gi, _, _ = clustered_gap_ci(df[(df.family == fam) & (df.size_b == sz) & (df.variant == "instruct")], n_boot, ci, rng)
            report["base_vs_instruct"].append({"family": fam, "size_b": float(sz), "gap_base": gb, "gap_instruct": gi})
            print(f"      {fam} {sz}B: gap base={gb:+.3f} vs instruct={gi:+.3f} (delta={gi-gb:+.3f})")
    size_trend = sorted({float(s) for s in df["size_b"]})
    report["size_trend"] = []
    for sz in size_trend:
        gs, ls, hs = clustered_gap_ci(df[df.size_b == sz], n_boot, ci, rng)
        report["size_trend"].append({"size_b": sz, "gap": gs, "ci_lo": ls, "ci_hi": hs})
    print("    gap vs size: " + ", ".join(f"{x['size_b']}B={x['gap']:+.2f}" for x in report["size_trend"]))

    # ---- verdict -----------------------------------------------------------------------
    verdict, detail = _verdict(report)
    report["verdict"], report["verdict_detail"] = verdict, detail
    print("\n" + "=" * 80)
    print(f"VERDICT: {verdict}")
    for line in detail:
        print("  - " + line)
    print("=" * 80)

    _figures(df, gens, report, figdir)
    analysis_path = os.path.join(REPO_ROOT, cfg["output"]["results"].replace(".parquet", "_analysis.json"))
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=float)
    print(f"figures -> {acfg['figure_dir']}/   analysis -> {os.path.relpath(analysis_path, REPO_ROOT)}")
    return 0


def _verdict(report) -> tuple:
    detail = []
    g = report["gap_overall"]
    held = g["ci_lo"] > 0
    detail.append(f"DiD gap (Swapped-No_prior) overall = {g['gap']:+.2f} [{g['ci_lo']:+.2f},{g['ci_hi']:+.2f}] "
                  + ("-> HELD: prior reassertion is real, net of generic rot." if held
                     else "-> NULL/weak: gap CI includes 0."))
    cb = report.get("dose_regression", {}).get("cluster_bootstrap", {})
    if cb.get("ci"):
        sig = cb["ci"][0] > 0
        detail.append(f"DOSE: gap dose-slope = {cb['slope']:+.2f} nats/tier-step "
                      f"[pair-clustered bootstrap {cb['ci'][0]:+.2f},{cb['ci'][1]:+.2f}; "
                      f"{100*cb['frac_positive']:.0f}% boots>0] "
                      + ("-> SIGNIFICANT: gap tracks prior strength (magnitude-aware, all rows)."
                         if sig else "-> CI includes 0."))
    # distance: does any model show CI-separated growth from min to max depth?
    grew = []
    for m, per in report["gap_by_model_depth"].items():
        ds = sorted(per)
        if len(ds) >= 2:
            lo_d, hi_d = per[ds[0]], per[ds[-1]]
            if hi_d["ci_lo"] > lo_d["ci_hi"]:
                grew.append((m, ds[0], ds[-1]))
    detail.append("DISTANCE: " + (f"growth (CI-separated) seen in {len(grew)} model(s): "
                  + ", ".join(m for m, _, _ in grew) if grew
                  else "no CI-separated growth from shortest to longest bin in any model -> immediate & distance-robust (pivot upheld)."))
    gen = report.get("generation", {})
    if "swapped" in gen and "no_prior" in gen:
        sp_prior = gen["swapped"]["klass_rates"].get("prior_style", 0.0)
        sw_break = gen["swapped"].get("broken_call_rate", float("nan"))
        detail.append(f"BEHAVIOR: swapped completions prior-style rate={sp_prior:.2f}, broken-call rate={sw_break:.2f} "
                      "-> the prior shows up in generated code and (often) would not resolve under the binding.")
    bvi = report.get("base_vs_instruct", [])
    if bvi:
        deltas = [x["gap_instruct"] - x["gap_base"] for x in bvi]
        detail.append(f"BASE-vs-INSTRUCT: mean gap delta (instruct-base) = {np.mean(deltas):+.2f} over {len(bvi)} matched pairs.")
    return ("EFFECT HELD (pivot upheld)" if held else "NULL / WEAK"), detail


def _load_jsonl(path):
    if not os.path.isfile(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _figures(df, gens, report, figdir):
    # Fig 1: dose curve (gap vs tier)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs, ys, los, his = [], [], [], []
    for i, t in enumerate(TIER_ORDER):
        if t in report["dose"]:
            d = report["dose"][t]
            xs.append(i); ys.append(d["gap"]); los.append(d["gap"] - d["ci_lo"]); his.append(d["ci_hi"] - d["gap"])
    ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=4, color="#d95f02")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(TIER_ORDER))); ax.set_xticklabels(TIER_ORDER)
    ax.set_xlabel("alias prior-strength tier"); ax.set_ylabel("DiD gap (Swapped - No_prior)")
    ax.set_title("Dose curve: prior-pull gap vs prior strength")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "full_dose_curve.png"), dpi=150); plt.close(fig)

    # Fig 2: gap vs depth, one line per model
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, per in report["gap_by_model_depth"].items():
        ds = sorted(per)
        ax.plot([max(d, 1) for d in ds], [per[d]["gap"] for d in ds], marker="o", label=m)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xscale("symlog")
    ax.set_xlabel("distance import->usage (tokens, symlog)"); ax.set_ylabel("DiD gap")
    ax.set_title("Distance characterisation: gap vs depth by model")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "full_gap_vs_depth.png"), dpi=150); plt.close(fig)

    # Fig 3: generation class rates + broken-call per condition
    if gens:
        gdf = pd.DataFrame(gens)
        conds = ["conventional", "swapped", "no_prior"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        klasses = ["prior_style", "bound_style", "other", "empty"]
        bottom = np.zeros(len(conds))
        for kl in klasses:
            vals = [float((gdf[gdf.condition == c]["gen_klass"] == kl).mean()) if len(gdf[gdf.condition == c]) else 0 for c in conds]
            a1.bar(conds, vals, bottom=bottom, label=kl); bottom += np.array(vals)
        a1.set_ylabel("rate"); a1.set_title("Generation class per condition"); a1.legend(fontsize=8)
        brates = []
        for c in conds:
            sub = gdf[(gdf.condition == c) & (gdf.validity_status.isin(["resolves", "broken"]))]
            brates.append(float((sub.validity_status == "broken").mean()) if len(sub) else 0.0)
        a2.bar(conds, brates, color="#7570b3"); a2.set_ylim(0, 1)
        a2.set_ylabel("broken-call rate"); a2.set_title("Would-not-resolve-under-binding rate")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "full_generation_arm.png"), dpi=150); plt.close(fig)

    # Fig 4: size trend + base/instruct
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    st = report["size_trend"]
    ax.errorbar([x["size_b"] for x in st], [x["gap"] for x in st],
                yerr=[[x["gap"] - x["ci_lo"] for x in st], [x["ci_hi"] - x["gap"] for x in st]],
                marker="s", capsize=4, color="#1b9e77", label="all models")
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xlabel("model size (B params, log)"); ax.set_ylabel("DiD gap")
    ax.set_title("Size trend (and base/instruct)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "full_size_trend.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
