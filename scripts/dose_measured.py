#!/usr/bin/env python
"""Dose effect on MEASURED corpus frequency (replaces the ordinal very-common/common/rare tiers).

For each swap pair the treatment alias has a measured corpus log-frequency from
results/corpus_freq.json (scripts/corpus_freq.py). We regress the per-row prior-pull on the
swapped indicator interacted with that continuous dose:

    pmo ~ is_swapped * dose_c + C(depth)      crossed RE (1|pair) + (1|model)

where pmo = logsumexp(L_prior) - logsumexp(L_other) and dose_c is the centered log corpus
frequency of the canonical ``import LIB as alias`` convention. The is_swapped:dose_c coefficient
is the gap's slope in nats of DiD per natural-log unit of corpus frequency. Reported with a
crossed-RE Wald CI and a pair-clustered bootstrap CI (robustness given few pair clusters).

  python scripts/dose_measured.py --config configs/full.yaml --dose log_conv_count
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from alias_inertia.lexicons import CANONICAL_ALIASES  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--corpus", default=os.path.join(REPO_ROOT, "results", "corpus_freq.json"))
    ap.add_argument("--dose", default="log_conv_count", choices=["log_conv_count", "log_odds", "p_canonical"])
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(args.corpus, encoding="utf-8") as fh:
        dose_tab = json.load(fh)["canonical_dose"]

    df = pd.read_parquet(os.path.join(REPO_ROOT, cfg["output"]["results"]))
    df["pmo"] = [float(r[f"logsumexp_{r['prior_lib']}"]) - float(r[f"logsumexp_{r['other_lib']}"])
                 for _, r in df.iterrows()]
    # dose of each row = measured corpus frequency of its treatment alias's canonical convention
    df["alias"] = df["prior_lib"].map(CANONICAL_ALIASES)
    df["dose"] = df["alias"].map(lambda a: dose_tab.get(a, {}).get(args.dose))
    d = df[df["condition"].isin(["swapped", "no_prior"])].dropna(subset=["dose"]).copy()
    d["is_swapped"] = (d["condition"] == "swapped").astype(float)
    d["dose"] = d["dose"].astype(float)
    d["dose_c"] = d["dose"] - d["dose"].mean()
    d["depth_f"] = d["depth_tokens_target"].astype(str)

    alias_dose = {a: float(dose_tab[a][args.dose]) for a in d["alias"].unique() if a in dose_tab}
    print("=" * 78)
    print(f"MEASURED dose regression (gap ~ {args.dose}) on {len(d)} rows, "
          f"{d['pair'].nunique()} pairs x {d['model'].nunique()} models")
    print("per-alias dose:", {k: round(v, 2) for k, v in sorted(alias_dose.items(), key=lambda kv: -kv[1])})
    print("=" * 78)

    import statsmodels.formula.api as smf

    d["grp"] = 1
    vc = {"pair": "0 + C(pair)", "model": "0 + C(model)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("pmo ~ is_swapped * dose_c + C(depth_f)", d, groups="grp",
                         vc_formula=vc, re_formula="0")
        mdf = md.fit(method="lbfgs", maxiter=2000)
    name = "is_swapped:dose_c"
    coef = float(mdf.params[name]); se = float(mdf.bse[name]); p = float(mdf.pvalues[name])
    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    print(f"\n(1) Mixed  pmo ~ is_swapped*dose_c + C(depth), RE (1|pair)+(1|model)")
    print(f"    dose slope (is_swapped:dose_c) = {coef:+.3f} nats per log-unit corpus freq "
          f"[Wald 95% {lo:+.3f},{hi:+.3f}] p={p:.4g}")

    def ols_slope(frame):
        m = smf.ols("pmo ~ is_swapped * dose_c + C(model) + C(depth_f)", frame).fit()
        return float(m.params[name])

    point_ols = ols_slope(d)
    pairs = d["pair"].unique()
    rng = np.random.default_rng(20260617)
    boots = []
    for _ in range(args.boot):
        samp = rng.choice(pairs, size=len(pairs), replace=True)
        fr = pd.concat([d[d["pair"] == p] for p in samp], ignore_index=True)
        if fr["dose"].nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boots.append(ols_slope(fr))
        except Exception:
            continue
    boots = np.array(boots)
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    print(f"\n(2) OLS slope = {point_ols:+.3f}; pair-clustered bootstrap 95% CI [{blo:+.3f},{bhi:+.3f}] "
          f"({len(boots)} boots, {len(pairs)} clusters, {100*(boots>0).mean():.1f}% > 0)")

    b_sw = float(mdf.params["is_swapped"])  # predicted gap at the mean dose (dose_c = 0)

    # Per-pair (and per-pair/model) DiD = mean(pmo|swapped) - mean(pmo|no_prior).
    def did(frame):
        s = frame[frame["is_swapped"] == 1]["pmo"].mean()
        n = frame[frame["is_swapped"] == 0]["pmo"].mean()
        return float(s - n)

    per_pair = {}
    for pr in d["pair"].unique():
        fp = d[d["pair"] == pr]
        alias = fp["alias"].iloc[0]
        per_pair[pr] = {"alias": alias, "dose": float(fp["dose"].iloc[0]), "did": did(fp),
                        "did_by_model": {m: did(fp[fp["model"] == m]) for m in fp["model"].unique()}}

    out = {
        "dose_variable": args.dose,
        "n_rows": int(len(d)), "n_pairs": int(d["pair"].nunique()), "n_models": int(d["model"].nunique()),
        "per_alias_dose": alias_dose,
        "mixed_model": {"slope": coef, "se": se, "p": p, "wald_ci": [lo, hi], "gap_at_mean_dose": b_sw,
                        "mean_dose": float(d["dose"].mean())},
        "ols_cluster_bootstrap": {"slope": point_ols, "ci": [float(blo), float(bhi)],
                                  "n_boot": int(len(boots)), "frac_positive": float((boots > 0).mean())},
        "per_pair": per_pair,
    }

    # Figure: per-(pair,model) DiD vs measured log corpus frequency, with the RE fit line.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean_dose = float(d["dose"].mean())
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        for pr, info in per_pair.items():
            xs = [info["dose"]] * len(info["did_by_model"])
            ax.scatter(xs, list(info["did_by_model"].values()), s=10, color="#7aa6c2", alpha=0.7,
                       zorder=2, edgecolors="none")
            ax.scatter([info["dose"]], [info["did"]], s=42, color="#1f4e79", zorder=3)
            ax.annotate(info["alias"], (info["dose"], info["did"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        xline = np.linspace(d["dose"].min(), d["dose"].max(), 50)
        ax.plot(xline, b_sw + coef * (xline - mean_dose), color="#c00000", lw=1.5,
                label=f"+{coef:.2f} nats / log-unit", zorder=1)
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_xlabel("log corpus frequency of the convention")
        ax.set_ylabel("DiD gap (nats)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        fig_path = os.path.join(REPO_ROOT, "figures", "full_dose_curve.png")
        fig.savefig(fig_path, dpi=200)
        print(f"figure -> {os.path.relpath(fig_path, REPO_ROOT)}")
    except Exception as e:  # pragma: no cover
        print("figure skipped:", type(e).__name__, str(e)[:120])
    with open(os.path.join(REPO_ROOT, "results", "dose_measured.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nsaved -> results/dose_measured.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
