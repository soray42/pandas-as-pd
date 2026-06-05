#!/usr/bin/env python
"""Dose effect, done right: a mixed/clustered regression on ALL rows that uses magnitude (not a
6-point rank correlation that collapsed 1683 rows -> 6 and threw away the 9.5-vs-3.4 separation).

Model (on swapped + no_prior rows; the DiD gap is their contrast):
    pmo ~ is_swapped * tier_c + C(depth)        with crossed random intercepts (1|pair) + (1|model)
where pmo = logsumexp(L_prior) - logsumexp(L_other), is_swapped in {0,1}, tier_c = tier_rank-2.
The **is_swapped:tier_c** coefficient is the gap's dose slope (nats of DiD gap per tier step);
gap(tier) = b_is_swapped + b_inter*tier_c (depth/model main effects cancel in the contrast).
pair is NOT a fixed effect (it is collinear with tier); it enters as a random intercept and as the
bootstrap cluster. Reports: (1) MixedLM crossed-RE Wald CI, (2) pair-clustered bootstrap CI on the
OLS slope (robustness given only 6 pairs).

  python scripts/dose_regression.py --config configs/full.yaml
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIER_RANK = {"very_common": 3, "common": 2, "rare": 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    df = pd.read_parquet(os.path.join(REPO_ROOT, cfg["output"]["results"]))
    df["pmo"] = [float(r[f"logsumexp_{r['prior_lib']}"]) - float(r[f"logsumexp_{r['other_lib']}"])
                 for _, r in df.iterrows()]
    d = df[df["condition"].isin(["swapped", "no_prior"])].copy()
    d["is_swapped"] = (d["condition"] == "swapped").astype(float)
    d["tier_rank"] = d["tier"].map(TIER_RANK).astype(float)
    d["tier_c"] = d["tier_rank"] - 2.0  # center: rare=-1, common=0, very_common=+1
    d["depth_f"] = d["depth_tokens_target"].astype(str)

    print("=" * 78)
    print(f"DOSE regression (gap ~ tier) on {len(d)} swapped+no_prior rows "
          f"({d['pair'].nunique()} pairs, {d['model'].nunique()} models)")
    print("=" * 78)

    import statsmodels.formula.api as smf

    # ---- (1) crossed-RE mixed model: (1|pair) + (1|model) ------------------------------
    d["grp"] = 1
    vc = {"pair": "0 + C(pair)", "model": "0 + C(model)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("pmo ~ is_swapped * tier_c + C(depth_f)", d, groups="grp",
                         vc_formula=vc, re_formula="0")
        mdf = md.fit(method="lbfgs", maxiter=2000)
    name = "is_swapped:tier_c"
    coef = float(mdf.params[name]); se = float(mdf.bse[name]); p = float(mdf.pvalues[name])
    lo, hi = coef - 1.96 * se, coef + 1.96 * se
    b_sw = float(mdf.params["is_swapped"])
    print("\n(1) Mixed model  pmo ~ is_swapped*tier_c + C(depth), RE: (1|pair)+(1|model)")
    print(f"    dose slope (is_swapped:tier_c) = {coef:+.3f} nats / tier-step  "
          f"[Wald 95% {lo:+.3f},{hi:+.3f}]  SE={se:.3f}  p={p:.4g}")
    print(f"    implied gap: rare={b_sw - coef:+.2f}  common={b_sw:+.2f}  very_common={b_sw + coef:+.2f}")
    print(f"    -> slope CI {'EXCLUDES 0 (dose effect significant)' if lo > 0 else 'includes 0'}")

    # ---- (2) pair-clustered bootstrap on the OLS slope (robustness; only 6 clusters) ----
    def ols_slope(frame):
        m = smf.ols("pmo ~ is_swapped * tier_c + C(model) + C(depth_f)", frame).fit()
        return float(m.params[name])

    point_ols = ols_slope(d)
    pairs = d["pair"].unique()
    rng = np.random.default_rng(20260604)
    boots = []
    for _ in range(args.boot):
        samp = rng.choice(pairs, size=len(pairs), replace=True)
        fr = pd.concat([d[d["pair"] == p] for p in samp], ignore_index=True)
        if fr["tier_rank"].nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boots.append(ols_slope(fr))
        except Exception:
            continue
    boots = np.array(boots)
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    print(f"\n(2) OLS slope = {point_ols:+.3f}; pair-clustered bootstrap 95% CI "
          f"[{blo:+.3f},{bhi:+.3f}]  (n_boot={len(boots)}, {len(pairs)} pair-clusters)")
    print(f"    -> {'EXCLUDES 0' if blo > 0 else 'includes 0'} "
          f"({100*(boots>0).mean():.1f}% of bootstrap slopes > 0)")

    out = {
        "n_rows": int(len(d)), "n_pairs": int(d["pair"].nunique()), "n_models": int(d["model"].nunique()),
        "mixed_model": {"dose_slope": coef, "se": se, "p": p, "wald_ci": [lo, hi],
                        "gap_rare": b_sw - coef, "gap_common": b_sw, "gap_very_common": b_sw + coef},
        "ols_cluster_bootstrap": {"slope": point_ols, "ci": [float(blo), float(bhi)],
                                  "n_boot": int(len(boots)), "frac_positive": float((boots > 0).mean())},
    }
    with open(os.path.join(REPO_ROOT, "results", "dose_regression.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nsaved -> results/dose_regression.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
