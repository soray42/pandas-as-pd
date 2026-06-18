#!/usr/bin/env python
"""Analyze the extended DeepSeek N-run forced-choice (graded per-item P(prior) over 12 aliases).

Each record carries p_prior_lib = fraction of N temperature>0 draws that chose the prior
library's method (a graded estimate that does not need arbitrary-continuation log-probs). We
report, per alias: P(prior) under swap and the no-prior control (averaged over nonce aliases),
their difference (the prior-pull), and the relationship to measured corpus frequency. With the
salience pass we report the swapped P(prior) with vs without an explicit binding cue.

  python scripts/analyze_deepseek_ext.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia.lexicons import CANONICAL_ALIASES  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN = os.path.join(REPO_ROOT, "results", "deepseek_ext_nrun.jsonl")
SAL = os.path.join(REPO_ROOT, "results", "deepseek_ext_nrun_salience.jsonl")
CORP = os.path.join(REPO_ROOT, "results", "corpus_freq.json")
OUT = os.path.join(REPO_ROOT, "results", "deepseek_ext_analysis.json")


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def pair_p(recs, pair, cond):
    vals = [r["p_prior_lib"] for r in recs
            if r["pair"] == pair and r["condition"] == cond and r.get("p_prior_lib") is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    recs = load(MAIN)
    if not recs:
        print("missing results/deepseek_ext_nrun.jsonl"); return 1
    dose = json.load(open(CORP, encoding="utf-8"))["canonical_dose"]
    pairs = sorted({r["pair"] for r in recs})

    per_pair = {}
    for pair in pairs:
        alias = CANONICAL_ALIASES[pair.split("__")[0]]
        sw = pair_p(recs, pair, "swapped")
        npr = pair_p(recs, pair, "no_prior")
        conv = pair_p(recs, pair, "conventional")
        d = dose.get(alias, {}).get("log_conv_count")
        per_pair[pair] = {"alias": alias, "dose": d, "p_swapped": sw, "p_no_prior": npr,
                          "p_conventional": conv,
                          "did": (sw - npr) if (sw is not None and npr is not None) else None}

    # dose-response across aliases (DiD ~ dose); bootstrap over pairs.
    pts = [(v["dose"], v["did"]) for v in per_pair.values() if v["dose"] is not None and v["did"] is not None]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    r = float(np.corrcoef(x, y)[0, 1]); slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(20260617)
    bslopes = []
    for _ in range(5000):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) > 1:
            bslopes.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    blo, bhi = np.percentile(bslopes, [2.5, 97.5])

    # overall DiD (mean over aliases) + bootstrap
    dids = y
    bmean = [float(np.mean(dids[rng.integers(0, len(dids), len(dids))])) for _ in range(5000)]
    mlo, mhi = np.percentile(bmean, [2.5, 97.5])

    out = {
        "n_aliases": len(pairs),
        "per_pair": per_pair,
        "overall_did_mean": float(np.mean(dids)), "overall_did_ci": [float(mlo), float(mhi)],
        "dose_response": {"pearson_r": r, "slope": slope, "slope_ci": [float(blo), float(bhi)],
                          "frac_positive": float(np.mean(np.array(bslopes) > 0))},
    }

    sal = load(SAL)
    if sal:
        salience = {}
        for pair in sorted({r["pair"] for r in sal}):
            alias = CANONICAL_ALIASES[pair.split("__")[0]]
            cue = pair_p(sal, pair, "swapped")
            nocue = per_pair.get(pair, {}).get("p_swapped")
            if cue is not None and nocue is not None:
                salience[pair] = {"alias": alias, "p_swapped_no_cue": nocue, "p_swapped_with_cue": cue}
        if salience:
            ncs = [v["p_swapped_no_cue"] for v in salience.values()]
            cs = [v["p_swapped_with_cue"] for v in salience.values()]
            out["salience"] = {"per_pair": salience,
                               "mean_no_cue": float(np.mean(ncs)),
                               "mean_with_cue": float(np.mean(cs))}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{len(pairs)} aliases | overall swapped-vs-noprior DiD in P(prior) = "
          f"{out['overall_did_mean']:+.3f} [{mlo:+.3f},{mhi:+.3f}]")
    print(f"dose-response: Pearson r={r:.2f}, slope={slope:+.3f} P(prior)/log-unit, "
          f"95% CI [{blo:+.3f},{bhi:+.3f}], {out['dose_response']['frac_positive']*100:.0f}% > 0")
    print(f"{'alias':6} {'dose':>5} {'swap':>5} {'nopr':>5} {'DiD':>6}")
    for pair, v in sorted(per_pair.items(), key=lambda kv: -(kv[1]['dose'] or -9)):
        if v["did"] is not None:
            print(f"{v['alias']:6} {v['dose']:5.2f} {v['p_swapped']:5.2f} {v['p_no_prior']:5.2f} {v['did']:+6.2f}")
    if "salience" in out:
        print(f"\nsalience (swapped P(prior)): no_cue={out['salience']['mean_no_cue']:.3f} "
              f"-> with_cue={out['salience']['mean_with_cue']:.3f}")
    print(f"saved -> {os.path.relpath(OUT, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
