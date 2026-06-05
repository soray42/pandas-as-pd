#!/usr/bin/env python
"""Per-condition RAW prior-pull (absolute), to show the no-prior control sits at ~0 in log-prob
and the DiD gap is clean. prior-pull(row) = logsumexp(L_prior_lib) - logsumexp(L_other_lib),
i.e. preference for the pair's conventional library over the other library, at the use site.

  conventional : alias bound to its conventional library  -> large positive (ceiling)
  swapped      : conventional library is the prior, other is bound -> positive (the effect)
  no_prior     : non-canonical alias bound to the other library -> ~0 (no competing prior)

DiD gap = mean(swapped) - mean(no_prior). 95% CIs are pair-clustered bootstrap.

  python scripts/raw_prior_pull.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDS = ["conventional", "swapped", "no_prior"]


def clustered_mean_ci(d, n_boot, rng):
    pairs = d["pair"].unique()
    by = {p: d[d["pair"] == p]["pmo"].values for p in pairs}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(pairs, size=len(pairs), replace=True)
        vals = np.concatenate([by[p] for p in chosen])
        boots.append(vals.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d["pmo"].mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    df = pd.read_parquet(os.path.join(REPO_ROOT, cfg["output"]["results"]))
    df["pmo"] = [float(r[f"logsumexp_{r['prior_lib']}"]) - float(r[f"logsumexp_{r['other_lib']}"])
                 for _, r in df.iterrows()]
    rng = np.random.default_rng(int(cfg["analysis"]["bootstrap_seed"]))

    out = {}
    print("condition      n     mean_prior_pull [95% pair-clustered CI]")
    for c in CONDS:
        d = df[df["condition"] == c]
        m, lo, hi = clustered_mean_ci(d, args.boot, rng)
        out[c] = {"n": int(len(d)), "mean": m, "ci_lo": lo, "ci_hi": hi}
        print(f"  {c:13} {len(d):4d}   {m:+7.3f}  [{lo:+.3f}, {hi:+.3f}]")
    out["did_gap_swapped_minus_no_prior"] = out["swapped"]["mean"] - out["no_prior"]["mean"]
    print(f"\n  DiD gap (swapped - no_prior) = {out['did_gap_swapped_minus_no_prior']:+.3f}")
    with open(os.path.join(REPO_ROOT, "results", "raw_prior_pull.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("saved -> results/raw_prior_pull.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
