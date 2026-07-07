#!/usr/bin/env python
"""Render the ablation summary figure from the m4 summary JSONs.

Reads mech/results/m4_k{1,3,5}/m4_summary.json and produces a two-panel
figure (one panel per model) showing mean proxy_pull vs ablation factor for
each (k, condition) combination with 95% bootstrap CI error bars.  The title
notes that ablating positive-DLA heads does not reduce the swapped pull.

  python scripts/mech_m4_figure.py [--out PATH] [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

K_VALUES = [1, 3, 5]
FACTORS = [0.0, 0.25, 0.5]        # ablation sweep factors
BASELINE_FACTOR = 1.0              # factor=1.0 is the unablated baseline
X_VALS = FACTORS + [BASELINE_FACTOR]   # x-axis order: 0, 0.25, 0.5, 1.0
CONDITIONS = ["swapped", "conventional"]
COND_COLOR = {"swapped": "#d62728", "conventional": "#2ca02c"}
K_LINESTYLE = {1: "-", 3: "--", 5: ":"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "mech", "figures", "m4_ablation.png"),
        help="output path for the figure (default: mech/figures/m4_ablation.png)",
    )
    ap.add_argument(
        "--results-dir",
        default=os.path.join(REPO_ROOT, "mech", "results"),
        help="directory containing m4_k{1,3,5}/ subdirectories (default: mech/results/)",
    )
    return ap.parse_args()


def _load_summaries(results_dir):
    """Return dict k -> model_name -> table dict."""
    data = {}
    for k in K_VALUES:
        path = os.path.join(results_dir, f"m4_k{k}", "m4_summary.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing summary file: {path}")
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        data[k] = {
            model_name: model_data["table"]
            for model_name, model_data in summary.get("models", {}).items()
        }
    return data


def _get_point(table, k, factor, condition):
    """Return (mean, ci_low, ci_high) for the given (k, factor, condition)."""
    if factor == BASELINE_FACTOR:
        key = f"{condition}_baseline"
    else:
        key = f"k{k}_f{factor}_{condition}"
    entry = table.get(key)
    if entry is None:
        return float("nan"), float("nan"), float("nan")
    return float(entry["mean"]), float(entry["ci_low"]), float(entry["ci_high"])


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = _load_summaries(args.results_dir)

    # Collect all model names (union across k values), sorted for stable order.
    all_models = sorted({m for k_data in data.values() for m in k_data})

    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), squeeze=False)

    for mi, model_name in enumerate(all_models):
        ax = axes[0][mi]
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")

        for k in K_VALUES:
            table = data[k].get(model_name)
            if table is None:
                continue
            for condition in CONDITIONS:
                means, ci_lows, ci_highs = [], [], []
                for factor in X_VALS:
                    m, lo, hi = _get_point(table, k, factor, condition)
                    means.append(m)
                    ci_lows.append(lo)
                    ci_highs.append(hi)

                means_arr = np.array(means, dtype=float)
                lo_arr = np.array(ci_lows, dtype=float)
                hi_arr = np.array(ci_highs, dtype=float)
                err_lo = means_arr - lo_arr
                err_hi = hi_arr - means_arr

                ax.errorbar(
                    X_VALS,
                    means_arr,
                    yerr=[err_lo, err_hi],
                    label=f"k={k} {condition}",
                    color=COND_COLOR[condition],
                    linestyle=K_LINESTYLE[k],
                    marker="o",
                    markersize=4,
                    capsize=3,
                    lw=1.2,
                )

        short = model_name.split("/")[-1]
        ax.set_xlabel("ablation factor (1.0 = baseline, no ablation)")
        ax.set_ylabel("mean proxy_pull (nats)")
        ax.set_xticks(X_VALS)
        ax.set_xticklabels(["0", "0.25", "0.5", "1.0\n(baseline)"])
        ax.set_title(short)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Ablating positive-DLA heads does not reduce the swapped pull",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[m4_figure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
