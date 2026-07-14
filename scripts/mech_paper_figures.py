#!/usr/bin/env python
"""Render the paper-facing mechanistic figures from the released mech records.

Regenerates the five figures referenced by paper/acl_latex.tex from
mech/results/ (no model loading): the logit-lens trajectory panels for both
models, the depth-0 patching heatmaps for both models, and the 1.5B
proxy-validation scatter. Styling matches the behavioral paper figures; the
figures carry no titles (captions in the paper describe them).

  python scripts/mech_paper_figures.py [--out figures]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = {"05b": "Qwen/Qwen2.5-0.5B", "15b": "Qwen/Qwen2.5-1.5B"}
CMAP = {"conventional": "#2ca02c", "swapped": "#d62728", "no_prior": "#1f77b4"}
GROUPS = ["import_span", "use_alias", "final_pos", "filler_span"]


def lens_figure(records, model, out_path, rng):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for di, depth in enumerate((0, 512)):
        ax = axes[0][di]
        for cond in ("conventional", "swapped", "no_prior"):
            ts = np.array([r["trajectory"] for r in records
                           if r["model"] == model and r["depth"] == depth and r["condition"] == cond])
            if not len(ts):
                continue
            mean = ts.mean(0)
            boots = np.array([ts[rng.integers(0, len(ts), len(ts))].mean(0) for _ in range(2000)])
            lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
            xs = np.arange(1, len(mean))
            ax.plot(xs, mean[1:], label=cond.replace("_", "-"), color=CMAP[cond])
            ax.fill_between(xs, lo[1:], hi[1:], alpha=0.2, color=CMAP[cond])
        ax.axhline(0, color="gray", lw=0.6, linestyle="--")
        ax.set_xlabel("blocks applied")
        ax.set_ylabel("proxy-pull (nats)")
        ax.set_title(f"depth = {depth}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def patch_figure(shard_path, n_layers, out_path):
    rows = [json.loads(line) for line in open(shard_path, encoding="utf-8")]
    rows = [r for r in rows if r["direction"] == "noprior_to_swapped"]
    grid = np.full((n_layers, len(GROUPS)), np.nan)
    for gi, group in enumerate(GROUPS):
        for layer in range(n_layers):
            vals = [r["fraction_restored"] for r in rows
                    if r["group"] == group and r["layer"] == layer
                    and r["fraction_restored"] == r["fraction_restored"]]
            if vals:
                grid[layer, gi] = float(np.mean(vals))
    fig_h = 0.32 * n_layers + 1.2
    fig, ax = plt.subplots(figsize=(4.6, fig_h))
    im = ax.imshow(np.nan_to_num(grid), cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    for layer in range(n_layers):
        for gi in range(len(GROUPS)):
            v = grid[layer, gi]
            if v == v:
                ax.text(gi, layer, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(["import line", "use-site alias", "final position", "filler"],
                       rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(0, n_layers, 2))
    ax.set_yticklabels([f"L{i}" for i in range(0, n_layers, 2)], fontsize=7)
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, shrink=0.6, label="fraction of gap restored")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def m0_figure(records, model, out_path):
    rows = [r for r in records if r["model"] == model]
    fig, ax = plt.subplots(figsize=(5, 4))
    for cond in ("conventional", "swapped", "no_prior"):
        xs = [r["full_prior_pull"] for r in rows if r["condition"] == cond]
        ys = [r["proxy_pull"] for r in rows if r["condition"] == cond]
        ax.scatter(xs, ys, s=12, alpha=0.5, color=CMAP[cond], label=cond.replace("_", "-"))
    allv = [r["full_prior_pull"] for r in rows] + [r["proxy_pull"] for r in rows]
    lim = max(abs(v) for v in allv) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("full prior-pull (nats)")
    ax.set_ylabel("proxy-pull at the final layer (nats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "figures"))
    ap.add_argument("--mech-results", default=os.path.join(REPO_ROOT, "mech", "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20260618)

    m1 = [json.loads(line) for line in open(os.path.join(args.mech_results, "m1_records.jsonl"), encoding="utf-8")]
    for short, model in MODELS.items():
        lens_figure(m1, model, os.path.join(args.out, f"mech_logitlens_{short}.png"), rng)
        print(f"wrote mech_logitlens_{short}.png")

    n_layers = {"05b": 24, "15b": 28}
    for short in MODELS:
        pattern = os.path.join(args.mech_results, f"m2_*{short.replace('b', 'B').replace('05', '0_5').replace('15', '1_5')}*_d0_records.jsonl")
        shards = glob.glob(pattern)
        if not shards:
            print(f"missing m2 shard for {short} ({pattern})")
            return 1
        patch_figure(shards[0], n_layers[short], os.path.join(args.out, f"mech_patch_{short}_d0.png"))
        print(f"wrote mech_patch_{short}_d0.png")

    m0 = [json.loads(line) for line in open(os.path.join(args.mech_results, "m0_records.jsonl"), encoding="utf-8")]
    m0_figure(m0, MODELS["15b"], os.path.join(args.out, "mech_proxyval_15b.png"))
    print("wrote mech_proxyval_15b.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
