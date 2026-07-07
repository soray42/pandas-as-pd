#!/usr/bin/env python
"""M2: residual-stream activation patching across layers and token groups.

Runs run_m2 for each (model x depth) shard. Checkpointing: skips a shard if its
records file already exists (unless --force). Produces per-shard JSONL records,
combined JSON summary, and heatmap figures.

Directions:
  noprior_to_swapped: patches no_prior SOURCE activations into the swapped DESTINATION run
    (does removing the prior-carrying import representation erase the prior pull?).
  swapped_to_noprior: patches swapped SOURCE activations into the no_prior DESTINATION run
    (is the swapped import representation sufficient to induce the pull in the no_prior run?).

fraction_restored = (pull_dst - pull_patched) / (pull_dst - pull_src)
  * 1.0: patching fully transplants the source pull (group is sufficient)
  * 0.0: patching has no effect (group is not necessary)
  * negative or >1: nonlinear interaction

Each record from run_m2 covers one (base, direction, layer, group). The runner
aggregates to (layer, group) means for the heatmap.

--stride N: after collecting records, retain only layers with layer % N == 0.
  run_m2 still evaluates all layers; stride is a post-filter for display only.

  python scripts/mech_m2_patching.py [--models M] [--depths D] [--items N] [--stride L] [--force] [--seed S] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_ALL_DIRECTIONS = ("noprior_to_swapped", "swapped_to_noprior")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=None,
                    help="comma-separated HF model names (default: MECH_MODELS)")
    ap.add_argument("--depths", default="0,512",
                    help="comma-separated filler token depths (default: 0,512)")
    ap.add_argument("--items", type=int, default=None,
                    help="cap bases per depth")
    ap.add_argument("--stride", type=int, default=1,
                    help="layer display stride: retain every Nth layer in heatmap (default: 1)")
    ap.add_argument("--directions", default=",".join(_ALL_DIRECTIONS),
                    help=f"comma-separated directions (default: {','.join(_ALL_DIRECTIONS)})")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if shard output already exists")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mech", "results"),
                    help="output directory (default: mech/results/)")
    ap.add_argument("--figs", default=os.path.join(REPO_ROOT, "mech", "figures"),
                    help="figure directory (default: mech/figures/)")
    return ap.parse_args()


def _model_short(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _shard_path(out: str, short: str, depth: int, suffix: str) -> str:
    return os.path.join(out, f"m2_{short}_d{depth}{suffix}")


def _heatmap(data, row_labels, col_labels, title, fig_path):
    """layer x group heatmap of mean fraction_restored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 1.2), max(3, len(row_labels) * 0.4)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("group")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if val == val:  # not nan
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _compute_ci(vals, rng, n=4000):
    import numpy as np
    v = np.array(vals, dtype=float)
    m = float(v.mean())
    if len(v) > 1:
        boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
    else:
        lo = hi = float("nan")
    return m, lo, hi


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    import numpy as np
    import torch

    from alias_inertia.determinism import environment_fingerprint, utc_now_iso
    from alias_inertia.mech.env import MECH_MODELS, load_model
    from alias_inertia.mech.manifest import update_manifest
    from alias_inertia.mech.patching import GROUPS, run_m2
    from alias_inertia.mech.proxy import build_proxy_lexicon
    from alias_inertia.mech.stimuli_mech import build_mech_stimuli
    from transformers import AutoTokenizer

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MECH_MODELS)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(models[0])
    stimuli_all = build_mech_stimuli(
        tokenizer, depths=tuple(depths), n_per_cell=30, seed=args.seed,
        pair_names=("numpy__pandas",),
    )
    lex = build_proxy_lexicon(tokenizer)

    all_summary: dict = {}

    for model_name in models:
        short = _model_short(model_name)
        print(f"\n[m2] model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tl = load_model(model_name, device=device)
        model_tl.eval()
        n_layers = model_tl.cfg.n_layers
        all_summary[model_name] = {"n_layers": n_layers, "shards": {}}

        for depth in depths:
            shard_key = f"d{depth}"
            rec_path = _shard_path(args.out, short, depth, "_records.jsonl")
            if os.path.exists(rec_path) and not args.force:
                print(f"  [m2] shard {short}/d{depth} exists; loading cached records")
                records = []
                with open(rec_path, encoding="utf-8") as fh:
                    for line in fh:
                        records.append(json.loads(line))
                print(f"  [m2] loaded {len(records)} cached records")
            else:
                stims = [s for s in stimuli_all if s.depth == depth]
                if args.items:
                    # Keep aligned bases: first N unique base_ids.
                    seen_bases: set[str] = set()
                    cap: list = []
                    for s in stims:
                        seen_bases.add(s.base_id)
                        if len(seen_bases) <= args.items:
                            cap.append(s)
                    stims = cap

                # run_m2 evaluates all layers; stride is applied to display below.
                records = run_m2(model_tl, stims, lex, directions=tuple(directions))
                # Annotate with model + depth.
                for r in records:
                    r["model"] = model_name
                    r["depth"] = depth

                with open(rec_path, "w", encoding="utf-8") as fh:
                    for r in records:
                        fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
                print(f"  [m2] {shard_key}: {len(records)} records -> {os.path.relpath(rec_path, REPO_ROOT)}")

            # Heatmap figures per direction (always regenerated from records).
            stride = args.stride
            for direction in directions:
                dir_recs = [r for r in records if r.get("direction") == direction]
                if not dir_recs:
                    continue
                # layers_seen after stride.
                all_layers = sorted({r["layer"] for r in dir_recs})
                layers_seen = [li for li in all_layers if li % stride == 0]
                # Collect fraction_restored per (layer, group).
                by_lg: dict[tuple, list] = {}
                for r in dir_recs:
                    if r["layer"] % stride != 0:
                        continue
                    g = r.get("group")
                    if g is None:
                        continue
                    frac = r.get("fraction_restored")
                    if frac is not None and frac == frac:  # not nan
                        by_lg.setdefault((r["layer"], g), []).append(float(frac))
                data = [[np.mean(by_lg.get((li, g), [float("nan")])) for g in GROUPS]
                        for li in layers_seen]
                row_labels = [f"L{li}" for li in layers_seen]
                title = f"M2 patching {direction}\n{short} depth={depth}"
                fig_path = os.path.join(args.figs, f"m2_patch_{direction}_{short}_d{depth}.png")
                _heatmap(data, row_labels, list(GROUPS), title, fig_path)
                print(f"  -> {os.path.relpath(fig_path, REPO_ROOT)}")

                # Companion CI JSON.
                ci_out: dict = {}
                for li in layers_seen:
                    ci_out[str(li)] = {}
                    for g in GROUPS:
                        raw_vals = by_lg.get((li, g), [])
                        if raw_vals:
                            m, lo, hi = _compute_ci(raw_vals, rng)
                            ci_out[str(li)][g] = {"mean": m, "ci_low": lo, "ci_high": hi,
                                                  "n": len(raw_vals)}
                ci_path = os.path.join(args.out, f"m2_ci_{direction}_{short}_d{depth}.json")
                with open(ci_path, "w", encoding="utf-8") as fh:
                    json.dump(ci_out, fh, ensure_ascii=False, indent=2)

            all_summary[model_name]["shards"][shard_key] = {
                "n_records": len(records), "stride_display": stride}
            print(f"  [m2] shard done: {len(records)} records")

        del model_tl
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Merge with prior per-model runs so split processes don't clobber ----
    sum_path = os.path.join(args.out, "m2_summary.json")
    if os.path.exists(sum_path):
        try:
            with open(sum_path, encoding="utf-8") as fh:
                prior = json.load(fh)
            for k, v in prior.get("models", {}).items():
                all_summary.setdefault(k, v)
            # Config must describe ALL data in the file, not the last invocation.
            models = sorted(set(prior.get("config", {}).get("models", [])) | set(models))
        except (json.JSONDecodeError, KeyError):
            pass

    summary = {
        "timestamp_utc": utc_now_iso(),
        "config": {
            "models": models, "depths": depths, "directions": directions,
            "items_cap": args.items, "stride": args.stride, "seed": args.seed,
        },
        "models": all_summary,
        "environment": environment_fingerprint(),
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n[m2] summary -> {os.path.relpath(sum_path, REPO_ROOT)}")

    update_manifest("m2_patching", {
        "models": models, "depths": depths, "directions": directions,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
