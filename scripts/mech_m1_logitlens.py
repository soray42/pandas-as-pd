#!/usr/bin/env python
"""M1: logit-lens trajectory of proxy_pull through the residual stream.

Pilot mode (--pilot): 0.5B model, depth 0 only. Gate printed + exit code:
  conventional mean final proxy_pull > 0 AND no_prior mean final < 0.
Full mode: both models x both depths.
Outputs per model:
  mech/results/m1_records.jsonl          one row per stimulus with full trajectory
  mech/results/m1_summary.json           per-condition mean trajectory + bootstrap CI
  mech/figures/m1_logitlens_{short}.png  3-condition overlay, depth panels

Crossover diagnostic per swapped item:
  - whether bound side (proxy_pull < 0) ever leads after block 4
  - argmax layer (highest swapped proxy_pull)
  - last layer where bound side leads

  python scripts/mech_m1_logitlens.py [--pilot] [--models M] [--depths D] [--items N] [--seed S] [--out DIR]
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
    ap.add_argument("--pilot", action="store_true",
                    help="quick pilot: 0.5B, depth 0 only, print gate and exit")
    ap.add_argument("--models", default=None,
                    help="comma-separated HF model names (default: MECH_MODELS)")
    ap.add_argument("--depths", default="0,512",
                    help="comma-separated filler token depths (default: 0,512)")
    ap.add_argument("--items", type=int, default=None,
                    help="cap stimuli per condition per depth")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mech", "results"),
                    help="output directory (default: mech/results/)")
    ap.add_argument("--figs", default=os.path.join(REPO_ROOT, "mech", "figures"),
                    help="figure directory (default: mech/figures/)")
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


def _trajectory_ci(traj_list, rng, n=4000):
    """Per-layer bootstrap CI over a list of trajectories (each a 1-D numpy array)."""
    import numpy as np
    if not traj_list:
        return [], [], []
    arr = np.stack(traj_list)  # [n_items, n_layers+1]
    means = arr.mean(axis=0).tolist()
    lows, highs = [], []
    for li in range(arr.shape[1]):
        col = arr[:, li]
        boot_means = [col[rng.integers(0, len(col), len(col))].mean() for _ in range(n)]
        lows.append(float(np.percentile(boot_means, 2.5)))
        highs.append(float(np.percentile(boot_means, 97.5)))
    return means, lows, highs


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    import numpy as np
    import torch

    from alias_inertia.determinism import environment_fingerprint, utc_now_iso
    from alias_inertia.mech.env import MECH_MODELS, load_model
    from alias_inertia.mech.logitlens import run_m1
    from alias_inertia.mech.manifest import update_manifest
    from alias_inertia.mech.proxy import build_proxy_lexicon
    from alias_inertia.mech.stimuli_mech import build_mech_stimuli
    from transformers import AutoTokenizer

    if args.pilot:
        models = [MECH_MODELS[0]]
        depths = [0]
        print("[m1] pilot mode: 0.5B, depth 0")
    else:
        models = [m.strip() for m in args.models.split(",")] if args.models else list(MECH_MODELS)
        depths = [int(d) for d in args.depths.split(",") if d.strip()]

    tokenizer = AutoTokenizer.from_pretrained(models[0])
    stimuli_all = build_mech_stimuli(
        tokenizer, depths=tuple(depths), n_per_cell=30, seed=args.seed,
        pair_names=("numpy__pandas",),
    )
    lex = build_proxy_lexicon(tokenizer)

    all_records = []
    all_summaries = {}
    gate_pass = True
    rng = np.random.default_rng(args.seed)

    for model_name in models:
        short = _model_short(model_name)
        print(f"\n[m1] model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tl = load_model(model_name, device=device)
        model_tl.eval()
        n_layers = model_tl.cfg.n_layers

        model_records = []
        trajs_by_cond_depth: dict[tuple, list] = {}
        n_done = 0

        for depth in depths:
            stims = [s for s in stimuli_all if s.depth == depth]
            if args.items:
                # Cap per condition to keep balanced.
                stims_cap = []
                cnt: dict[str, int] = {}
                for s in stims:
                    if cnt.get(s.condition, 0) < args.items:
                        cnt[s.condition] = cnt.get(s.condition, 0) + 1
                        stims_cap.append(s)
                stims = stims_cap

            records_depth = run_m1(model_tl, stims, lex)

            for rec in records_depth:
                traj = np.array(rec["trajectory"])  # [n_layers+1]
                key = (rec["condition"], depth)
                trajs_by_cond_depth.setdefault(key, []).append(traj)
                model_records.append({
                    **{k: v for k, v in rec.items() if k != "trajectory"},
                    "trajectory": rec["trajectory"],
                    "model": model_name,
                    "depth": depth,
                })
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done} stimuli done")

        # Crossover diagnostic for swapped items.
        crossover_stats = {}
        for depth in depths:
            sw_trajs = trajs_by_cond_depth.get(("swapped", depth), [])
            if not sw_trajs:
                continue
            bound_leads_after4 = []
            last_bound_leads_layer = []
            argmax_layers = []
            for traj in sw_trajs:
                after4 = traj[5:]  # layers > 4
                bound_leads_after4.append(bool((after4 < 0).any()))
                last_bl = int(np.where(traj < 0)[0][-1]) if (traj < 0).any() else -1
                last_bound_leads_layer.append(last_bl)
                argmax_layers.append(int(np.argmax(traj)))
            frac_bound_leads = float(np.mean(bound_leads_after4))
            frac_final_prior = float(np.mean([t[-1] > 0 for t in sw_trajs]))
            crossover_stats[str(depth)] = {
                "n_items": len(sw_trajs),
                "frac_bound_leads_after_block4": frac_bound_leads,
                "frac_final_sign_prior": frac_final_prior,
                "mean_argmax_layer": float(np.mean(argmax_layers)),
                "mean_last_bound_leads_layer": float(np.mean([x for x in last_bound_leads_layer if x >= 0])
                                                     if any(x >= 0 for x in last_bound_leads_layer) else float("nan")),
            }
            print(f"  [swapped depth={depth}] bound_leads_after4={frac_bound_leads:.3f} "
                  f"final_sign_prior={frac_final_prior:.3f}")

        # Gate check (pilot mode).
        if args.pilot:
            final_conv = [r["trajectory"][-1] for r in model_records if r["condition"] == "conventional"]
            final_nop = [r["trajectory"][-1] for r in model_records if r["condition"] == "no_prior"]
            conv_mean = float(np.mean(final_conv)) if final_conv else float("nan")
            nop_mean = float(np.mean(final_nop)) if final_nop else float("nan")
            gate = (conv_mean > 0) and (nop_mean < 0)
            gate_pass = gate_pass and gate
            print(f"  PILOT GATE: conventional_final={conv_mean:.3f} no_prior_final={nop_mean:.3f} "
                  f"-> {'PASS' if gate else 'FAIL'}")

        # Per-condition mean trajectory + CI.
        per_cond_depth = {}
        for (cond, depth), trajs in trajs_by_cond_depth.items():
            means, lows, highs = _trajectory_ci(trajs, rng)
            per_cond_depth[f"{cond}__d{depth}"] = {
                "condition": cond, "depth": depth,
                "n_items": len(trajs),
                "mean": means, "ci_low": lows, "ci_high": highs,
            }

        # Figure per model.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = {"conventional": "#2ca02c", "swapped": "#d62728", "no_prior": "#1f77b4"}
        n_depths = len(depths)
        fig, axes = plt.subplots(1, n_depths, figsize=(5 * n_depths, 4), squeeze=False)
        for di, depth in enumerate(depths):
            ax = axes[0][di]
            for cond in ("conventional", "swapped", "no_prior"):
                key = f"{cond}__d{depth}"
                if key not in per_cond_depth:
                    continue
                cd = per_cond_depth[key]
                # Skip checkpoint 0 (embedding-only stream): its lexical unembed
                # artifact (~+40 nats) squashes the informative range.
                xs = list(range(1, len(cd["mean"])))
                ax.plot(xs, cd["mean"][1:], label=cond, color=cmap.get(cond, "gray"))
                ax.fill_between(xs, cd["ci_low"][1:], cd["ci_high"][1:],
                                alpha=0.2, color=cmap.get(cond, "gray"))
            ax.axhline(0, color="gray", lw=0.6, linestyle="--")
            ax.set_xlabel("blocks applied")
            ax.set_ylabel("proxy_pull (nats)")
            ax.set_title(f"depth={depth}")
            ax.legend(fontsize=7)
        fig.suptitle(f"M1 logit-lens | {short}", fontsize=10)
        fig.tight_layout()
        fig_path = os.path.join(args.figs, f"m1_logitlens_{short}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  -> {os.path.relpath(fig_path, REPO_ROOT)}")

        all_records.extend(model_records)
        all_summaries[model_name] = {
            "n_layers": n_layers,
            "per_condition_depth": per_cond_depth,
            "crossover_diagnostic": crossover_stats,
        }

        del model_tl
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Write outputs (merge with prior per-model runs so split processes don't clobber) ----
    rec_path = os.path.join(args.out, "m1_records.jsonl")
    if os.path.exists(rec_path) and not args.pilot:
        kept = []
        with open(rec_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("model") not in models:
                    kept.append(row)
        all_records = kept + all_records
    with open(rec_path, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sum_path = os.path.join(args.out, "m1_summary.json")
    if os.path.exists(sum_path) and not args.pilot:
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
        "config": {"models": models, "depths": depths, "items_cap": args.items, "seed": args.seed},
        "n_records": len(all_records),
        "pilot_gate_pass": gate_pass if args.pilot else None,
        "models": all_summaries,
        "environment": environment_fingerprint(),
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n[m1] wrote {len(all_records)} records -> {os.path.relpath(rec_path, REPO_ROOT)}")
    print(f"[m1] wrote summary -> {os.path.relpath(sum_path, REPO_ROOT)}")

    update_manifest("m1_logitlens", {
        "pilot": args.pilot,
        "models": models,
        "n_records": len(all_records),
        "pilot_gate_pass": gate_pass if args.pilot else None,
    })

    if args.pilot and not gate_pass:
        print("[m1] PILOT GATE FAILED.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
