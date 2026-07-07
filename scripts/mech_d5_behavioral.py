#!/usr/bin/env python
"""D5 behavioral arms: restatement + instruction conditions vs swapped-plain baseline.

Four conditions built by post-processing the standard swapped prompt (stimuli.py is never
modified):
  swapped_plain       : standard swapped prompt (import other_lib as treatment_alias)
  swapped_restatement : swapped_plain with '# note: {alias} is {bound_lib} in this file'
                        inserted immediately BEFORE the use line
  no_prior            : nonce-alias arm (import other_lib as {zz,qx,vv}, round-robin per rep)
  swapped_instruction : instruct models ONLY; two comment lines prepended ABOVE the import:
                        '# INSTRUCTION: ...' / '# Do not assume conventional alias meanings.'

Pairs: numpy<->pandas (very_common tier) + first rare-tier pair in SWAP_PAIRS containing
xgboost. Two pairs; no pair-clustered pooling (requires >= 3 clusters; reported per-pair).
Models: Qwen2.5-{0.5B, 0.5B-Instruct, 1.5B, 1.5B-Instruct} via HF fp16 GPU, one at a time.
Metric: full prior_pull via compute_metric_row (teacher-forced continuation scoring).
Depths: 0 and 512 filler tokens; >= 24 items per cell at depth 512 with 2 templates x 12 reps.

Outputs (mech/ dirs are created on first run):
  mech/results/d5_records.jsonl       one row per scored stimulus
  mech/results/d5_summary.json        per (pair, model, depth, condition) stats + config echo
  mech/results/example_prompts.json   3 example prompts per condition for the paper appendix
  mech/figures/d5_arms.png            bar chart per model (conditions x pairs)

  python scripts/mech_d5_behavioral.py [--dry-run] [--models ...] [--limit N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.determinism import (  # noqa: E402
    environment_fingerprint,
    sha256_text,
    stable_hash,
    utc_now_iso,
)
from alias_inertia.lexicons import LEXICONS, SWAP_PAIRS  # noqa: E402
from alias_inertia.metrics import compute_metric_row  # noqa: E402
from alias_inertia.stimuli import (  # noqa: E402
    NONCE_ALIASES,
    build_stimulus,
    reps_for_depth,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default model suite: base + instruct at both sizes.
_DEFAULT_MODELS: list[tuple[str, bool]] = [
    ("Qwen/Qwen2.5-0.5B",           False),
    ("Qwen/Qwen2.5-0.5B-Instruct",  True),
    ("Qwen/Qwen2.5-1.5B",           False),
    ("Qwen/Qwen2.5-1.5B-Instruct",  True),
]

# Two instruction comment lines prepended ABOVE the import for the swapped_instruction arm.
_INSTR_COMMENT_1 = (
    "# INSTRUCTION: resolve every alias strictly according to the import statements in this file."
)
_INSTR_COMMENT_2 = "# Do not assume conventional alias meanings."

D5_SEED = 20260618


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def _np_pd_pair():
    for p in SWAP_PAIRS:
        if p.name == "numpy__pandas":
            return p
    raise RuntimeError("numpy__pandas not found in SWAP_PAIRS")


def _xgb_rare_pair():
    """First rare-tier pair in SWAP_PAIRS where either library is xgboost."""
    for p in SWAP_PAIRS:
        if p.tier == "rare" and "xgboost" in (p.prior_lib, p.other_lib):
            return p
    raise RuntimeError("no rare-tier xgboost pair found in SWAP_PAIRS")


# ---------------------------------------------------------------------------
# Prompt post-processors (never touch stimuli.py)
# ---------------------------------------------------------------------------

def _restatement_prompt(prompt: str, alias: str, bound_lib: str) -> str:
    """Insert '# note: ...' immediately before the use line (the final '{alias}.' line)."""
    marker = f"\n{alias}."
    idx = prompt.rfind(marker)
    if idx < 0:
        raise ValueError(f"use-line marker {marker!r} not found in prompt")
    note = f"\n# note: {alias} is {bound_lib} in this file"
    return prompt[:idx] + note + prompt[idx:]


def _instruction_prompt(prompt: str) -> str:
    """Prepend the two instruction comment lines immediately before the import line."""
    idx = prompt.index("import ")
    prefix = _INSTR_COMMENT_1 + "\n" + _INSTR_COMMENT_2 + "\n"
    return prompt[:idx] + prefix + prompt[idx:]


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------

def _boot_ci(vals: list, rng: np.random.Generator, n: int = 4000):
    """Bootstrap 95% CI; returns (mean, lo, hi) or (None, None, None)."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None, None, None
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _boot_delta_ci(a_vals: list, b_vals: list, rng: np.random.Generator, n: int = 4000):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    a = np.asarray([x for x in a_vals if x is not None], float)
    b = np.asarray([x for x in b_vals if x is not None], float)
    if len(a) == 0 or len(b) == 0:
        return None, None, None
    boots = [
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(n)
    ]
    return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# GPU cleanup
# ---------------------------------------------------------------------------

def _free_gpu() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    is_instruct: bool,
    pairs,
    depths: list[int],
    reps: int,
    templates: list[str],
    seed: int,
    limit: int | None,
) -> tuple[list[dict], dict]:
    """Score all items for one model; returns (records, example_prompts_for_this_model)."""
    be = build_backend("hf", {
        "model": model_name,
        "device": "cuda",
        "dtype": "float16",
        "add_special_tokens": False,
    })
    conditions_base = ["swapped_plain", "swapped_restatement", "no_prior"]
    conditions_all = conditions_base + (["swapped_instruction"] if is_instruct else [])

    records: list[dict] = []
    # example_prompts: condition -> list of up to 3 dicts with the prompt text (paper appendix)
    example_prompts: dict[str, list] = {}

    for pair in pairs:
        lex = {
            pair.prior_lib: LEXICONS[pair.prior_lib],
            pair.other_lib: LEXICONS[pair.other_lib],
        }
        treatment_alias = pair.treatment_alias
        bound_lib = pair.other_lib
        n_done = 0

        for depth in depths:
            n_reps_at_depth = reps_for_depth(depth, reps)
            for tmpl in templates:
                for rep in range(n_reps_at_depth):
                    if limit is not None and n_done >= limit:
                        break
                    # Nonce alias: round-robin by rep index across NONCE_ALIASES.
                    nc_alias = NONCE_ALIASES[rep % len(NONCE_ALIASES)]

                    # Build base swapped stimulus (defines the shared filler and use line).
                    stim_plain = build_stimulus(
                        pair=pair,
                        condition="swapped",
                        depth_tokens=depth,
                        template_id=tmpl,
                        rep=rep,
                        count_tokens=be.count_tokens,
                        seed=seed,
                    )
                    # No-prior stimulus uses nonce alias; filler differs from swapped_plain by design
                    # (build_stimulus keys its RNG on alias, so fillers are condition-specific).
                    stim_nop = build_stimulus(
                        pair=pair,
                        condition="no_prior",
                        depth_tokens=depth,
                        template_id=tmpl,
                        rep=rep,
                        count_tokens=be.count_tokens,
                        seed=seed,
                        non_canonical_alias=nc_alias,
                    )

                    # base_id groups items by (pair, depth, template, rep) across conditions.
                    base_id = stable_hash(
                        ["d5", pair.name, depth, tmpl, rep], length=16
                    )

                    # Map condition name -> prompt string.
                    plain_prompt = stim_plain.prompt
                    cond_prompts: dict[str, str] = {
                        "swapped_plain": plain_prompt,
                        "swapped_restatement": _restatement_prompt(
                            plain_prompt, treatment_alias, bound_lib
                        ),
                        "no_prior": stim_nop.prompt,
                    }
                    if is_instruct:
                        cond_prompts["swapped_instruction"] = _instruction_prompt(plain_prompt)

                    # Collect up to 3 example prompts per condition (paper appendix).
                    for cond, p in cond_prompts.items():
                        bucket = example_prompts.setdefault(cond, [])
                        if len(bucket) < 3:
                            bucket.append({
                                "pair": pair.name,
                                "model": model_name,
                                "depth": depth,
                                "template_id": tmpl,
                                "rep": rep,
                                "prompt": p,
                            })

                    for cond in conditions_all:
                        if cond not in cond_prompts:
                            continue
                        prompt = cond_prompts[cond]
                        item_alias = nc_alias if cond == "no_prior" else treatment_alias
                        stim_ref = stim_nop if cond == "no_prior" else stim_plain
                        stimulus_id = stable_hash(
                            [
                                "d5", pair.name, cond, item_alias,
                                depth, tmpl, rep, sha256_text(prompt),
                            ],
                            length=16,
                        )
                        try:
                            m = compute_metric_row(
                                be, prompt,
                                prior_lib=pair.prior_lib,
                                bound_lib=pair.other_lib,
                                lexicons=lex,
                            )
                        except Exception as exc:
                            print(
                                f"  skip {pair.name}/{cond}/d={depth}/t={tmpl}/r={rep}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue
                        records.append({
                            "model": model_name,
                            "pair_name": pair.name,
                            "prior_lib": pair.prior_lib,
                            "bound_lib": bound_lib,
                            "tier": pair.tier,
                            "condition": cond,
                            "alias": item_alias,
                            "nonce": nc_alias,
                            "depth": depth,
                            "depth_tokens_actual": stim_ref.meta.get("depth_tokens_actual", depth),
                            "template_id": tmpl,
                            "rep": rep,
                            "stimulus_id": stimulus_id,
                            "base_id": base_id,
                            "prompt_sha256": sha256_text(prompt),
                            "prior_pull": m["prior_pull"],
                            "bound_mass": m["bound_mass"],
                            "boundary_merge_any": m["boundary_merge_any"],
                        })
                        n_done += 1

                    if n_done % 20 == 0 and n_done > 0:
                        print(f"  {pair.name}: {n_done} items scored")

                if limit is not None and n_done >= limit:
                    break
            if limit is not None and n_done >= limit:
                break

        print(f"  {pair.name}: {n_done} total items scored")

    try:
        be.close()
    except Exception:
        pass
    del be
    _free_gpu()

    return records, example_prompts


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

_ALL_CONDITIONS = [
    "swapped_plain",
    "swapped_restatement",
    "swapped_instruction",
    "no_prior",
]


def compute_summary(
    records: list[dict],
    pairs,
    models_cfg: list[tuple[str, bool]],
    depths: list[int],
    boot_seed: int,
) -> dict:
    rng = np.random.default_rng(boot_seed)
    per_pair: dict = {}

    for pair in pairs:
        pr = [r for r in records if r["pair_name"] == pair.name]
        per_model: dict = {}
        for model_name, _is_instr in models_cfg:
            mr = [r for r in pr if r["model"] == model_name]
            per_depth: dict = {}
            for depth in depths:
                dr = [r for r in mr if r["depth"] == depth]
                plain_vals = [r["prior_pull"] for r in dr if r["condition"] == "swapped_plain"]
                per_cond: dict = {}
                for cond in _ALL_CONDITIONS:
                    cvals = [r["prior_pull"] for r in dr if r["condition"] == cond]
                    if not cvals:
                        continue
                    mean, lo, hi = _boot_ci(cvals, rng)
                    delta, dlo, dhi = _boot_delta_ci(cvals, plain_vals, rng)
                    per_cond[cond] = {
                        "n": len(cvals),
                        "mean_prior_pull": mean,
                        "ci_95": [lo, hi],
                        "delta_vs_swapped_plain": delta,
                        "delta_ci_95": [dlo, dhi],
                    }
                per_depth[depth] = per_cond
            per_model[model_name] = per_depth
        per_pair[pair.name] = {"tier": pair.tier, "per_model": per_model}

    return {
        "per_pair": per_pair,
        "note_no_cluster_bootstrap": (
            "2 clusters do not support cluster bootstrap; "
            "results are reported per-pair with item-level bootstrap only."
        ),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_COND_COLOR = {
    "swapped_plain":        "#1f77b4",
    "swapped_restatement":  "#ff7f0e",
    "swapped_instruction":  "#2ca02c",
    "no_prior":             "#7f7f7f",
}
_COND_LABEL = {
    "swapped_plain":        "plain",
    "swapped_restatement":  "restatement",
    "swapped_instruction":  "instruction",
    "no_prior":             "no-prior",
}


def make_figure(
    records: list[dict],
    pairs,
    models_cfg: list[tuple[str, bool]],
    depths: list[int],
    out_path: str,
) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    depth_fig = max(depths)  # figure uses the deepest bin
    n_models = len(models_cfg)
    ncols = min(n_models, 2)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    # Flatten to 1-D list; pad with None if grid is uneven.
    axes_flat: list = []
    if n_models == 1:
        axes_flat = [axes_grid]
    elif nrows == 1:
        axes_flat = list(axes_grid)
    else:
        for row in axes_grid:
            axes_flat.extend(row)

    x_base = np.arange(len(pairs))
    pair_labels = [p.name.replace("__", " vs\n") for p in pairs]

    for idx, (model_name, is_instruct) in enumerate(models_cfg):
        ax = axes_flat[idx]
        conds = (
            ["swapped_plain", "swapped_restatement", "swapped_instruction", "no_prior"]
            if is_instruct
            else ["swapped_plain", "swapped_restatement", "no_prior"]
        )
        n_conds = len(conds)
        group_w = 0.8
        bar_w = group_w / n_conds

        mr = [r for r in records if r["model"] == model_name and r["depth"] == depth_fig]
        for ci, cond in enumerate(conds):
            means, errs = [], []
            for pair in pairs:
                vals = [r["prior_pull"] for r in mr
                        if r["pair_name"] == pair.name and r["condition"] == cond]
                means.append(float(np.mean(vals)) if vals else 0.0)
                errs.append(
                    float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
            xs = x_base + (ci - (n_conds - 1) / 2.0) * bar_w
            ax.bar(
                xs, means, bar_w * 0.9,
                yerr=errs,
                label=_COND_LABEL[cond],
                color=_COND_COLOR[cond],
                capsize=3,
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(x_base)
        ax.set_xticklabels(pair_labels, fontsize=8)
        short = model_name.replace("Qwen/Qwen2.5-", "Q2.5-")
        ax.set_title(short, fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("prior_pull (nats)")
        ax.legend(fontsize=7, loc="upper right")

    # Hide any unused axes.
    for idx in range(len(models_cfg), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"D5 behavioral arms  (depth={depth_fig} tokens)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[d5] figure -> {os.path.relpath(out_path, REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "D5 behavioral arms: restatement + instruction conditions "
            "vs swapped-plain baseline on Qwen2.5 HF models."
        )
    )
    ap.add_argument(
        "--models", default="",
        help=(
            "comma-separated HF model names "
            "(default: all 4 Qwen2.5-{0.5B,0.5B-Instruct,1.5B,1.5B-Instruct})"
        ),
    )
    ap.add_argument(
        "--depths", default="0,512",
        help="comma-separated filler depths in tokens (default: 0,512)",
    )
    ap.add_argument(
        "--reps", type=int, default=12,
        help="reps per (depth>0, template); 12 gives 24 items at depth 512 with 2 templates",
    )
    ap.add_argument("--templates", default="t1,t2")
    ap.add_argument("--seed", type=int, default=D5_SEED)
    ap.add_argument("--boot-seed", type=int, default=20260619)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "mech", "results", "d5"),
        help="output path prefix (suffixed with _records.jsonl / _summary.json)",
    )
    ap.add_argument(
        "--fig-dir",
        default=os.path.join(REPO_ROOT, "mech", "figures"),
        help="directory for output figures",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap scored items per (model, pair) for a smoke run",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print grid size and exit without scoring",
    )
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    templates = [t for t in args.templates.split(",") if t.strip()]

    # Resolve the two target pairs.
    pairs = [_np_pd_pair(), _xgb_rare_pair()]

    # Resolve the model list.
    if args.models:
        wanted = set(args.models.split(","))
        models_cfg = [(n, b) for n, b in _DEFAULT_MODELS if n in wanted]
        if not models_cfg:
            # Allow bare short names like "0.5B" as a convenience subset.
            models_cfg = [
                (n, b) for n, b in _DEFAULT_MODELS
                if any(w in n for w in wanted)
            ]
    else:
        models_cfg = list(_DEFAULT_MODELS)

    # Dry-run: print grid and exit.
    n_base_items = sum(len(templates) * reps_for_depth(d, args.reps) for d in depths)
    print("[d5] pairs:", [p.name for p in pairs])
    print("[d5] models:", [m for m, _ in models_cfg])
    print("[d5] depths:", depths, " reps:", args.reps, " templates:", templates)
    print(
        f"[d5] ~{n_base_items} base items per (model, pair); "
        f"3 base conditions + 1 instruct-only condition"
    )
    if args.dry_run:
        print("[d5] dry-run: no scoring.")
        return 0

    # Create output directories.
    results_dir = os.path.dirname(args.out)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Run all models.
    all_records: list[dict] = []
    merged_examples: dict[str, list] = {}

    for model_name, is_instruct in models_cfg:
        print(f"\n[d5] === model: {model_name} ===")
        recs, ex = run_model(
            model_name=model_name,
            is_instruct=is_instruct,
            pairs=pairs,
            depths=depths,
            reps=args.reps,
            templates=templates,
            seed=args.seed,
            limit=args.limit,
        )
        all_records.extend(recs)
        # Merge example prompts: keep up to 3 per condition across models.
        for cond, bucket in ex.items():
            dst = merged_examples.setdefault(cond, [])
            for item in bucket:
                if len(dst) < 3:
                    dst.append(item)

    # Persist records.
    records_path = args.out + "_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[d5] {len(all_records)} records -> {os.path.relpath(records_path, REPO_ROOT)}")

    # Compute and persist summary.
    summary = compute_summary(all_records, pairs, models_cfg, depths, args.boot_seed)
    summary["config"] = {
        "pairs": [p.name for p in pairs],
        "models": [m for m, _ in models_cfg],
        "depths": depths,
        "reps": args.reps,
        "templates": templates,
        "seed": args.seed,
        "boot_seed": args.boot_seed,
    }
    summary["n_records"] = len(all_records)
    summary["timestamp_utc"] = utc_now_iso()
    summary["environment"] = environment_fingerprint()

    summary_path = args.out + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[d5] summary -> {os.path.relpath(summary_path, REPO_ROOT)}")

    # Persist example prompts for the paper appendix.
    ex_path = os.path.join(results_dir, "example_prompts.json")
    with open(ex_path, "w", encoding="utf-8") as fh:
        json.dump(merged_examples, fh, ensure_ascii=False, indent=2)
    print(f"[d5] example prompts -> {os.path.relpath(ex_path, REPO_ROOT)}")

    # Figure.
    if all_records:
        make_figure(
            all_records, pairs, models_cfg, depths,
            os.path.join(args.fig_dir, "d5_arms.png"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
