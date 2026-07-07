#!/usr/bin/env python
"""M3: per-head attention-to-import and direct logit attribution (DLA).

For each model x depth:
  - attention_to_import: per stimulus x layer x head, attention mass from the
    final position onto the import span (attn_import_span), use_alias (attn_use_alias),
    and final_pos (attn_final_pos).
  - head_dla: DLA of every head at the final position via dla_contribution;
    head_class is prior_promoting or binding_promoting.

Records from both functions are indexed by (stimulus_id, layer, head) and merged.

Outputs:
  mech/results/m3_records.jsonl        per stimulus x head x layer merged records
  mech/results/m3_summary.json         top-k by |DLA| on swapped, per condition means
  mech/results/m3_top_heads.md         markdown table: layer, head, DLA swapped/no_prior,
                                        import-attn swapped/no_prior, class
  mech/figures/m3_heads_{short}.png    bar chart: top-k heads colored by class

  python scripts/mech_m3_heads.py [--models M] [--depths D] [--items N] [--top-k K] [--seed S] [--out DIR]
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
    ap.add_argument("--models", default=None,
                    help="comma-separated HF model names (default: MECH_MODELS)")
    ap.add_argument("--depths", default="0,512",
                    help="comma-separated filler depths (default: 0,512)")
    ap.add_argument("--items", type=int, default=None,
                    help="cap items per condition per depth")
    ap.add_argument("--top-k", type=int, default=15,
                    help="top-k heads by |DLA| in table/figure (default: 15)")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mech", "results"),
                    help="output directory (default: mech/results/)")
    ap.add_argument("--figs", default=os.path.join(REPO_ROOT, "mech", "figures"),
                    help="figure directory (default: mech/figures/)")
    return ap.parse_args()


def _model_short(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _write_md_table(rows, out_path):
    header = ("layer", "head", "DLA_swapped", "DLA_no_prior",
              "import_attn_swapped", "import_attn_no_prior", "class")
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        cells = [
            str(r["layer"]), str(r["head"]),
            f"{r['dla_swapped']:.4f}", f"{r['dla_no_prior']:.4f}",
            f"{r['import_attn_swapped']:.4f}", f"{r['import_attn_no_prior']:.4f}",
            r["class"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    import numpy as np
    import torch

    from alias_inertia.determinism import environment_fingerprint, utc_now_iso
    from alias_inertia.mech.env import MECH_MODELS, load_model
    from alias_inertia.mech.heads import attention_to_import, head_dla
    from alias_inertia.mech.manifest import update_manifest
    from alias_inertia.mech.proxy import build_proxy_lexicon
    from alias_inertia.mech.stimuli_mech import build_mech_stimuli
    from transformers import AutoTokenizer

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MECH_MODELS)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    tokenizer = AutoTokenizer.from_pretrained(models[0])
    stimuli_all = build_mech_stimuli(
        tokenizer, depths=tuple(depths), n_per_cell=30, seed=args.seed,
        pair_names=("numpy__pandas",),
    )
    lex = build_proxy_lexicon(tokenizer)

    all_records: list[dict] = []
    all_summaries: dict = {}
    all_top_heads: list[dict] = []

    for model_name in models:
        short = _model_short(model_name)
        print(f"\n[m3] model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_tl = load_model(model_name, device=device)
        model_tl.eval()
        n_layers = model_tl.cfg.n_layers
        n_heads = model_tl.cfg.n_heads

        stims = stimuli_all[:]
        if args.items:
            cap: list = []
            cnt: dict[str, int] = {}
            for s in stims:
                if cnt.get(s.condition, 0) < args.items:
                    cnt[s.condition] = cnt.get(s.condition, 0) + 1
                    cap.append(s)
            stims = cap

        print(f"  running attention_to_import ({len(stims)} stimuli) ...")
        attn_recs = attention_to_import(model_tl, stims)

        print(f"  running head_dla ({len(stims)} stimuli) ...")
        dla_recs = head_dla(model_tl, stims, lex)

        # Index DLA records by (stimulus_id, layer, head).
        dla_idx: dict[tuple, dict] = {}
        for r in dla_recs:
            dla_idx[(r["stimulus_id"], r["layer"], r["head"])] = r

        # Merge: attn records are the primary, DLA fields are added from the index.
        merged: list[dict] = []
        for r in attn_recs:
            key = (r["stimulus_id"], r["layer"], r["head"])
            dla_r = dla_idx.get(key, {})
            row = dict(r)
            row["model"] = model_name
            # DLA fields: dla_contribution, head_class, final_proxy_pull
            for field in ("dla_contribution", "head_class", "final_proxy_pull"):
                row[field] = dla_r.get(field)
            merged.append(row)

        all_records.extend(merged)
        print(f"  {len(merged)} head-level records merged")

        # Top-k heads by mean |DLA contribution| on swapped items.
        swapped_merged = [r for r in merged if r.get("condition") == "swapped"]
        noprior_merged = [r for r in merged if r.get("condition") == "no_prior"]

        dla_sw: dict[tuple, list] = {}
        attn_sw: dict[tuple, list] = {}
        dla_np: dict[tuple, list] = {}
        attn_np: dict[tuple, list] = {}

        for r in swapped_merged:
            lh = (r["layer"], r["head"])
            val = r.get("dla_contribution")
            if val is not None:
                dla_sw.setdefault(lh, []).append(float(val))
            attn_sw.setdefault(lh, []).append(float(r.get("attn_import_span", 0.0)))
        for r in noprior_merged:
            lh = (r["layer"], r["head"])
            val = r.get("dla_contribution")
            if val is not None:
                dla_np.setdefault(lh, []).append(float(val))
            attn_np.setdefault(lh, []).append(float(r.get("attn_import_span", 0.0)))

        all_lh = set(dla_sw.keys()) | set(dla_np.keys())
        head_rows: list[dict] = []
        for (layer, head) in sorted(all_lh):
            d_sw = float(np.mean(dla_sw.get((layer, head), [0.0])))
            d_np = float(np.mean(dla_np.get((layer, head), [0.0])))
            a_sw = float(np.mean(attn_sw.get((layer, head), [0.0])))
            a_np = float(np.mean(attn_np.get((layer, head), [0.0])))
            head_cls = "prior_promoting" if d_sw > 0 else "binding_promoting"
            head_rows.append({
                "model": model_name,
                "layer": int(layer), "head": int(head),
                "dla_swapped": d_sw, "dla_no_prior": d_np,
                "import_attn_swapped": a_sw, "import_attn_no_prior": a_np,
                "class": head_cls,
                "abs_dla_swapped": abs(d_sw),
            })

        head_rows.sort(key=lambda x: -x["abs_dla_swapped"])
        top_k = head_rows[: args.top_k]
        all_top_heads.extend(top_k)

        # Figure: bar chart of top-k heads.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, args.top_k * 0.7), 4))
        labels = [f"L{r['layer']}H{r['head']}" for r in top_k]
        values = [r["dla_swapped"] for r in top_k]
        colors = ["#d62728" if r["class"] == "prior_promoting" else "#1f77b4" for r in top_k]
        ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_ylabel("mean DLA on swapped (nats)")
        ax.set_title(f"M3 top-{args.top_k} heads | {short}\nred=prior_promoting  blue=binding_promoting")
        fig.tight_layout()
        fig_path = os.path.join(args.figs, f"m3_heads_{short}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  -> {os.path.relpath(fig_path, REPO_ROOT)}")

        all_summaries[model_name] = {
            "n_layers": n_layers, "n_heads": n_heads,
            "top_heads": top_k,
            "n_records": len(merged),
        }

        del model_tl
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Write outputs (merge with prior per-model runs so split processes don't clobber) ----
    rec_path = os.path.join(args.out, "m3_records.jsonl")
    if os.path.exists(rec_path):
        kept = []
        with open(rec_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("model") not in models:
                    kept.append(row)
        all_records = kept + all_records
    with open(rec_path, "w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    sum_path = os.path.join(args.out, "m3_summary.json")
    if os.path.exists(sum_path):
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
        "config": {"models": models, "depths": depths, "items_cap": args.items,
                   "top_k": args.top_k, "seed": args.seed},
        "n_records": len(all_records),
        "models": all_summaries,
        "environment": environment_fingerprint(),
    }
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)

    # Markdown table: regenerate from merged summary so all models present are covered.
    md_path = os.path.join(args.out, "m3_top_heads.md")
    merged_top_heads: list[dict] = []
    for v in all_summaries.values():
        merged_top_heads.extend(v.get("top_heads", []))
    merged_top_heads.sort(key=lambda x: -x["abs_dla_swapped"])
    _write_md_table(merged_top_heads, md_path)

    print(f"\n[m3] {len(all_records)} records -> {os.path.relpath(rec_path, REPO_ROOT)}")
    print(f"[m3] summary -> {os.path.relpath(sum_path, REPO_ROOT)}")
    print(f"[m3] markdown table -> {os.path.relpath(md_path, REPO_ROOT)}")

    update_manifest("m3_heads", {
        "models": models, "n_records": len(all_records),
        "top_k": args.top_k,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
