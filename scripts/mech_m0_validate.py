#!/usr/bin/env python
"""M0 gate: validate TL numerics, proxy-vs-full correlation, and stimulus alignment.

Steps:
  1. Build stimuli; assert alignment; print span table for 2 example bases.
  2. Build proxy lexicon; print surviving/dropped strings.
  3. TL-vs-HF numerics: final logits argmax + max |logprob diff| on 6 stimuli.
  4. Proxy validation: proxy_pull (TL final layer) vs full continuation-scored prior_pull
     (existing HF backend + compute_metric_row) for all conditions x depths.
     Gate: Spearman >= 0.7 AND sign agreement >= 0.9; exit 1 on failure.
  5. Scatter figure mech/figures/m0_proxy_vs_full.png.
  6. Write mech/results/m0_validation.json.

  python scripts/mech_m0_validate.py [--models MODEL] [--items N] [--seed SEED] [--out DIR]
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
                    help="comma-separated HF model names (default: both MECH_MODELS)")
    ap.add_argument("--items", type=int, default=None,
                    help="cap total stimuli (takes first N regardless of condition balance; for quick smoke runs)")
    ap.add_argument("--depths", default="0,512",
                    help="comma-separated filler token depths (default: 0,512)")
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mech", "results"),
                    help="output directory (default: mech/results/)")
    ap.add_argument("--figs", default=os.path.join(REPO_ROOT, "mech", "figures"),
                    help="figure output directory (default: mech/figures/)")
    return ap.parse_args()


def _model_short(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    from alias_inertia.backends import build_backend
    from alias_inertia.determinism import environment_fingerprint, utc_now_iso
    from alias_inertia.lexicons import LEXICONS
    from alias_inertia.metrics import compute_metric_row
    from alias_inertia.mech.env import MECH_MODELS, load_model
    from alias_inertia.mech.manifest import update_manifest
    from alias_inertia.mech.proxy import build_proxy_lexicon, proxy_pull
    from alias_inertia.mech.stimuli_mech import build_mech_stimuli

    models = [m.strip() for m in args.models.split(",")] if args.models else list(MECH_MODELS)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    # ---- Step 1: stimuli + alignment ---------------------------------------------------
    print("[m0] building stimuli (tokenizer-level, no GPU) ...")
    # Use first model's tokenizer for stimuli (both share the same tokenizer per spec)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models[0])
    stimuli = build_mech_stimuli(
        tokenizer, depths=tuple(depths), n_per_cell=30, seed=args.seed,
        pair_names=("numpy__pandas",),
    )
    if args.items:
        stimuli = stimuli[: args.items]
    print(f"[m0] {len(stimuli)} stimuli built (pair: numpy__pandas; alignment checked).")

    # Print span table for 2 example bases.
    seen_bases: set[str] = set()
    printed = 0
    for s in stimuli:
        if s.base_id not in seen_bases and s.condition == "swapped":
            seen_bases.add(s.base_id)
            print(f"  base={s.base_id} depth={s.depth} import_span={s.import_span} "
                  f"filler_span={s.filler_span} use_alias_pos={s.use_alias_pos} "
                  f"final_pos={s.final_pos} n_tokens={len(s.token_ids)}")
            printed += 1
            if printed >= 2:
                break

    # ---- Step 2: proxy lexicon ---------------------------------------------------------
    print("[m0] building proxy lexicon ...")
    lex = build_proxy_lexicon(tokenizer)
    print(f"  prior_ids ({lex.prior_lib}): {len(lex.prior_ids)} tokens: "
          f"{lex.prior_strings}")
    print(f"  bound_ids ({lex.bound_lib}): {len(lex.bound_ids)} tokens: "
          f"{lex.bound_strings}")
    print(f"  dropped (collision): {lex.dropped}")

    # ---- Step 3 + 4: per-model TL-vs-HF + proxy-vs-full validation --------------------
    all_results = {}
    passed = True

    for model_name in models:
        short = _model_short(model_name)
        print(f"\n[m0] model: {model_name}")
        import torch

        # Two passes so TL and HF weights never co-reside on the 8 GB GPU:
        # pass 1 = TL (proxy pulls + probe logprobs), pass 2 = HF (numerics + full metric).
        probe_stims = []
        seen_cond_depth: set[tuple] = set()
        for s in stimuli:
            key = (s.condition, s.depth)
            if key not in seen_cond_depth:
                seen_cond_depth.add(key)
                probe_stims.append(s)
            if len(probe_stims) >= 6:
                break

        # ---- pass 1: TL ----
        model_tl = load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        model_tl.eval()

        tl_probe_logsoftmax = []
        for s in probe_stims:
            input_ids = torch.tensor([s.token_ids], device=model_tl.cfg.device)
            with torch.no_grad():
                tl_logits = model_tl(input_ids)[0, -1, :].float().cpu()  # [V]
            tl_probe_logsoftmax.append(torch.log_softmax(tl_logits, dim=-1).numpy())

        proxy_vals, conditions_list = [], []
        for n_done, s in enumerate(stimuli, 1):
            input_ids = torch.tensor([s.token_ids], device=model_tl.cfg.device)
            with torch.no_grad():
                tl_logits_final = model_tl(input_ids)[0, -1, :]  # [V]
            proxy_vals.append(float(proxy_pull(tl_logits_final.unsqueeze(0), lex).squeeze().item()))
            conditions_list.append(s.condition)
            if n_done % 40 == 0:
                print(f"  [tl] proxy scored {n_done}/{len(stimuli)}")

        del model_tl
        torch.cuda.empty_cache()

        # ---- pass 2: HF backend (matches tokenization: add_special_tokens=False) ----
        hf_device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_backend = build_backend("hf", {
            "model": model_name,
            "device": hf_device,
            "dtype": "float16",
            "add_special_tokens": False,
        })

        argmax_agreements = []
        max_logprob_diffs = []
        for s, tl_log_softmax in zip(probe_stims, tl_probe_logsoftmax):
            hf_ids = hf_backend.tokenize(s.prompt)
            hf_t = torch.tensor([hf_ids], device=hf_device, dtype=torch.long)
            with torch.no_grad():
                hf_logits = hf_backend._forward_kept_logits(hf_t, 1)[-1, :].float().cpu()
            hf_log_softmax = torch.log_softmax(hf_logits, dim=-1).numpy()

            argmax_agreements.append(int(np.argmax(tl_log_softmax)) == int(np.argmax(hf_log_softmax)))
            max_logprob_diffs.append(float(np.max(np.abs(tl_log_softmax - hf_log_softmax))))

        argmax_rate = float(np.mean(argmax_agreements))
        mean_max_diff = float(np.mean(max_logprob_diffs))
        print(f"  TL-vs-HF: argmax_agree={argmax_rate:.3f} mean_max_logprob_diff={mean_max_diff:.4f} nats")

        lex_full = {"numpy": LEXICONS["numpy"], "pandas": LEXICONS["pandas"]}
        full_vals = []
        for n_done, s in enumerate(stimuli, 1):
            metric = compute_metric_row(
                hf_backend, s.prompt,
                prior_lib="numpy", bound_lib="pandas", lexicons=lex_full,
            )
            full_vals.append(float(metric["prior_pull"]))
            if n_done % 40 == 0:
                print(f"  [hf] full scored {n_done}/{len(stimuli)}")

        proxy_arr = np.array(proxy_vals)
        full_arr = np.array(full_vals)
        pearson_r, _ = pearsonr(proxy_arr, full_arr)
        spearman_r, _ = spearmanr(proxy_arr, full_arr)
        # Sign agreement on items with |full| > 1 nat; fall back to all items when < 10 strong.
        strong = np.abs(full_arr) > 1.0
        n_strong = int(strong.sum())
        if n_strong >= 10:
            sign_agree = float(np.mean(np.sign(proxy_arr[strong]) == np.sign(full_arr[strong])))
        else:
            sign_agree = float(np.mean(np.sign(proxy_arr) == np.sign(full_arr)))
            print(f"  NOTE: only {n_strong} items with |full|>1 nat; sign_agree computed on all items")
        gate_pass = (spearman_r >= 0.7) and (sign_agree >= 0.9)

        # Per-item records so the gate statistics are independently re-derivable.
        rec_path = os.path.join(args.out, "m0_records.jsonl")
        kept_rows = []
        if os.path.exists(rec_path):
            with open(rec_path, encoding="utf-8") as fh:
                kept_rows = [json.loads(line) for line in fh
                             if json.loads(line).get("model") != model_name]
        for s, pv, fv in zip(stimuli, proxy_vals, full_vals):
            kept_rows.append({
                "model": model_name, "stimulus_id": s.stimulus_id, "base_id": s.base_id,
                "condition": s.condition, "depth": s.depth, "prompt_sha256": s.prompt_sha256,
                "proxy_pull": pv, "full_prior_pull": fv,
            })
        with open(rec_path, "w", encoding="utf-8") as fh:
            for row in kept_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not gate_pass:
            passed = False
        print(f"  Pearson={pearson_r:.4f} Spearman={spearman_r:.4f} "
              f"sign_agree={sign_agree:.3f} gate={'PASS' if gate_pass else 'FAIL'}")

        # Scatter figure.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = {"conventional": "#2ca02c", "swapped": "#d62728", "no_prior": "#1f77b4"}
        for cond in ("conventional", "swapped", "no_prior"):
            mask = np.array([c == cond for c in conditions_list])
            ax.scatter(full_arr[mask], proxy_arr[mask], s=12, alpha=0.5,
                       color=cmap.get(cond, "gray"), label=cond)
        lim = max(abs(full_arr).max(), abs(proxy_arr).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("full prior_pull (nats)")
        ax.set_ylabel("proxy_pull (TL final layer)")
        ax.set_title(f"M0 proxy vs full | {short}\nSpearman={spearman_r:.3f}")
        ax.legend(fontsize=7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        fig_path = os.path.join(args.figs, f"m0_proxy_vs_full_{short}.png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  -> {os.path.relpath(fig_path, REPO_ROOT)}")

        all_results[model_name] = {
            "tl_vs_hf": {
                "n_probed": len(probe_stims),
                "argmax_agreement": argmax_rate,
                "mean_max_logprob_diff": mean_max_diff,
            },
            "proxy_vs_full": {
                "n_items": len(proxy_vals),
                "pearson": float(pearson_r),
                "spearman": float(spearman_r),
                "sign_agreement_strong": float(sign_agree),
                "n_strong": n_strong,
                "gate_pass": gate_pass,
            },
        }

        # Free model before next.
        try:
            del model_tl
        except Exception:
            pass
        try:
            hf_backend.close()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Write results (merge with any prior per-model run) ----------------------------
    out_path = os.path.join(args.out, "m0_validation.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                prior = json.load(fh)
            merged = dict(prior.get("models", {}))
            merged.update(all_results)
            all_results = merged
            passed = passed and all(
                r["proxy_vs_full"]["gate_pass"] for r in all_results.values()
            )
            # Config must describe ALL data in the file, not the last invocation.
            models = sorted(set(prior.get("config", {}).get("models", [])) | set(models))
        except (json.JSONDecodeError, KeyError):
            pass
    payload = {
        "timestamp_utc": utc_now_iso(),
        "config": {
            "models": models, "depths": depths,
            "items_cap": args.items, "seed": args.seed,
        },
        "stimuli_count": len(stimuli),
        "proxy_lexicon": {
            "prior_lib": lex.prior_lib, "bound_lib": lex.bound_lib,
            "prior_strings": lex.prior_strings, "bound_strings": lex.bound_strings,
            "dropped": lex.dropped, "sha256": lex.sha256,
        },
        "models": all_results,
        "gate_overall": passed,
        "environment": environment_fingerprint(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[m0] wrote {os.path.relpath(out_path, REPO_ROOT)}")

    update_manifest("m0_validate", {
        "gate_overall": passed,
        "models": list(all_results.keys()),
        "n_stimuli": len(stimuli),
    })

    if not passed:
        print("[m0] GATE FAILED: Spearman < 0.7 or sign_agreement < 0.9 on at least one model.")
        return 1
    print("[m0] GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
