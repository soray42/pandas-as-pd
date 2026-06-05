#!/usr/bin/env python
"""FULL RUN: dose-axis grid (pairs x conditions x distance) x model suite, plus the behavioral
generation arm and the consequence/validity arm.

  python scripts/run.py --config configs/full.yaml [--models a,b] [--limit N]

Per model: load -> score its grid (cached) -> generate+classify+validity on a subset -> free GPU.
Resumable: scores and generations are content-addressed on disk, so re-running after an
interruption recomputes only what's missing. Outputs:
  results/full.parquet              one row per (model x stimulus): prior_pull, bound_mass, ...
  results/full_generations.jsonl    generation arm + classification + broken-call validity
  results/full_stimuli_meta.jsonl   stimulus metadata (prompts are regenerable from seed+coords)
  results/full_manifest.json        config + per-model fingerprints + environment + timings
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, set_determinism, stable_hash, utc_now_iso  # noqa: E402
from alias_inertia.generation import GEN_VERSION, classify_generation  # noqa: E402
from alias_inertia.lexicons import LEXICONS, get_pair  # noqa: E402
from alias_inertia.metrics import compute_metric_row  # noqa: E402
from alias_inertia.scoring import SCORING_VERSION, CachingScorer, DiskCache  # noqa: E402
from alias_inertia.stimuli import generate_grid  # noqa: E402
from alias_inertia.validity import VALIDITY_VERSION, library_available, resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def free_gpu():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--models", default=None, help="comma-separated subset of model labels to run")
    ap.add_argument("--limit", type=int, default=None, help="cap stimuli per (model,pair) for a smoke run")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg["seed"])
    det_state = set_determinism(seed)
    scoring_version = cfg.get("scoring_version", SCORING_VERSION)

    cache_dir = os.path.join(REPO_ROOT, cfg["cache"]["dir"]) if cfg["cache"]["enabled"] else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    gen_cache = DiskCache(os.path.join(cache_dir, "generations")) if cache_dir else None

    pairs_all = cfg["pairs_all"]
    pairs_cpu = cfg["pairs_cpu_subset"]
    gen_cfg = cfg["generation"]

    models = cfg["models"]
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m["label"] in wanted]

    rows, gen_rows, stim_meta, skipped, per_model_fp = [], [], [], [], {}
    t_start = time.time()

    for mi, mcfg in enumerate(models):
        label = mcfg["label"]
        backend_name = mcfg["backend"]
        max_depth = int(mcfg["max_depth"])
        pair_names = pairs_all if mcfg["pairs"] == "all" else pairs_cpu
        depths = [d for d in cfg["depths_tokens"] if int(d) <= max_depth]

        print(f"\n[run] === model {mi+1}/{len(models)}: {label} ({backend_name}, max_depth={max_depth}) ===")
        bcfg = dict(mcfg.get(backend_name, {}))
        try:
            backend = build_backend(backend_name, bcfg, cache_dir=cache_dir)
        except Exception as e:
            print(f"[run] FAILED to load {label}: {type(e).__name__}: {e}; skipping model")
            skipped.append({"model": label, "reason": f"load_failed: {e}"})
            continue
        scorer = CachingScorer(backend, cache_dir=cache_dir, scoring_version=scoring_version,
                               enabled=bool(cfg["cache"]["enabled"]))
        per_model_fp[label] = {
            "backend_id": backend.id, "fingerprint": backend.fingerprint(),
            "cache_id": scorer.cache_id, "scoring_code_hash": scorer.code_hash,
            **{k: mcfg.get(k) for k in ("family", "size_b", "variant", "max_depth", "pairs")},
        }
        max_ctx = getattr(backend, "max_context", 1 << 30)

        for pname in pair_names:
            pair = get_pair(pname)
            lex = {pair.prior_lib: LEXICONS[pair.prior_lib], pair.other_lib: LEXICONS[pair.other_lib]}
            max_cont = max(backend.count_tokens(c) for cc in lex.values() for c in cc)

            deep = cfg.get("deep_bins") or {}
            stimuli = list(generate_grid(
                pair=pair, conditions=cfg["conditions"], depths_tokens=depths, templates=cfg["templates"],
                repetitions=int(cfg["repetitions"]), count_tokens=backend.count_tokens, seed=seed,
                non_canonical_alias=cfg.get("non_canonical_alias", "zz"),
                depth_tolerance_tokens=int(cfg.get("depth_tolerance_tokens", 48)),
                deep_threshold=deep.get("threshold"), deep_templates=deep.get("templates"),
                deep_reps=int(deep.get("reps", 1)),
            ))
            if args.limit:
                stimuli = stimuli[: args.limit]

            t_pair = time.time()
            n_done = 0
            for s in stimuli:
                meta = s.meta
                if meta["prompt_tokens_total"] + max_cont + 1 > max_ctx:
                    skipped.append({"model": label, "stimulus_id": meta["stimulus_id"],
                                    "reason": "exceeds_max_context", "prompt_tokens": meta["prompt_tokens_total"]})
                    continue
                try:
                    metric = compute_metric_row(scorer, s.prompt, prior_lib=meta["prior_lexicon_lib"],
                                                bound_lib=meta["bound_lexicon_lib"], lexicons=lex)
                except Exception as e:  # CUDA OOM (RuntimeError) / ctx-overflow or prefix-mismatch
                    # (ValueError) / etc. -> skip this stimulus, NEVER abort the overnight run.
                    print(f"[run]   score skip on {label} {pname} d={meta['depth_tokens_target']}: "
                          f"{type(e).__name__}: {e}")
                    skipped.append({"model": label, "stimulus_id": meta["stimulus_id"],
                                    "reason": f"score_error: {type(e).__name__}: {e}"})
                    free_gpu()
                    continue

                row = {k: v for k, v in meta.items() if k != "prompt"}
                row["prompt_preview"] = s.prompt[:80].replace("\n", "\\n")
                row.update({
                    "model": label, "family": mcfg.get("family"), "size_b": mcfg.get("size_b"),
                    "variant": mcfg.get("variant"), "backend": backend_name,
                    "tier": pair.tier, "tier_rank": pair.tier_rank, "treatment_alias": pair.treatment_alias,
                    "prior_pull": metric["prior_pull"], "bound_mass": metric["bound_mass"],
                    "boundary_merge_any": metric["boundary_merge_any"],
                    "backend_id": backend.id, "scoring_version": scoring_version,
                })
                for lib, lse in metric["logsumexp_by_lib"].items():
                    row[f"logsumexp_{lib}"] = lse
                rows.append(row)
                stim_meta.append({k: v for k, v in meta.items() if k != "prompt"})

                # --- generation + validity arm (subset of cells) ---
                if (gen_cfg.get("enabled") and meta["depth_tokens_target"] in gen_cfg["depths"]
                        and meta["template_id"] in gen_cfg["templates"] and meta["rep"] < gen_cfg["reps"]):
                    try:
                        text = _cached_generate(backend, scorer.cache_id, s.prompt,
                                                int(gen_cfg["max_new_tokens"]), gen_cache)
                        cls = classify_generation(text, prior_lib=meta["prior_lexicon_lib"],
                                                  bound_lib=meta["bound_lexicon_lib"], lexicons=lex)
                        val = resolves_on(cls["attribute"], meta["bound_target"])
                        gen_rows.append({
                            "model": label, "family": mcfg.get("family"), "size_b": mcfg.get("size_b"),
                            "variant": mcfg.get("variant"), "pair": pair.name, "tier": pair.tier,
                            "condition": meta["condition"], "alias": meta["alias"],
                            "prior_target": meta["prior_target"], "bound_target": meta["bound_target"],
                            "depth_tokens_target": meta["depth_tokens_target"], "template_id": meta["template_id"],
                            "stimulus_id": meta["stimulus_id"], "prompt": s.prompt,
                            "generation": text, **{f"gen_{k}": v for k, v in cls.items()},
                            "validity_status": val["status"], "attr_resolves_on_bound": val["exists"],
                        })
                    except Exception as e:  # a bad generation must not lose the score row or abort
                        skipped.append({"model": label, "stimulus_id": meta["stimulus_id"],
                                        "reason": f"generation_error: {type(e).__name__}: {e}"})
                        free_gpu()
                n_done += 1

            print(f"[run]   {label} / {pname} ({pair.tier}): {n_done} stimuli scored "
                  f"({time.time()-t_pair:.0f}s, cache hits={scorer.stats['hits']})")

        # free this model before loading the next
        try:
            backend.close()
        except Exception:
            pass
        del backend, scorer
        free_gpu()

    # ---- persist -----------------------------------------------------------------------
    out = cfg["output"]
    df = pd.DataFrame(rows)
    rp = os.path.join(REPO_ROOT, out["results"])
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    df.to_parquet(rp, index=False)
    print(f"\n[run] wrote {len(df)} score rows -> {rp}")

    with open(os.path.join(REPO_ROOT, out["generations"]), "w", encoding="utf-8") as fh:
        for g in gen_rows:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"[run] wrote {len(gen_rows)} generations -> {out['generations']}")

    with open(os.path.join(REPO_ROOT, out["stimuli_meta"]), "w", encoding="utf-8") as fh:
        for s in stim_meta:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    manifest = {
        "created_utc": utc_now_iso(), "config_path": os.path.relpath(args.config, REPO_ROOT),
        "config": cfg, "config_hash": stable_hash(cfg, length=32),
        "scoring_version": scoring_version, "gen_version": GEN_VERSION, "validity_version": VALIDITY_VERSION,
        "determinism": det_state, "environment": environment_fingerprint(),
        "models": per_model_fp, "n_score_rows": len(df), "n_generations": len(gen_rows),
        "n_skipped": len(skipped), "skipped": skipped, "elapsed_seconds": elapsed,
        "validity_lib_available": {lib: library_available(lib) for lib in
                                   ("numpy", "pandas", "torch", "sklearn", "xgboost", "matplotlib.pyplot")},
    }
    with open(os.path.join(REPO_ROOT, out["manifest"]), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"[run] wrote manifest -> {out['manifest']}  ({elapsed/60:.1f} min total)")

    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        # Drop "-e <local path>" editable installs: they carry machine-specific paths and are not
        # dependencies of this project.
        freeze = "".join(ln for ln in freeze.splitlines(keepends=True) if not ln.startswith("-e "))
        with open(os.path.join(REPO_ROOT, "requirements-lock.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"# Environment freeze captured {utc_now_iso()}\n" + freeze)
    except Exception:
        pass
    print("\n[run] DONE. Next: python scripts/analyze.py --config", args.config)
    return 0


def _cached_generate(backend, cache_id, prompt, max_new_tokens, gen_cache):
    if gen_cache is not None:
        key = stable_hash([cache_id, "gen", max_new_tokens, prompt], length=32)
        hit = gen_cache.get(key)
        if hit is not None:
            return hit["text"]
    text = backend.generate(prompt, max_new_tokens=max_new_tokens)
    if gen_cache is not None:
        gen_cache.put(key, {"text": text})
    return text


if __name__ == "__main__":
    raise SystemExit(main())
