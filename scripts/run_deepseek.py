#!/usr/bin/env python
"""Run the DeepSeek API behavioral probe over the binding-condition grid.

Three API-feasible tasks (forced choice, free generation, verbal recognition) are run in both
non-thinking and thinking modes across depths up to 128k+ filler tokens. This single experiment
covers the reviewer asks the local continuation-scoring arm cannot: a frontier hosted model, a
long-context sweep, and whether explicit reasoning (thinking mode) overrides the alias prior.

The API key is read from DEEPSEEK_API_KEY for this process only and is never written to the
results, the response cache key, or the manifest. Responses are cached on disk so re-runs are
free and deterministic; --dry-run prints the grid size and spends nothing.

  DEEPSEEK_API_KEY=... python scripts/run_deepseek.py --dry-run
  DEEPSEEK_API_KEY=... python scripts/run_deepseek.py --depths 0,2048 --deep-depths 8192,32768,131072

Filler depth is targeted with a tiktoken proxy tokenizer (the DeepSeek tokenizer is not shipped
offline); the TRUE in-context length is recorded per call from the API usage field.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alias_inertia import deepseek_probe as dp  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, stable_hash, utc_now_iso  # noqa: E402
from alias_inertia.lexicons import EXTENDED_PAIRS, IMPORT_NAMES, SWAP_PAIRS, get_pair  # noqa: E402
from alias_inertia.stimuli import build_stimulus, reps_for_depth  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# tiktoken proxy tokenizer for filler depth targeting (offline; not the DeepSeek tokenizer).
_ENC = None


def count_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text))


# Verbal-answer surface forms per library key (lenient substring match).
_SURFACE = {
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "torch": ["torch", "pytorch"],
    "sklearn": ["sklearn", "scikit"],
    "xgboost": ["xgboost"],
    "matplotlib.pyplot": ["matplotlib", "pyplot"],
}


def match_verbal(answer: str, libs) -> str | None:
    low = answer.lower()
    hits = [lib for lib in libs for s in _SURFACE.get(lib, [lib]) if s in low]
    # Prefer the longest/most specific surface that matched, deterministic on ties.
    return hits[0] if hits else None


def parse_depths(s: str):
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def build_grid(args):
    """Yield (pair, condition, depth, template, rep, is_deep, nc_alias) for the configured grid.

    The no-prior condition is emitted once per nonce alias (default just "zz"); other conditions
    use a fixed placeholder alias that build_stimulus ignores.
    """
    default_pairs = EXTENDED_PAIRS if getattr(args, "extended", False) else SWAP_PAIRS
    short_pairs = [get_pair(p) for p in args.pairs] if args.pairs else list(default_pairs)
    deep_pairs = [get_pair(p) for p in args.deep_pairs] if args.deep_pairs else short_pairs[:2]
    nonce = args.nonce_aliases or ["zz"]
    conditions = args.conditions

    def aliases_for(cond):
        return nonce if cond == "no_prior" else ["zz"]

    for pair in short_pairs:
        for depth in args.depths:
            # depth 0 has no filler to vary, so extra reps would be identical prompts.
            for rep in range(reps_for_depth(depth, args.reps)):
                for template in args.templates:
                    for cond in conditions:
                        for nc in aliases_for(cond):
                            yield pair, cond, depth, template, rep, False, nc
    for pair in deep_pairs:
        for depth in args.deep_depths:
            for rep in range(reps_for_depth(depth, args.deep_reps)):
                for cond in conditions:
                    for nc in aliases_for(cond):
                        yield pair, cond, depth, "t1", rep, True, nc


def _letter_to_lib(choice, info, bound_lib, distractor_lib):
    if choice == info["bound_letter"]:
        return bound_lib
    if choice == info["distractor_letter"]:
        return distractor_lib
    return None


def forced_choice_record(client, meta, mode, rng, note=None, n_runs=1, temperature=0.0):
    prior_lib = meta["prior_target"]
    bound_lib = meta["bound_target"]
    other_lib = meta["other_lib"]
    distractor_lib = prior_lib if bound_lib != prior_lib else other_lib
    methods = dp.pick_methods(distractor_lib, bound_lib, rng)
    if methods is None:
        return None
    distractor_method, bound_method = methods
    messages, info = dp.forced_choice_messages(meta, distractor_method, bound_method, rng, note=note)
    nothink = mode == "nothink"
    base = {
        "task": "forced_choice",
        "distractor_lib": distractor_lib,
        "distractor_method": distractor_method,
        "bound_method": bound_method,
        "option_a": info["option_a"],
        "option_b": info["option_b"],
        "distractor_letter": info["distractor_letter"],
        "bound_letter": info["bound_letter"],
        "salience": note is not None,
    }

    if n_runs > 1:
        # temperature>0 sampling, N draws (distinct seeds): per-item P(prior) as a graded estimate
        # that does not depend on the API exposing arbitrary-continuation log-probs.
        prior_hits = bound_hits = valid = 0
        choices = []
        last = None
        for i in range(n_runs):
            res = client.chat(messages, thinking=(False if nothink else True),
                              max_tokens=(4 if nothink else 6000), temperature=temperature, seed=i)
            last = res
            ch = dp.parse_choice(res)
            lib = _letter_to_lib(ch, info, bound_lib, distractor_lib)
            choices.append(ch)
            if lib is not None:
                valid += 1
                prior_hits += int(lib == prior_lib)
                bound_hits += int(lib == bound_lib)
        p_prior = (prior_hits / valid) if valid else None
        base.update({
            "n_runs": n_runs, "temperature": temperature, "n_valid": valid,
            "choices": "".join(c or "?" for c in choices),
            "p_prior_lib": p_prior,
            "chose_prior_lib": p_prior,                       # continuous in [0,1] for the analyzer
            "correct": (bound_hits / valid) if valid else None,
            "chosen_logprob": None,
            "finish_reason": last.finish_reason if last else None,
            "usage": last.usage if last else {},
            "from_cache": last.cached if last else False,
        })
        return base

    res = client.chat(
        messages,
        thinking=(False if nothink else True),
        max_tokens=(4 if nothink else 6000),
        temperature=temperature,
        logprobs=nothink,
        top_logprobs=12,
    )
    choice = dp.parse_choice(res)
    choice_lib = _letter_to_lib(choice, info, bound_lib, distractor_lib)
    lp = dp.chosen_logprob(res, choice) if choice else None
    base.update({
        "raw_content": res.content[:80],
        "choice": choice,
        "choice_lib": choice_lib,
        "chose_prior_lib": (choice_lib == prior_lib) if choice_lib else None,
        "correct": (choice_lib == bound_lib) if choice_lib else None,
        "chosen_logprob": lp,
        "finish_reason": res.finish_reason,
        "usage": res.usage,
        "from_cache": res.cached,
    })
    return base


def generation_record(client, meta, mode, note=None):
    prior_lib = meta["prior_target"]
    bound_lib = meta["bound_target"]
    nothink = mode == "nothink"
    messages = dp.generation_messages(meta, note=note)
    res = client.chat(
        messages,
        thinking=(False if nothink else True),
        max_tokens=(24 if nothink else 6000),
        temperature=0.0,
    )
    attr = dp.parse_attribute(res.content, alias=meta["alias"])
    rb = resolves_on(attr, bound_lib) if attr else {"status": "unknown_attr", "exists": None}
    rp = resolves_on(attr, prior_lib) if attr else {"status": "unknown_attr", "exists": None}
    return {
        "task": "generation",
        "raw_content": res.content[:120],
        "attribute": attr,
        "resolves_bound": rb["exists"],
        "resolves_prior": rp["exists"],
        "broken_on_bound": (rb["exists"] is False) if attr else None,
        "prior_only": bool(rp["exists"]) and (rb["exists"] is False) if attr else None,
        "salience": note is not None,
        "finish_reason": res.finish_reason,
        "usage": res.usage,
        "from_cache": res.cached,
    }


def verbal_record(client, meta, mode, note=None):
    prior_lib = meta["prior_target"]
    bound_lib = meta["bound_target"]
    other_lib = meta["other_lib"]
    nothink = mode == "nothink"
    messages = dp.verbal_messages(meta, note=note)
    res = client.chat(
        messages,
        thinking=(False if nothink else True),
        max_tokens=(16 if nothink else 6000),
        temperature=0.0,
    )
    libs = sorted({prior_lib, bound_lib, other_lib})
    matched = match_verbal(res.content, libs)
    return {
        "task": "verbal",
        "raw_content": res.content[:120],
        "matched_lib": matched,
        "correct": (matched == bound_lib) if matched else None,
        "named_prior": (matched == prior_lib) if matched else None,
        "salience": note is not None,
        "finish_reason": res.finish_reason,
        "usage": res.usage,
        "from_cache": res.cached,
    }


TASK_FNS = {
    "forced_choice": forced_choice_record,
    "generation": generation_record,
    "verbal": verbal_record,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=dp.DEFAULT_MODEL)
    ap.add_argument("--pairs", default="", help="comma-separated pair names for short bins (default: all 6)")
    ap.add_argument("--deep-pairs", default="numpy__pandas,pandas__numpy")
    ap.add_argument("--depths", default="0,2048", type=str)
    ap.add_argument("--deep-depths", default="8192,32768,131072", type=str)
    ap.add_argument("--templates", default="t1", help="comma-separated template ids for short bins")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--deep-reps", type=int, default=1)
    ap.add_argument("--conditions", default="conventional,swapped,no_prior")
    ap.add_argument("--modes", default="nothink,think")
    ap.add_argument("--tasks", default="forced_choice,generation,verbal")
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--max-calls", type=int, default=2000)
    ap.add_argument("--nonce-aliases", default="", help="comma-separated nonce aliases for no_prior (default: zz)")
    ap.add_argument("--salience", action="store_true", help="prepend an explicit binding cue (intervention)")
    ap.add_argument("--extended", action="store_true", help="use the 12-alias EXTENDED_PAIRS set")
    ap.add_argument("--n-runs", type=int, default=1, help="forced-choice samples per item (temp>0 -> graded P(prior))")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "deepseek_raw.jsonl"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".cache", "deepseek"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.pairs = [p for p in args.pairs.split(",") if p]
    args.deep_pairs = [p for p in args.deep_pairs.split(",") if p]
    args.depths = parse_depths(args.depths)
    args.deep_depths = parse_depths(args.deep_depths)
    args.templates = [t for t in args.templates.split(",") if t]
    args.conditions = [c for c in args.conditions.split(",") if c]
    args.nonce_aliases = [a for a in args.nonce_aliases.split(",") if a]
    modes = [m for m in args.modes.split(",") if m]
    tasks = [t for t in args.tasks.split(",") if t]

    grid = list(build_grid(args))
    n_units = len(grid)
    n_calls_planned = n_units * len(modes) * len(tasks)
    print(f"grid: {n_units} stimulus units x {len(modes)} modes x {len(tasks)} tasks "
          f"= {n_calls_planned} calls (before forced_choice skips)")
    print(f"  modes={modes} tasks={tasks}")
    print(f"  short: pairs={args.pairs or 'all6'} depths={args.depths} templates={args.templates} reps={args.reps}")
    print(f"  deep:  pairs={args.deep_pairs} depths={args.deep_depths} reps={args.deep_reps}")
    if args.dry_run:
        print("dry-run: no API calls made.")
        return 0

    client = dp.DeepSeekClient(model=args.model, cache_dir=args.cache_dir, max_calls=args.max_calls)
    records = []
    n = 0
    for pair, cond, depth, template, rep, is_deep, nc_alias in grid:
        stim = build_stimulus(
            pair=pair, condition=cond, depth_tokens=depth, template_id=template,
            rep=rep, count_tokens=count_tokens, seed=args.seed, non_canonical_alias=nc_alias,
        )
        meta = stim.meta
        note = dp.salience_note(meta) if args.salience else None
        coords = {
            "model": args.model,
            "pair": meta["pair"], "tier": pair.tier, "condition": cond,
            "alias": meta["alias"], "prior_lib": meta["prior_target"],
            "bound_lib": meta["bound_target"], "other_lib": meta["other_lib"],
            "depth_target": depth, "depth_actual_tiktoken": meta["depth_tokens_actual"],
            "template_id": template, "rep": rep, "is_deep": is_deep,
            "stimulus_id": meta["stimulus_id"],
        }
        for mode in modes:
            # forced_choice option order is seeded per (stimulus, mode) with a DETERMINISTIC
            # hash (builtin hash() is per-process randomised and would break reproducibility).
            rng = random.Random(int(stable_hash([args.seed, meta["stimulus_id"], mode], length=12), 16))
            for task in tasks:
                fn = TASK_FNS[task]
                try:
                    rec = (fn(client, meta, mode, rng, note=note,
                              n_runs=args.n_runs, temperature=args.temperature)
                           if task == "forced_choice" else fn(client, meta, mode, note=note))
                except dp.DeepSeekError as e:
                    print(f"  STOP: {e}")
                    _flush(records, args.out)
                    _write_manifest(client, args, modes, tasks, n_units, n, REPO_ROOT)
                    return 1
                if rec is None:
                    continue
                rec = {**coords, "mode": mode, **rec}
                # in-context length actually seen by the model, from the API
                rec["prompt_tokens_api"] = int(rec.get("usage", {}).get("prompt_tokens", 0) or 0)
                records.append(rec)
                n += 1
                if n % 20 == 0:
                    print(f"  {n} records | live={client.live_calls} cache={client.cache_hits} "
                          f"| in={client.total_prompt_tokens} out={client.total_completion_tokens} tok")
    _flush(records, args.out)
    _write_manifest(client, args, modes, tasks, n_units, n, REPO_ROOT)
    print(f"done: {n} records, {client.live_calls} live calls, {client.cache_hits} cache hits")
    print(f"  tokens: in={client.total_prompt_tokens} out={client.total_completion_tokens}")
    print(f"  -> {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


def _flush(records, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_manifest(client, args, modes, tasks, n_units, n_records, repo_root):
    manifest = {
        "deepseek_probe_version": dp.DEEPSEEK_PROBE_VERSION,
        "timestamp_utc": utc_now_iso(),
        "fingerprint": client.fingerprint(),  # contains no key
        "modes": modes,
        "tasks": tasks,
        "conditions": args.conditions,
        "short": {"pairs": args.pairs or "all6", "depths": args.depths,
                  "templates": args.templates, "reps": args.reps},
        "deep": {"pairs": args.deep_pairs, "depths": args.deep_depths, "reps": args.deep_reps},
        "seed": args.seed,
        "n_records": n_records,
        "live_calls": client.live_calls,
        "cache_hits": client.cache_hits,
        "tokens_in": client.total_prompt_tokens,
        "tokens_out": client.total_completion_tokens,
        "filler_token_counter": "tiktoken/cl100k_base (proxy; true length in prompt_tokens_api)",
        "environment": environment_fingerprint(),
    }
    path = os.path.join(repo_root, "results", "deepseek_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
