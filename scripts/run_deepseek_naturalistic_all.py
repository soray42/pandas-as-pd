#!/usr/bin/env python
"""DeepSeek-V4-Pro on the naturalistic task-context arm across ALL 12 alias conventions.

Frontier-model generalisation of scripts/run_deepseek_naturalistic.py: for each EXTENDED_PAIRS
pair, take its bound library's task scenarios, rebind the alias (swapped / no-prior / correct),
let the model complete the line, and check whether the accessed attribute resolves on the bound
library and whether it is the operation the task needs. Run in direct (non-thinking) and thinking
modes. The API cannot teacher-force, so the metric is broken-call and wrote-correct-op rates.

  DEEPSEEK_API_KEY=... python scripts/run_deepseek_naturalistic_all.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia import deepseek_probe as dp  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, sha256_text, utc_now_iso  # noqa: E402
from alias_inertia.lexicons import CANONICAL_ALIASES, EXTENDED_PAIRS, IMPORT_NAMES  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402
from scripts.run_naturalistic_all import load_scenarios_by_lib  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def gen_messages(bound_lib, alias, setup, use_var):
    code = f"import {IMPORT_NAMES[bound_lib]} as {alias}\n{setup.rstrip()}\n{use_var} = {alias}."
    user = (
        "Complete the LAST line of this Python script. Output only the code that comes after "
        f"`{alias}.` to finish that one line, nothing else (no explanation, no code fence).\n\n"
        "```python\n" + code + "\n```"
    )
    return [{"role": "user", "content": user}], code


def boot_rate(vals, rng, n=4000):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None, None, None
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandas", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios.json"))
    ap.add_argument("--more", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios_more.json"))
    ap.add_argument("--model", default=dp.DEFAULT_MODEL)
    ap.add_argument("--modes", default="nothink,think")
    ap.add_argument("--max-scen", type=int, default=12, help="cap scenarios per bound lib (cost control)")
    ap.add_argument("--max-calls", type=int, default=1600)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "deepseek_naturalistic_all"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".cache", "deepseek"))
    args = ap.parse_args()

    by_lib = load_scenarios_by_lib(args.pandas, args.more)
    by_lib = {k: v[: args.max_scen] for k, v in by_lib.items()}
    pairs = [p for p in EXTENDED_PAIRS if by_lib.get(p.other_lib)]
    modes = [m for m in args.modes.split(",") if m]
    print("scenarios per lib (capped):", {k: len(v) for k, v in by_lib.items()}, "| pairs:", len(pairs))
    client = dp.DeepSeekClient(model=args.model, cache_dir=args.cache_dir, max_calls=args.max_calls)

    records, n = [], 0
    for pair in pairs:
        prior, bound = pair.prior_lib, pair.other_lib
        conds = [("swapped", pair.treatment_alias), ("no_prior", "zz"), ("correct", CANONICAL_ALIASES[bound])]
        for s in by_lib[bound]:
            for cond, alias in conds:
                messages, _ = gen_messages(bound, alias, s["setup"], s["use_var"])
                for mode in modes:
                    nothink = mode == "nothink"
                    try:
                        res = client.chat(messages, thinking=(False if nothink else True),
                                          max_tokens=(24 if nothink else 6000), temperature=0.0)
                    except dp.DeepSeekError as e:
                        print(f"  STOP: {e}"); _save(records, args, modes, pairs, client); return 1
                    attr = dp.parse_attribute(res.content, alias=alias)
                    rb = resolves_on(attr, bound) if attr else {"exists": None}
                    records.append({
                        "model": args.model, "pair": pair.name, "prior_lib": prior, "bound_lib": bound,
                        "tier": pair.tier, "target_op": s["target_op"], "condition": cond, "alias": alias,
                        "mode": mode, "gen_attr": attr, "gen_text": res.content.strip()[:60],
                        "broken_on_bound": (rb.get("exists") is False) if attr else None,
                        "wrote_target_op": (attr == s["target_op"]) if attr else False,
                        "finish_reason": res.finish_reason, "prompt_sha256": sha256_text(messages[0]["content"]),
                        "from_cache": res.cached})
                    n += 1
                    if n % 40 == 0:
                        print(f"  {n} calls | live={client.live_calls} cache={client.cache_hits}")
        print(f"  {pair.name} done ({n} calls so far)")
    _save(records, args, modes, pairs, client)
    return 0


def _save(records, args, modes, pairs, client):
    with open(args.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(20260618)
    out = {"model": args.model, "modes": modes, "fingerprint": client.fingerprint(),
           "live_calls": client.live_calls, "per_mode": {}, "timestamp_utc": utc_now_iso()}
    for mode in modes:
        mr = [r for r in records if r["mode"] == mode]
        block = {"overall": {}, "per_pair": {}}
        for c in ("correct", "no_prior", "swapped"):
            rs = [r for r in mr if r["condition"] == c]
            br = boot_rate([r["broken_on_bound"] for r in rs if r["broken_on_bound"] is not None], rng)
            wt = boot_rate([1.0 if r["wrote_target_op"] else 0.0 for r in rs], rng)
            block["overall"][c] = {"broken": br[0], "broken_ci": [br[1], br[2]],
                                   "wrote_target": wt[0], "n": len(rs),
                                   "truncated": sum(1 for r in rs if r["finish_reason"] == "length")}
        for pair in pairs:
            pr = [r for r in mr if r["pair"] == pair.name and r["condition"] == "swapped"]
            br = [r["broken_on_bound"] for r in pr if r["broken_on_bound"] is not None]
            block["per_pair"][pair.name] = {"tier": pair.tier,
                "swapped_broken": float(np.mean(br)) if br else None,
                "swapped_wrote_target": float(np.mean([1.0 if r["wrote_target_op"] else 0.0 for r in pr])) if pr else None,
                "n": len(pr)}
        out["per_mode"][mode] = block
    out["environment"] = environment_fingerprint()
    with open(args.out + "_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.relpath(args.out + '_results.json', REPO_ROOT)}")
    for mode in modes:
        ov = out["per_mode"][mode]["overall"]
        print(f"\n[{args.model} / {mode}] broken-on-bound (correct / no_prior / swapped):")
        for c in ("correct", "no_prior", "swapped"):
            x = ov[c]
            print(f"  {c:9} broken={x['broken']} wrote_target={x['wrote_target']} n={x['n']} trunc={x['truncated']}")


if __name__ == "__main__":
    raise SystemExit(main())
