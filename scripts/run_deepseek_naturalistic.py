#!/usr/bin/env python
"""DeepSeek-V4-Pro on the naturalistic task-demanding arm (free generation).

The frontier API model cannot be teacher-forced, so we use the most direct probe: give it the
realistic pandas-demanding context with the library aliased to ``np`` (swapped), ``zz`` (no-prior),
or ``pd`` (correct), let it complete the line, and check whether the accessed attribute resolves on
pandas and whether it is the operation the task actually needs. Run in both direct (non-thinking)
and thinking modes. The same scenarios and validation as scripts/run_naturalistic.py; prompts hashed.

  DEEPSEEK_API_KEY=... python scripts/run_deepseek_naturalistic.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from alias_inertia import deepseek_probe as dp  # noqa: E402
from alias_inertia.determinism import environment_fingerprint, sha256_text, utc_now_iso  # noqa: E402
from alias_inertia.validity import resolves_on  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONDITIONS = [("swapped", "import pandas as np", "np"),
              ("no_prior", "import pandas as zz", "zz"),
              ("correct", "import pandas as pd", "pd")]


def gen_messages(import_line, setup, use_var, alias):
    code = f"{import_line}\n{setup.rstrip()}\n{use_var} = {alias}."
    user = (
        "Complete the LAST line of this Python script. Output only the code that comes after "
        f"`{alias}.` to finish that one line, nothing else (no explanation, no code fence).\n\n"
        "```python\n" + code + "\n```"
    )
    return [{"role": "user", "content": user}], code


def boot_ci(vals, rng, n=5000):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return None, None, None
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=os.path.join(REPO_ROOT, "results", "naturalistic_scenarios.json"))
    ap.add_argument("--model", default=dp.DEFAULT_MODEL)
    ap.add_argument("--modes", default="nothink,think")
    ap.add_argument("--max-calls", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "deepseek_naturalistic"))
    ap.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".cache", "deepseek"))
    args = ap.parse_args()

    data = json.load(open(args.scenarios, encoding="utf-8"))
    scenarios = [s for s in data["scenarios"] if s.get("scenario_id")]  # already validated upstream
    modes = [m for m in args.modes.split(",") if m]
    client = dp.DeepSeekClient(model=args.model, cache_dir=args.cache_dir, max_calls=args.max_calls)

    records = []
    n = 0
    for s in scenarios:
        for cond, import_line, alias in CONDITIONS:
            messages, code = gen_messages(import_line, s["setup"], s["use_var"], alias)
            for mode in modes:
                nothink = mode == "nothink"
                try:
                    res = client.chat(messages, thinking=(False if nothink else True),
                                      max_tokens=(24 if nothink else 6000), temperature=0.0)
                except dp.DeepSeekError as e:
                    print(f"  STOP: {e}"); _save(records, args, modes, client); return 1
                attr = dp.parse_attribute(res.content, alias=alias)
                rb = resolves_on(attr, "pandas") if attr else {"exists": None}
                records.append({
                    "model": args.model, "scenario_id": s["scenario_id"], "domain": s.get("domain", ""),
                    "pandas_op": s["pandas_op"], "condition": cond, "alias": alias, "mode": mode,
                    "gen_attr": attr, "gen_text": res.content.strip()[:80],
                    "broken_on_pandas": (rb.get("exists") is False) if attr else None,
                    "wrote_target_op": (attr == s["pandas_op"]) if attr else False,
                    "finish_reason": res.finish_reason, "prompt_sha256": sha256_text(code),
                    "from_cache": res.cached,
                })
                n += 1
                if n % 20 == 0:
                    print(f"  {n} calls | live={client.live_calls} cache={client.cache_hits}")
    _save(records, args, modes, client)
    return 0


def _save(records, args, modes, client):
    with open(args.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(20260618)
    summary = {"model": args.model, "modes": modes, "fingerprint": client.fingerprint(),
               "n_calls_live": client.live_calls, "per_mode": {}, "timestamp_utc": utc_now_iso()}
    for mode in modes:
        mr = [r for r in records if r["mode"] == mode]
        block = {}
        for c in ("correct", "no_prior", "swapped"):
            rs = [r for r in mr if r["condition"] == c]
            br, blo, bhi = boot_ci([r["broken_on_pandas"] for r in rs if r["broken_on_pandas"] is not None], rng)
            wt, wlo, whi = boot_ci([1.0 if r["wrote_target_op"] else 0.0 for r in rs], rng)
            trunc = sum(1 for r in rs if r["finish_reason"] == "length")
            block[c] = {"n": len(rs), "broken_on_pandas": br, "broken_ci": [blo, bhi],
                        "wrote_target_op": wt, "wrote_target_ci": [wlo, whi], "truncated": trunc}
        summary["per_mode"][mode] = block
    summary["environment"] = environment_fingerprint()
    with open(args.out + "_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.relpath(args.out + '_results.json', REPO_ROOT)}")
    for mode in modes:
        b = summary["per_mode"][mode]
        print(f"\n[{args.model} / {mode}]  (broken-on-pandas / wrote-correct-op)")
        for c in ("correct", "no_prior", "swapped"):
            x = b[c]
            print(f"  {c:9} broken={x['broken_on_pandas']} wrote_target={x['wrote_target_op']} "
                  f"n={x['n']} trunc={x['truncated']}")


if __name__ == "__main__":
    raise SystemExit(main())
