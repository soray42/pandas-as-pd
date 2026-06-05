#!/usr/bin/env python
"""Generate and preview the stimulus grid for one swap pair, without scoring.

Useful to inspect prompts and confirm the in-context token depths land on target before a
(slow) scoring run. Depth is measured with the chosen model's own tokenizer.

  python scripts/gen_stimuli.py --config configs/full.yaml --pair numpy__pandas --show 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.determinism import set_determinism  # noqa: E402
from alias_inertia.lexicons import get_pair  # noqa: E402
from alias_inertia.stimuli import generate_grid  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--model-label", default=None, help="model in config['models'] to tokenize with (default: first)")
    ap.add_argument("--pair", default=None, help="swap-pair name, e.g. numpy__pandas (default: first in pairs_all)")
    ap.add_argument("--show", type=int, default=2, help="print this many example prompts")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "preview_stimuli.jsonl"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    set_determinism(int(cfg["seed"]))

    models = cfg["models"]
    mcfg = next((m for m in models if m["label"] == args.model_label), models[0])
    backend = build_backend(mcfg["backend"], mcfg[mcfg["backend"]])
    print(f"[gen_stimuli] tokenizer = {backend.id} (model '{mcfg['label']}')")

    pair = get_pair(args.pair or cfg["pairs_all"][0])
    deep = cfg.get("deep_bins") or {}
    stimuli = list(generate_grid(
        pair=pair, conditions=cfg["conditions"], depths_tokens=cfg["depths_tokens"],
        templates=cfg["templates"], repetitions=int(cfg["repetitions"]),
        count_tokens=backend.count_tokens, seed=int(cfg["seed"]),
        non_canonical_alias=cfg.get("non_canonical_alias", "zz"),
        depth_tolerance_tokens=int(cfg.get("depth_tolerance_tokens", 48)),
        deep_threshold=deep.get("threshold"), deep_templates=deep.get("templates"),
        deep_reps=int(deep.get("reps", 1)),
    ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for s in stimuli:
            fh.write(json.dumps({k: v for k, v in s.meta.items() if k != "prompt"}, ensure_ascii=False) + "\n")
    print(f"[gen_stimuli] pair={pair.name} ({pair.tier}); wrote {len(stimuli)} stimuli -> {args.out}")

    print("\n[gen_stimuli] depth target vs actual (in-context tokens):")
    for d in cfg["depths_tokens"]:
        acts = [s.meta["depth_tokens_actual"] for s in stimuli if s.meta["depth_tokens_target"] == d]
        if acts:
            print(f"   target {d:>6}: actual min={min(acts)} max={max(acts)} (n={len(acts)})")

    print("\n[gen_stimuli] examples:")
    for s in stimuli[: args.show]:
        m = s.meta
        print("-" * 70)
        print(f"  {m['condition']} | depth={m['depth_tokens_target']} | {m['template_id']} | "
              f"alias={m['alias']} | bound={m['bound_target']} | prior={m['prior_target']}")
        for line in s.prompt.splitlines():
            print("    " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
