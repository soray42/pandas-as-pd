#!/usr/bin/env python
"""Quick scorer sanity check for one model in a config, before a batch run.

Sanity checks:
  * a plausible continuation must score higher (less negative) than an implausible one;
  * on a swapped prompt (import pandas as np), prints the numpy vs pandas log-sum-exp so the
    prior-pull wiring can be eyeballed.

  python scripts/smoke_backend.py --config configs/full.yaml --model-label qwen2.5-0.5b-base
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from alias_inertia.backends import build_backend  # noqa: E402
from alias_inertia.lexicons import LEXICONS  # noqa: E402
from alias_inertia.metrics import logsumexp  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "full.yaml"))
    ap.add_argument("--model-label", default=None, help="model in config['models'] (default: first)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    models = cfg["models"]
    mcfg = next((m for m in models if m["label"] == args.model_label), models[0])

    print(f"[smoke] building model '{mcfg['label']}' ({mcfg['backend']}) ...")
    backend = build_backend(mcfg["backend"], mcfg[mcfg["backend"]])
    print(f"[smoke] backend id = {backend.id}")

    good = backend.score_continuation("The capital of France is", " Paris").logprob
    bad = backend.score_continuation("The capital of France is", " banana").logprob
    print(f"[smoke] logP(' Paris')={good:+.3f}  logP(' banana')={bad:+.3f}  ordered={'OK' if good > bad else 'FAIL'}")

    prompt = "import pandas as np\nnp."
    lse_numpy = logsumexp([r.logprob for r in backend.score_many(prompt, LEXICONS["numpy"])])
    lse_pandas = logsumexp([r.logprob for r in backend.score_many(prompt, LEXICONS["pandas"])])
    print(f"[smoke] swapped 'import pandas as np': L(numpy)={lse_numpy:+.3f}  L(pandas)={lse_pandas:+.3f}  "
          f"prior_pull={lse_numpy - lse_pandas:+.3f}")

    ok = good > bad
    print(f"\n[smoke] RESULT: {'PASS' if ok else 'FAIL'} (plausibility ordering)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
