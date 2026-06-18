#!/usr/bin/env python
"""Measure canonical alias-library frequencies from a public Python corpus.

This replaces the ordinal prior-strength tiers (very common / common / rare) with a measured
dose variable. For each alias ``a`` we count ``import MODULE as a`` over a streamed sample of the
codeparrot Python corpus and report, per canonical library ``A``:

  conv_count      count(import A as a)                      raw exposure to the convention
  alias_total     sum over modules M of count(import M as a)
  p_canonical     conv_count / alias_total                  dominance of A given the alias
  log_odds        log(conv_count + a) - log(others + a)     prior strength (the dose variable)

It also counts ``alias.method`` use-site occurrences for the canonical aliases, which the
matched-candidate robustness check (token length / method frequency) uses. Output is written to
results/corpus_freq.json. The corpus is read by streaming, so nothing large is downloaded.

  python scripts/corpus_freq.py --n-files 150000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from alias_inertia.lexicons import CANONICAL_ALIASES, IMPORT_NAMES  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_IMPORT_AS = re.compile(r"^[ \t]*import[ \t]+([\w.]+)[ \t]+as[ \t]+(\w+)", re.M)
# Plain ``import MODULE`` (no alias): for aliases that equal the module basename (e.g. torch),
# the plain import also establishes the name->library binding and must count toward the prior.
_IMPORT_PLAIN = re.compile(r"^[ \t]*import[ \t]+([\w.]+)[ \t]*(?:#.*)?$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="codeparrot/codeparrot-clean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="content")
    ap.add_argument("--n-files", type=int, default=150000)
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing for log-odds")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "corpus_freq.json"))
    args = ap.parse_args()

    from datasets import load_dataset

    # aliases whose method use-sites we additionally tally (the canonical aliases we score)
    method_aliases = sorted(set(CANONICAL_ALIASES.values()))
    method_re = re.compile(r"\b(" + "|".join(re.escape(a) for a in method_aliases) + r")\.([A-Za-z_]\w*)")

    import_counts: Counter = Counter()       # (module, alias) -> n
    plain_import_counts: Counter = Counter()  # module -> n  (``import MODULE`` with no alias)
    alias_method_counts: dict[str, Counter] = {a: Counter() for a in method_aliases}

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    n = 0
    for ex in ds:
        if n >= args.n_files:
            break
        n += 1
        code = ex.get(args.field, "") or ""
        for m in _IMPORT_AS.finditer(code):
            import_counts[(m.group(1), m.group(2))] += 1
        for m in _IMPORT_PLAIN.finditer(code):
            plain_import_counts[m.group(1)] += 1
        for m in method_re.finditer(code):
            alias_method_counts[m.group(1)][m.group(2)] += 1
        if n % 20000 == 0:
            print(f"  {n} files | {sum(import_counts.values())} import-as | "
                  f"{len(import_counts)} distinct (module,alias)")

    # Per-alias breakdown over the modules bound to it.
    per_alias: dict[str, dict] = {}
    aliases = {a for (_, a) in import_counts}
    for a in aliases:
        mods = Counter()
        for (mod, al), c in import_counts.items():
            if al == a:
                mods[mod] += c
        per_alias[a] = {
            "alias_total": int(sum(mods.values())),
            "top_modules": [[m, int(c)] for m, c in mods.most_common(6)],
        }

    # Canonical-convention dose for each alias we score (alias -> its conventional library).
    import math

    canonical_dose: dict[str, dict] = {}
    for lib, alias in CANONICAL_ALIASES.items():
        module = IMPORT_NAMES.get(lib, lib)
        conv = int(import_counts.get((module, alias), 0))
        plain = 0
        # An alias equal to the module basename (e.g. torch) is also established by plain import.
        if alias == module.split(".")[-1]:
            plain = int(plain_import_counts.get(module, 0))
        conv_eff = conv + plain
        total = int(per_alias.get(alias, {}).get("alias_total", 0)) + plain
        others = total - conv_eff
        canonical_dose[alias] = {
            "library": lib,
            "module": module,
            "conv_count_as": conv,
            "plain_import_count": plain,
            "conv_count": conv_eff,
            "alias_total": total,
            "p_canonical": (conv_eff / total) if total else None,
            "log_conv_count": math.log(conv_eff + args.alpha),
            "log_odds": math.log(conv_eff + args.alpha) - math.log(others + args.alpha),
        }

    out = {
        "dataset": args.dataset,
        "split": args.split,
        "n_files": n,
        "alpha": args.alpha,
        "total_import_as": int(sum(import_counts.values())),
        "distinct_module_alias": len(import_counts),
        "top_import_as": [[mod, al, int(c)] for (mod, al), c in import_counts.most_common(120)],
        "per_alias": per_alias,
        "canonical_dose": canonical_dose,
        "method_use_site_counts": {
            a: [[meth, int(c)] for meth, c in alias_method_counts[a].most_common(40)]
            for a in method_aliases
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\ndone: {n} files, {out['total_import_as']} import-as statements")
    print("canonical-convention dose (alias -> library):")
    for a, d in sorted(canonical_dose.items(), key=lambda kv: -kv[1]["conv_count"]):
        print(f"  {a:6} -> {d['library']:18} conv={d['conv_count']:7d} "
              f"P(lib|alias)={d['p_canonical']} log_odds={d['log_odds']:.2f}")
    print(f"saved -> {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
