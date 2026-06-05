"""Per-library *discriminative* continuation strings and the swap-pair definitions.

A "continuation" is the text that appears immediately after ``{alias}.`` at the usage
site, e.g. for ``np.`` the continuation ``"array("`` makes ``np.array(``. We score the
teacher-forced log-prob of the whole multi-token continuation (see ``scoring.py``), so
multi-token method names are handled correctly.

Design rules (locked):
  * Each list contains only tokens that are *discriminative* for that library - names a
    competent reader would attribute to numpy but not pandas, etc.
  * Tokens shared across libraries (``sum``, ``mean``, ``T``, ``max``, ``reshape`` ...)
    are EXCLUDED - they cannot separate prior from bound target. See ``SHARED_BLOCKLIST``.
  * Continuations that are callables end with ``"("`` so the model commits to "this is a
    call to <name>", which sharpens the discriminative signal.
  * Every canonical library has its conventional alias in ``CANONICAL_ALIASES``.

This is the single place swap pairs are defined; the design scales by adding entries here,
with no code change elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

LEXICON_VERSION = "1.1"

# --- Canonical import aliases (prior target -> its near-universal alias) -----------------
CANONICAL_ALIASES: dict[str, str] = {
    "numpy": "np",
    "pandas": "pd",
    "torch": "torch",
    "sklearn": "sk",
    "xgboost": "xgb",
    "matplotlib.pyplot": "plt",
}

# Ordinal prior-strength tier per canonical alias (the DOSE axis for the full run). This is a
# coarse, defensible ordering of how near-universal each alias is in Python corpora; rigorous
# corpus-frequency estimation is deliberately reserved for the main conf (SCOPE §6/§8).
ALIAS_PRIOR_TIER: dict[str, str] = {
    "np": "very_common",
    "pd": "very_common",
    "torch": "common",
    "plt": "common",
    "sk": "rare",
    "xgb": "rare",
}
# Numeric rank for plotting / ordinal correlation (higher = stronger prior).
TIER_RANK: dict[str, int] = {"very_common": 3, "common": 2, "rare": 1}

# How each library is written in an ``import X as alias`` statement (the module path).
IMPORT_NAMES: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "matplotlib.pyplot": "matplotlib.pyplot",
}

# --- Discriminative continuation sets ----------------------------------------------------
LEXICONS: dict[str, list[str]] = {
    "numpy": ["array(", "arange(", "zeros(", "linspace(", "dot(", "ndarray"],
    "pandas": ["DataFrame(", "read_csv(", "concat(", "Series(", "merge(", "groupby("],
    "torch": ["tensor(", "randn(", "from_numpy(", "nn.", "cuda.", "autograd."],
    "sklearn": [
        "linear_model.",
        "datasets.",
        "preprocessing.",
        "model_selection.",
        "cluster.",
        "ensemble.",
    ],
    "xgboost": ["XGBClassifier(", "XGBRegressor(", "DMatrix(", "Booster(", "train(", "cv("],
    "matplotlib.pyplot": ["plot(", "scatter(", "figure(", "xlabel(", "subplots(", "hist("],
}

# Names that are NOT discriminative (shared across numeric/data libs). No lexicon entry may
# collide with these; ``verify_lexicons`` enforces it.
SHARED_BLOCKLIST: frozenset[str] = frozenset(
    {
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "abs",
        "T",
        "shape",
        "reshape",
        "astype",
        "copy",
        "values",
        "argmax",
        "argmin",
        "all",
        "any",
        "cumsum",
        "prod",
        "round",
        "sort",
        "transpose",
    }
)


@dataclass(frozen=True)
class SwapPair:
    """A pair of canonical libraries used to build the three binding conditions.

    ``prior_lib`` owns the *treatment alias* (its canonical alias). ``other_lib`` is the
    real library bound to that alias in the swapped / no-prior conditions.

    Example (numpy, pandas):
      conventional  -> ``import numpy as np``   (alias np, bound=numpy=prior)
      swapped       -> ``import pandas as np``  (alias np, bound=pandas, prior=numpy)
      no_prior      -> ``import pandas as zz``  (alias zz, bound=pandas, no canonical prior)
    """

    prior_lib: str
    other_lib: str

    @property
    def treatment_alias(self) -> str:
        return CANONICAL_ALIASES[self.prior_lib]

    @property
    def name(self) -> str:
        return f"{self.prior_lib}__{self.other_lib}"

    @property
    def tier(self) -> str:
        """Prior-strength tier of the treatment alias (the dose level for this pair)."""
        return ALIAS_PRIOR_TIER[self.treatment_alias]

    @property
    def tier_rank(self) -> int:
        return TIER_RANK[self.tier]


# Full-run dose axis: swap pairs spanning the prior-strength range, two aliases per tier, so the
# DiD gap can be read as a function of prior strength. The treatment alias (prior_lib's canonical
# alias) is what carries the prior; other_lib is the real library bound to it in swapped/no_prior.
SWAP_PAIRS: list[SwapPair] = [
    SwapPair("numpy", "pandas"),            # np  - very_common
    SwapPair("pandas", "numpy"),            # pd  - very_common
    SwapPair("torch", "sklearn"),           # torch - common
    SwapPair("matplotlib.pyplot", "pandas"),  # plt - common
    SwapPair("sklearn", "xgboost"),         # sk  - rare
    SwapPair("xgboost", "sklearn"),         # xgb - rare
]


def normalize_member(member: str) -> str:
    """Strip the call/attr suffix to get the bare attribute name (for blocklist checks)."""
    return member.rstrip("(").rstrip(".")


def verify_lexicons() -> None:
    """Self-consistency checks (also exercised by the test-suite).

    Raises AssertionError on any violation: missing aliases, blocklisted (shared) tokens,
    or cross-library collisions that would make a continuation non-discriminative.
    """
    # 1. Every lexicon library has a canonical alias and an import name.
    for lib in LEXICONS:
        assert lib in CANONICAL_ALIASES, f"missing canonical alias for {lib}"
        assert lib in IMPORT_NAMES, f"missing import name for {lib}"

    # 2. No lexicon entry collides with the shared (non-discriminative) blocklist.
    for lib, members in LEXICONS.items():
        for m in members:
            bare = normalize_member(m)
            assert bare not in SHARED_BLOCKLIST, f"{lib}.{m!r} is in SHARED_BLOCKLIST"

    # 3. Continuations are disjoint across libraries (a continuation must map to one lib).
    seen: dict[str, str] = {}
    for lib, members in LEXICONS.items():
        for m in members:
            assert m not in seen, f"continuation {m!r} shared by {seen.get(m)} and {lib}"
            seen[m] = lib

    # 4. Swap pairs reference known libraries.
    for pair in SWAP_PAIRS:
        assert pair.prior_lib in LEXICONS, f"unknown prior_lib {pair.prior_lib}"
        assert pair.other_lib in LEXICONS, f"unknown other_lib {pair.other_lib}"


def get_pair(name_or_index) -> SwapPair:
    """Resolve a swap pair by ``[prior, other]`` list, ``"prior__other"`` name, or index."""
    if isinstance(name_or_index, int):
        return SWAP_PAIRS[name_or_index]
    if isinstance(name_or_index, (list, tuple)) and len(name_or_index) == 2:
        return SwapPair(str(name_or_index[0]), str(name_or_index[1]))
    if isinstance(name_or_index, str):
        for p in SWAP_PAIRS:
            if p.name == name_or_index:
                return p
    raise KeyError(f"unknown swap pair: {name_or_index!r}")


# Validate at import time so a malformed edit fails fast.
verify_lexicons()
