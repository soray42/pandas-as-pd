"""Programmatic minimal-pair stimulus generator.

Produces a prompt that ENDS exactly at ``{alias}.`` (nothing after) plus a metadata row.
The three binding conditions and the alias-safe, token-length-controlled neutral filler are
constructed here. Distance is measured in **tokens, in context** (the marginal tokens the
filler adds to the prompt under the model's own tokenizer).

Conditions (for swap pair (prior_lib, other_lib), treatment alias = prior_lib's canonical alias):
  conventional : ``import {prior_lib} as {alias}``   bound == prior  (CEILING / positive control)
  swapped      : ``import {other_lib} as {alias}``   alias bound to the OTHER lib  (TREATMENT)
  no_prior     : ``import {other_lib} as {rand}``    non-canonical alias, same lib as swapped
                                                     (GENERIC long-context-rot BASELINE)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .determinism import sha256_text, stable_hash
from .lexicons import CANONICAL_ALIASES, IMPORT_NAMES, SwapPair

STIMULI_VERSION = "1.2"

CONDITIONS = ("conventional", "swapped", "no_prior")
TEMPLATES = ("t1", "t2")

_TEMPLATE_HEADERS = {
    "t1": "",  # bare script
    "t2": "# data processing\n",  # header framing
}

# Identifiers used in the filler. NONE may contain a library-alias substring; that is
# enforced by ``_assert_identifier_safe`` at import time. Filler must never "mention" the
# alias, so we keep the vocabulary deliberately bland and alias-free.
_FILLER_VARS = (
    "acc",
    "total",
    "count",
    "idx",
    "val",
    "res",
    "tmp",
    "item",
    "buf",
    "cur",
    "delta",
    "gain",
    "limit",
    "offset",
    "ratio",
    "score",
    "depth",
    "width",
    "height",
    "weight",
    "value",
    "holder",
    "bucket",
    "helper",
    "factor",
    "margin",
    "cursor",
    "amount",
    "balance",
)

# Substrings that must never appear inside a filler identifier (all known aliases + module
# names). The configured non-canonical alias is checked separately at generation time.
_FORBIDDEN_SUBSTRINGS = tuple(
    set(CANONICAL_ALIASES.values())
    | set(IMPORT_NAMES.keys())
    | {v for v in IMPORT_NAMES.values()}
    | {"pyplot", "matplotlib"}
)


def _assert_identifier_safe() -> None:
    for name in _FILLER_VARS:
        for bad in _FORBIDDEN_SUBSTRINGS:
            assert bad not in name, f"filler identifier {name!r} contains forbidden substring {bad!r}"


_assert_identifier_safe()

# Pool of non-canonical "nonce" aliases for the no-prior control. None is a canonical alias and
# none contains a library substring, so none carries a convention. Averaging the control over
# several nonce aliases (rather than a single "zz") shows it is not an artifact of one token.
# "zz" stays first so the single-alias default is unchanged.
NONCE_ALIASES: tuple[str, ...] = ("zz", "qx", "vv")


def _assert_nonce_safe() -> None:
    canon = set(CANONICAL_ALIASES.values())
    for a in NONCE_ALIASES:
        assert a not in canon, f"nonce alias {a!r} is actually a canonical alias"
        for bad in _FORBIDDEN_SUBSTRINGS:
            assert bad not in a, f"nonce alias {a!r} contains forbidden substring {bad!r}"


_assert_nonce_safe()


@dataclass(frozen=True)
class Stimulus:
    prompt: str
    meta: dict = field(default_factory=dict)


def _rng_for(seed: int, *parts) -> random.Random:
    """Deterministic RNG keyed by the global seed and the stimulus coordinates."""
    return random.Random(int(stable_hash([seed, *parts], length=16), 16))


def _statement(rng: random.Random) -> str:
    """One neutral Python statement (1-3 lines) using only alias-free identifiers."""
    a, b, c, i, fn = (rng.choice(_FILLER_VARS) for _ in range(5))
    n1, n2 = rng.randint(1, 99), rng.randint(1, 9)
    kind = rng.randint(0, 5)
    if kind == 0:
        return f"{a} = {n1}"
    if kind == 1:
        return f"{a} = {b} + {n1} * {n2}"
    if kind == 2:
        return f"{a} = {a} - {n2}"
    if kind == 3:
        return f"def {fn}_{n1}({c}):\n    return {c} + {n2}"
    if kind == 4:
        return f"for {i} in range({n2}):\n    {a} = {a} + {i}"
    return f"if {b} > {n2}:\n    {a} = {b} - {n1}"


def _render(header: str, import_line: str, filler: str, alias: str) -> str:
    """Assemble the prompt, ending exactly at ``{alias}.`` (no trailing whitespace)."""
    if filler:
        body = f"{import_line}\n{filler}\n{alias}."
    else:
        body = f"{import_line}\n{alias}."
    return f"{header}{body}"


def _build_filler(
    *,
    header: str,
    import_line: str,
    alias: str,
    depth_tokens: int,
    count_tokens: Callable[[str], int],
    rng: random.Random,
    tolerance: int,
) -> tuple[str, int]:
    """Grow neutral filler until its *in-context* token contribution reaches ``depth_tokens``.

    Returns (filler_text, depth_actual). ``depth_actual`` is the number of tokens the filler
    adds to the prompt as tokenised by the model - i.e. distance in tokens, in context.

    Efficient for long depths (8k/32k): a self-correcting batch add/remove. Each iteration
    measures the current in-context delta, then adds (or removes) a *calculated* number of
    statements based on a running tokens-per-statement estimate. Converges in a handful of
    tokenisations regardless of depth - no single huge overshoot, and no O(n) one-by-one
    trimming (which made 32k pathologically slow).
    """
    if depth_tokens <= 0:
        return "", 0

    base_tokens = count_tokens(_render(header, import_line, "", alias))
    lines: list[str] = []

    def measure() -> int:
        return count_tokens(_render(header, import_line, "\n".join(lines), alias)) - base_tokens

    est = 10.0  # tokens per statement; self-corrects after the first measurement
    delta = 0
    best_lines, best_err = [], float("inf")
    for _ in range(200):  # converges in ~3-8 iterations; cap is a safety net
        err = abs(delta - depth_tokens)
        if err < best_err:
            best_err, best_lines = err, list(lines)
        if depth_tokens - tolerance <= delta <= depth_tokens + tolerance:
            break
        gap = depth_tokens - delta
        n = max(1, int(round(abs(gap) / est)))
        if gap > 0:
            lines.extend(_statement(rng) for _ in range(n))
        else:
            del lines[len(lines) - min(n, len(lines)):]
        delta = measure()
        if lines:
            est = max(1.0, delta / len(lines))
    else:
        # Did not land in band within the cap (rare): use the closest configuration seen.
        lines = best_lines
        delta = measure()

    return "\n".join(lines), delta


def build_stimulus(
    *,
    pair: SwapPair,
    condition: str,
    depth_tokens: int,
    template_id: str,
    rep: int,
    count_tokens: Callable[[str], int],
    seed: int,
    non_canonical_alias: str = "zz",
    depth_tolerance_tokens: int = 24,
) -> Stimulus:
    """Build one stimulus (prompt + metadata) for the given coordinates."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if template_id not in TEMPLATES:
        raise ValueError(f"unknown template {template_id!r}")

    prior_lib = pair.prior_lib
    other_lib = pair.other_lib
    treatment_alias = pair.treatment_alias

    if condition == "conventional":
        import_lib, alias, bound_lib = prior_lib, treatment_alias, prior_lib
        alias_has_prior = True
    elif condition == "swapped":
        import_lib, alias, bound_lib = other_lib, treatment_alias, other_lib
        alias_has_prior = True
    else:  # no_prior
        import_lib, alias, bound_lib = other_lib, non_canonical_alias, other_lib
        alias_has_prior = False
        # A non-canonical alias must not itself be a canonical alias (else it has a prior).
        if non_canonical_alias in CANONICAL_ALIASES.values():
            raise ValueError(f"non_canonical_alias {non_canonical_alias!r} is actually canonical")

    header = _TEMPLATE_HEADERS[template_id]
    import_line = f"import {IMPORT_NAMES[import_lib]} as {alias}"

    rng = _rng_for(seed, pair.name, condition, alias, depth_tokens, template_id, rep)
    filler, depth_actual = _build_filler(
        header=header,
        import_line=import_line,
        alias=alias,
        depth_tokens=depth_tokens,
        count_tokens=count_tokens,
        rng=rng,
        tolerance=depth_tolerance_tokens,
    )

    prompt = _render(header, import_line, filler, alias)

    # The alias must never appear inside the filler (the binding is only "needed" at usage).
    if filler:
        assert alias not in filler, f"alias {alias!r} leaked into filler"

    # The prompt must end exactly at ``{alias}.``.
    assert prompt.endswith(f"{alias}."), "prompt does not end at the usage site"

    prompt_tokens_total = count_tokens(prompt)

    meta = {
        "stimuli_version": STIMULI_VERSION,
        "pair": pair.name,
        "prior_lib": prior_lib,  # the library whose canonical alias is the treatment alias
        "other_lib": other_lib,  # the library bound in swapped / no_prior
        "condition": condition,
        "alias": alias,
        "import_lib": import_lib,  # the library actually imported (== bound target)
        "import_line": import_line,
        "prior_target": prior_lib,  # "prior" lexicon for the metric (numpy, counterfactual in no_prior)
        "bound_target": bound_lib,  # "bound" lexicon (what correct tracking looks like)
        "prior_lexicon_lib": prior_lib,
        "bound_lexicon_lib": bound_lib,
        "alias_has_canonical_prior": alias_has_prior,
        "template_id": template_id,
        "rep": rep,
        "seed": seed,
        "depth_tokens_target": depth_tokens,
        "depth_tokens_actual": depth_actual,
        "prompt_tokens_total": prompt_tokens_total,
        "prompt_sha256": sha256_text(prompt),
        "filler_sha256": sha256_text(filler),
        "prompt": prompt,  # stored verbatim for exact reproducibility / inspection
    }
    meta["stimulus_id"] = stable_hash(
        [meta["stimuli_version"], pair.name, condition, alias, template_id, rep, depth_tokens, meta["prompt_sha256"]],
        length=16,
    )
    return Stimulus(prompt=prompt, meta=meta)


def reps_for_depth(depth_tokens: int, repetitions: int) -> int:
    """Number of repetitions for a depth bin.

    At depth 0 there is no filler to vary, so extra reps would produce identical prompts and
    artificially narrow CIs. We therefore use a single rep at
    depth 0; depths > 0 get genuine per-rep filler variation.
    """
    return 1 if int(depth_tokens) <= 0 else repetitions


def grid_plan_for_depth(depth, templates, repetitions, deep_threshold, deep_templates, deep_reps):
    """(templates, reps) to use at a given depth.

    depth 0  -> 1 rep, all templates (no filler to vary -> reps would be identical).
    deep     -> deep_templates / deep_reps (deep bins like 8k/32k are O(L^2) to score, so we
                thin them: enough to CHARACTERISE the gap-vs-distance shape, not full statistics).
    else     -> all templates, full reps (the short bins carry the dose/size statistics).
    """
    d = int(depth)
    if d <= 0:
        return list(templates), 1
    if deep_threshold is not None and d >= int(deep_threshold):
        return list(deep_templates or templates), int(deep_reps)
    return list(templates), int(repetitions)


def generate_grid(
    *,
    pair: SwapPair,
    conditions,
    depths_tokens,
    templates,
    repetitions: int,
    count_tokens: Callable[[str], int],
    seed: int,
    non_canonical_alias: str = "zz",
    nonce_aliases=None,
    depth_tolerance_tokens: int = 24,
    deep_threshold=None,
    deep_templates=None,
    deep_reps: int = 1,
):
    """Yield every stimulus in the grid (conditions x depths x templates x reps).

    Depth 0 uses a single rep (no filler to vary); deep bins (>= deep_threshold) are thinned to
    deep_templates/deep_reps to keep the expensive long-context scoring tractable.

    ``nonce_aliases`` (optional): when given, the no-prior condition is emitted once per nonce
    alias (each with its own filler), so the control averages over several non-canonical aliases
    rather than the single ``non_canonical_alias``. The other conditions are unaffected. Leaving
    it ``None`` reproduces the original single-alias grid exactly.
    """
    for condition in conditions:
        for depth in depths_tokens:
            tmpls, reps = grid_plan_for_depth(
                depth, templates, repetitions, deep_threshold, deep_templates, deep_reps
            )
            for template_id in tmpls:
                for rep in range(reps):
                    if condition == "no_prior" and nonce_aliases:
                        aliases = list(nonce_aliases)
                    else:
                        aliases = [non_canonical_alias]
                    for nc_alias in aliases:
                        yield build_stimulus(
                            pair=pair,
                            condition=condition,
                            depth_tokens=int(depth),
                            template_id=template_id,
                            rep=rep,
                            count_tokens=count_tokens,
                            seed=seed,
                            non_canonical_alias=nc_alias,
                            depth_tolerance_tokens=depth_tolerance_tokens,
                        )
