"""The prior-pull metric.

At the prefix ending ``{alias}.`` we score a discriminative continuation set per library
(teacher-forced summed log-prob, robust to multi-token names) and aggregate with log-sum-exp:

    L(lib)      = logsumexp_{c in lexicon[lib]}  logP(c | prompt)
    prior_pull  = L(prior_lib) - L(bound_lib)

``prior_pull > 0`` => the model puts mass on the *prior* target's methods (corpus inertia);
``< 0`` => it tracks the *bound* target. For the conventional condition prior_lib == bound_lib,
so ``prior_pull`` is degenerate (~0); report ``bound_mass = L(bound_lib)`` as the ceiling
check instead (is mass on the right methods at all?).
"""

from __future__ import annotations

import math
from typing import Sequence


def logsumexp(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    m = max(vals)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in vals))


def score_lexicon(scorer, prompt: str, continuations: Sequence[str]) -> dict:
    """Score one library's continuation set against the prompt; return logsumexp + details."""
    results = scorer.score_many(prompt, list(continuations))
    per = {}
    logps = []
    boundary_any = False
    for cont, res in zip(continuations, results):
        per[cont] = {
            "logprob": res.logprob,
            "n_tokens": res.n_tokens,
            "boundary_merge": res.boundary_merge,
            "token_logprobs": list(res.token_logprobs),
        }
        logps.append(res.logprob)
        boundary_any = boundary_any or res.boundary_merge
    return {
        "logsumexp": logsumexp(logps),
        "mean_logprob": sum(logps) / len(logps) if logps else float("-inf"),
        "per_continuation": per,
        "boundary_merge_any": boundary_any,
    }


def compute_metric_row(
    scorer,
    prompt: str,
    *,
    prior_lib: str,
    bound_lib: str,
    lexicons: dict[str, Sequence[str]],
) -> dict:
    """Score every library in ``lexicons`` and derive prior_pull / bound_mass for this prompt.

    ``lexicons`` should contain at least ``prior_lib`` and ``bound_lib`` (typically both libs
    of the swap pair). All libraries are scored and stored so the analysis can recompute any
    contrast; prior_pull and bound_mass are the headline numbers.
    """
    lex_scores = {lib: score_lexicon(scorer, prompt, conts) for lib, conts in lexicons.items()}
    lse = {lib: s["logsumexp"] for lib, s in lex_scores.items()}

    prior_pull = lse[prior_lib] - lse[bound_lib]
    bound_mass = lse[bound_lib]

    return {
        "prior_pull": prior_pull,
        "bound_mass": bound_mass,
        "logsumexp_by_lib": lse,
        "lexicon_scores": lex_scores,
        "boundary_merge_any": any(s["boundary_merge_any"] for s in lex_scores.values()),
    }
