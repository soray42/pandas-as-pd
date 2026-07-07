"""Proxy lexicon and proxy_pull metric for the mech arm.

ProxyLexicon: first-token-discriminative subsets of the full LEXICONS for a pair,
enabling O(1) logit lookup rather than full continuation scoring. Used by logit-lens
trajectories and direct logit attribution (M1/M3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProxyLexicon:
    """First-token-id discriminative sets for prior vs bound library.

    prior_ids / bound_ids: first-token ids surviving the collision filter.
    dropped: continuation strings whose first token collided across the two sets
             or appeared twice within the same set (intra-lib dup).
    sha256: hash of the surviving sets for provenance.
    """

    prior_lib: str
    bound_lib: str
    prior_ids: list[int]
    bound_ids: list[int]
    prior_strings: list[str]
    bound_strings: list[str]
    dropped: list[str]
    sha256: str


def build_proxy_lexicon(
    tokenizer,
    prior_lib: str = "numpy",
    bound_lib: str = "pandas",
) -> ProxyLexicon:
    """Build the proxy lexicon for one swap pair (prior_lib, bound_lib).

    Continuations are taken from alias_inertia.lexicons.LEXICONS. A continuation
    survives iff its first token id is unique across both sets AND not a duplicate
    within its own set (first-token collision would make the first-token logit
    non-discriminative). Order of LEXICONS lists is preserved for surviving entries;
    alphabetical order among dropped strings.
    """
    from alias_inertia.determinism import stable_hash  # noqa: PLC0415
    from alias_inertia.lexicons import LEXICONS  # noqa: PLC0415

    prior_conts: list[str] = LEXICONS[prior_lib]
    bound_conts: list[str] = LEXICONS[bound_lib]

    def first_token(s: str) -> int:
        return tokenizer.encode(s, add_special_tokens=False)[0]

    prior_ft = {c: first_token(c) for c in prior_conts}
    bound_ft = {c: first_token(c) for c in bound_conts}

    # First-token ids that appear in both sets -> cross-lib collision, drop from both.
    collision_fts: set[int] = set(prior_ft.values()) & set(bound_ft.values())

    dropped: list[str] = []

    # Filter prior set: drop cross-lib collisions and intra-lib first-token duplicates.
    prior_ids: list[int] = []
    prior_strings: list[str] = []
    seen_prior: set[int] = set()
    for c in prior_conts:
        ft = prior_ft[c]
        if ft in collision_fts or ft in seen_prior:
            dropped.append(c)
        else:
            prior_ids.append(ft)
            prior_strings.append(c)
            seen_prior.add(ft)

    # Filter bound set likewise.
    bound_ids: list[int] = []
    bound_strings: list[str] = []
    seen_bound: set[int] = set()
    for c in bound_conts:
        ft = bound_ft[c]
        if ft in collision_fts or ft in seen_bound:
            dropped.append(c)
        else:
            bound_ids.append(ft)
            bound_strings.append(c)
            seen_bound.add(ft)

    dropped.sort()

    sha = stable_hash(
        {
            "prior_lib": prior_lib,
            "bound_lib": bound_lib,
            "prior_ids": sorted(prior_ids),
            "bound_ids": sorted(bound_ids),
        }
    )

    return ProxyLexicon(
        prior_lib=prior_lib,
        bound_lib=bound_lib,
        prior_ids=prior_ids,
        bound_ids=bound_ids,
        prior_strings=prior_strings,
        bound_strings=bound_strings,
        dropped=dropped,
        sha256=sha,
    )


def proxy_pull(logits: torch.Tensor, lex: ProxyLexicon) -> torch.Tensor:
    """logsumexp(logits[prior_ids]) - logsumexp(logits[bound_ids]).

    Operates on raw logits (not log-softmax); the additive shift from log-partition
    cancels in the subtraction, making the result shift-invariant.

    Args:
        logits: [..., d_vocab] raw logits tensor.
        lex: ProxyLexicon with prior_ids and bound_ids.

    Returns:
        Scalar or [...] tensor of proxy pull values.
    """
    prior_ids = torch.tensor(lex.prior_ids, dtype=torch.long, device=logits.device)
    bound_ids = torch.tensor(lex.bound_ids, dtype=torch.long, device=logits.device)
    prior_logits = logits[..., prior_ids]   # [..., n_prior]
    bound_logits = logits[..., bound_ids]   # [..., n_bound]
    lse_prior = torch.logsumexp(prior_logits, dim=-1)
    lse_bound = torch.logsumexp(bound_logits, dim=-1)
    return lse_prior - lse_bound
