"""Behavioral generation arm: greedy-decode after ``{alias}.`` and classify the completion as
prior-style / bound-style / other.

This is purely behavioral (it looks at what the model *writes*, never inside the model), so it
is allowed in the workshop scope - mechanistic interpretability stays reserved for the main conf.

Classification is driven by the same discriminative lexicons used for scoring: the first
attribute the model accesses on the alias (the identifier immediately after ``{alias}.``) is
compared against the prior-library and bound-library member names.
"""

from __future__ import annotations

import re
from typing import Sequence

from .lexicons import normalize_member

GEN_VERSION = "1.0"

_IDENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


def extract_first_attribute(generation: str) -> str | None:
    """The attribute accessed on the alias = the leading identifier of the generation.

    The prompt ends exactly at ``{alias}.`` so the generation begins with the attribute name
    (e.g. ``"array([1,2,3])"`` -> ``"array"``; ``"nn.Linear(...)"`` -> ``"nn"``).
    """
    if not generation:
        return None
    m = _IDENT_RE.match(generation)
    return m.group(1) if m else None


def _member_names(lexicon: Sequence[str]) -> set[str]:
    return {normalize_member(m) for m in lexicon}


def classify_generation(
    generation: str,
    *,
    prior_lib: str,
    bound_lib: str,
    lexicons: dict[str, Sequence[str]],
) -> dict:
    """Classify a single greedy generation.

    Returns {attribute, klass} where klass in {prior_style, bound_style, other, empty}. When
    prior_lib == bound_lib (conventional condition) a match is reported as ``bound_style``
    (the correct/expected library), and ``prior_match``/``bound_match`` flags are also recorded.
    """
    attr = extract_first_attribute(generation)
    prior_names = _member_names(lexicons.get(prior_lib, []))
    bound_names = _member_names(lexicons.get(bound_lib, []))

    prior_match = attr in prior_names if attr else False
    bound_match = attr in bound_names if attr else False

    if attr is None:
        klass = "empty"
    elif bound_match and prior_lib != bound_lib:
        klass = "bound_style"
    elif prior_match and prior_lib != bound_lib:
        klass = "prior_style"
    elif bound_match:  # conventional: prior == bound, a match is the correct library
        klass = "bound_style"
    else:
        klass = "other"

    return {
        "attribute": attr,
        "klass": klass,
        "prior_match": bool(prior_match),
        "bound_match": bool(bound_match),
    }
