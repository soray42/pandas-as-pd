"""Backend contract for teacher-forced continuation scoring.

All backends return a :class:`ScoreResult` from one interface:

    score_continuation(prompt, continuation) -> ScoreResult   # summed teacher-forced logP
    score_many(prompt, [continuations])      -> [ScoreResult]  # prompt-forward reuse

``logprob`` is the summed log-probability of the continuation tokens given the prompt,
computed by teacher forcing (the continuation is supplied, not generated).

**Boundary handling (critical correctness point).** Method names are multi-token and the
prompt ends in ``.`` - under BPE the dot frequently merges rightward into the first
continuation token (``np.`` + ``array(`` -> ``...np`` + ``.array`` + ``(``). We therefore
never split by character offset. Instead we tokenise the prompt and the full
``prompt+continuation`` and score from the first token where they diverge (the longest
common *token* prefix). ``boundary_merge`` records whether that divergence occurred before
the prompt's last token (i.e. a merge happened); the score still measures
``logP(continuation completion | prompt)`` consistently across all continuations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ScoreResult:
    logprob: float
    n_tokens: int
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    boundary_merge: bool

    def to_json(self) -> dict:
        return {
            "logprob": self.logprob,
            "n_tokens": self.n_tokens,
            "token_ids": list(self.token_ids),
            "token_logprobs": list(self.token_logprobs),
            "boundary_merge": self.boundary_merge,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ScoreResult":
        return cls(
            logprob=float(d["logprob"]),
            n_tokens=int(d["n_tokens"]),
            token_ids=tuple(int(t) for t in d["token_ids"]),
            token_logprobs=tuple(float(x) for x in d["token_logprobs"]),
            boundary_merge=bool(d["boundary_merge"]),
        )


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    k = 0
    n = min(len(a), len(b))
    while k < n and a[k] == b[k]:
        k += 1
    return k


class Backend(Protocol):
    """Structural interface every scoring backend implements."""

    id: str

    def fingerprint(self) -> dict:
        """Full, human-readable provenance for the manifest (may include post-hoc fields)."""
        ...

    def cache_fingerprint(self) -> dict:
        """The subset of settings that change the numeric score, for the cache key.

        must include every parameter that alters a returned log-prob (model identity, dtype,
        tokenisation flags, llama.cpp n_threads/n_batch/seed/n_ctx, ...). The disk cache keys on
        a hash of this dict, so two configs that would score differently never share an entry.
        """
        ...

    def tokenize(self, text: str) -> list[int]: ...

    def count_tokens(self, text: str) -> int: ...

    def score_continuation(self, prompt: str, continuation: str) -> ScoreResult: ...

    def score_many(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]: ...

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        """Greedy-decode up to ``max_new_tokens`` after the prompt; return the generated text
        only (not the prompt). Used by the behavioral generation arm."""
        ...
