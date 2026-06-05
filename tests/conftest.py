"""Shared pytest fixtures.

Model-free tests (lexicons, stimuli, metrics) run anywhere. Scorer-correctness tests need a
tiny causal LM (sshleifer/tiny-gpt2, gpt2 tokenizer + random weights - fine for checking the
teacher-forced math); they skip gracefully if it cannot be loaded.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alias_inertia.backends.base import ScoreResult  # noqa: E402

TINY_MODEL = os.environ.get("ALIAS_INERTIA_TEST_MODEL", "sshleifer/tiny-gpt2")


def fake_count_tokens(text: str) -> int:
    """Deterministic word/punct token counter for model-free stimulus tests."""
    return len(re.findall(r"\w+|\S", text))


class StubBackend:
    """Backend returning scripted log-probs keyed by continuation string (for metric tests)."""

    def __init__(self, table: dict[str, float], *, params: dict | None = None):
        self.table = table
        self.params = params or {}
        self.calls = 0

    @property
    def id(self):
        return "stub:test"

    def fingerprint(self):
        return {"backend": "stub", **self.params}

    def cache_fingerprint(self):
        # include score-affecting params so the cache key reflects them
        return {"backend": "stub", **self.params}

    def tokenize(self, text):
        return list(range(fake_count_tokens(text)))

    def count_tokens(self, text):
        return fake_count_tokens(text)

    def score_continuation(self, prompt, continuation):
        return self.score_many(prompt, [continuation])[0]

    def score_many(self, prompt, continuations):
        self.calls += len(continuations)
        out = []
        for c in continuations:
            lp = float(self.table.get(c, -100.0))
            out.append(ScoreResult(logprob=lp, n_tokens=1, token_ids=(0,), token_logprobs=(lp,), boundary_merge=False))
        return out


@pytest.fixture
def fake_tokenizer():
    return fake_count_tokens


@pytest.fixture
def stub_backend_factory():
    return StubBackend


@pytest.fixture(scope="session")
def hf_backend():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from alias_inertia.backends.hf import HFBackend

    try:
        return HFBackend(model=TINY_MODEL)
    except Exception as e:  # pragma: no cover - offline / model unavailable
        pytest.skip(f"cannot load tiny HF model {TINY_MODEL!r}: {e}")
