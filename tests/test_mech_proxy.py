"""Tests for alias_inertia.mech.proxy: ProxyLexicon, proxy_pull, build_proxy_lexicon.

proxy_pull operates on raw logits (no log-softmax): the additive shift from the
log-partition cancels in the logsumexp difference, making the result invariant.
CPU-only tests (synthetic tensors) run without any model download or GPU.
"""

from __future__ import annotations

import pytest
import torch

from alias_inertia.mech.proxy import ProxyLexicon, build_proxy_lexicon, proxy_pull


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lex(prior_ids, bound_ids):
    return ProxyLexicon(
        prior_lib="numpy",
        bound_lib="pandas",
        prior_ids=list(prior_ids),
        bound_ids=list(bound_ids),
        prior_strings=[],
        bound_strings=[],
        dropped=[],
        sha256="",
    )


def _logits(size=200, overrides=None):
    """1-D tensor of zeros with selective overrides {token_id: logit}."""
    t = torch.zeros(size)
    if overrides:
        for idx, val in overrides.items():
            t[int(idx)] = val
    return t


# ---------------------------------------------------------------------------
# sign contract
# ---------------------------------------------------------------------------

def test_proxy_pull_positive_when_prior_higher():
    # prior ids carry high logits -> pull > 0 (corpus inertia dominates)
    lex = _lex([10, 11], [20, 21])
    logits = _logits(overrides={10: 5.0, 11: 5.0, 20: 1.0, 21: 1.0})
    assert proxy_pull(logits, lex).item() > 0


def test_proxy_pull_negative_when_bound_higher():
    lex = _lex([10, 11], [20, 21])
    logits = _logits(overrides={10: 1.0, 11: 1.0, 20: 5.0, 21: 5.0})
    assert proxy_pull(logits, lex).item() < 0


def test_proxy_pull_zero_when_symmetric():
    # equal logits -> logsumexp(prior) == logsumexp(bound)
    lex = _lex([0, 1, 2], [3, 4, 5])
    logits = torch.zeros(100)
    assert abs(proxy_pull(logits, lex).item()) < 1e-6


# ---------------------------------------------------------------------------
# numerical correctness
# ---------------------------------------------------------------------------

def test_proxy_pull_numerical_logsumexp():
    # logsumexp([5, 5]) - logsumexp([1, 1]) = (5 + log2) - (1 + log2) = 4.0
    lex = _lex([10, 11], [20, 21])
    logits = torch.full((100,), -1e6)
    logits[10] = 5.0
    logits[11] = 5.0
    logits[20] = 1.0
    logits[21] = 1.0
    val = proxy_pull(logits, lex).item()
    assert abs(val - 4.0) < 1e-4


def test_proxy_pull_single_id_each_side():
    # single id each side: pull = logits[prior] - logits[bound]
    lex = _lex([7], [42])
    logits = _logits(overrides={7: 3.5, 42: 1.5})
    val = proxy_pull(logits, lex).item()
    assert abs(val - 2.0) < 1e-4


def test_proxy_pull_invariant_to_additive_shift():
    # adding a constant to ALL logits shifts both logsumexp equally -> pull unchanged
    lex = _lex([5, 6], [7, 8])
    logits = _logits(overrides={5: 2.0, 6: 3.0, 7: 1.0, 8: 0.5})
    base = proxy_pull(logits, lex).item()
    shifted = proxy_pull(logits + 100.0, lex).item()
    assert abs(base - shifted) < 1e-3


# ---------------------------------------------------------------------------
# batched / higher-dim logits
# ---------------------------------------------------------------------------

def test_proxy_pull_batched_shape():
    lex = _lex([10], [20])
    logits = torch.zeros(4, 100)
    logits[0, 10] = 5.0   # prior higher in row 0
    logits[1, 20] = 5.0   # bound higher in row 1
    result = proxy_pull(logits, lex)
    assert result.shape == (4,)
    assert result[0].item() > 0
    assert result[1].item() < 0
    assert abs(result[2].item()) < 1e-6
    assert abs(result[3].item()) < 1e-6


def test_proxy_pull_3d_logits():
    # [batch, seq, vocab] shaped input
    lex = _lex([0], [1])
    logits = torch.zeros(2, 3, 50)
    logits[0, 2, 0] = 2.0
    result = proxy_pull(logits, lex)
    assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# ProxyLexicon dataclass
# ---------------------------------------------------------------------------

def test_proxy_lexicon_fields():
    lex = ProxyLexicon(
        prior_lib="numpy",
        bound_lib="pandas",
        prior_ids=[1, 2],
        bound_ids=[3, 4],
        prior_strings=["array(", "arange("],
        bound_strings=["DataFrame(", "read_csv("],
        dropped=["sum"],
        sha256="abcdef",
    )
    assert lex.prior_lib == "numpy"
    assert lex.bound_lib == "pandas"
    assert lex.prior_ids == [1, 2]
    assert lex.bound_ids == [3, 4]
    assert lex.dropped == ["sum"]
    assert len(lex.sha256) > 0


def test_proxy_lexicon_empty_dropped():
    lex = _lex([0], [1])
    assert lex.dropped == []


# ---------------------------------------------------------------------------
# build_proxy_lexicon with a stub tokenizer (no model download)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Deterministic stub: assign each unique leading character a unique id."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next = 10

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        key = text[:3]  # first 3 chars as discriminator
        if key not in self._vocab:
            self._vocab[key] = self._next
            self._next += 1
        return [self._vocab[key]]


def test_build_proxy_lexicon_returns_proxy_lexicon():
    tok = _StubTokenizer()
    lex = build_proxy_lexicon(tok, "numpy", "pandas")
    assert isinstance(lex, ProxyLexicon)
    assert lex.prior_lib == "numpy"
    assert lex.bound_lib == "pandas"


def test_build_proxy_lexicon_ids_are_lists_of_ints():
    tok = _StubTokenizer()
    lex = build_proxy_lexicon(tok, "numpy", "pandas")
    assert isinstance(lex.prior_ids, list)
    assert isinstance(lex.bound_ids, list)
    assert all(isinstance(i, int) for i in lex.prior_ids)
    assert all(isinstance(i, int) for i in lex.bound_ids)


def test_build_proxy_lexicon_no_id_collision():
    # No first-token id should appear in both prior_ids and bound_ids.
    tok = _StubTokenizer()
    lex = build_proxy_lexicon(tok, "numpy", "pandas")
    assert len(set(lex.prior_ids) & set(lex.bound_ids)) == 0


def test_build_proxy_lexicon_dropped_are_strings():
    tok = _StubTokenizer()
    lex = build_proxy_lexicon(tok, "numpy", "pandas")
    assert isinstance(lex.dropped, list)
    assert all(isinstance(d, str) for d in lex.dropped)


def test_build_proxy_lexicon_sha256_nonempty():
    tok = _StubTokenizer()
    lex = build_proxy_lexicon(tok, "numpy", "pandas")
    assert isinstance(lex.sha256, str) and len(lex.sha256) > 0


def test_build_proxy_lexicon_requires_tokenizer():
    # Passing None as tokenizer should raise an AttributeError (tokenizer.encode is called).
    with pytest.raises((AttributeError, TypeError)):
        build_proxy_lexicon(None, "numpy", "pandas")


# ---------------------------------------------------------------------------
# tokenizer-level CPU check (no model download)
# ---------------------------------------------------------------------------

def test_proxy_pull_device_consistent_with_logits():
    lex = _lex([0], [1])
    logits = torch.tensor([1.0, 0.5, 0.0])
    result = proxy_pull(logits, lex)
    # result should be a scalar-like tensor
    assert result.numel() == 1 or result.shape == ()
