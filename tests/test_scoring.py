"""Scorer correctness: the HF backend's teacher-forced log-prob must match an INDEPENDENT
reference computed position-by-position (a different code path), and handle BPE boundary
merges (prompt ending in '.') correctly.

Needs a tiny causal LM (sshleifer/tiny-gpt2). Skips if it cannot be loaded.
"""

from __future__ import annotations

import pytest

from alias_inertia.backends.base import common_prefix_len

pytestmark = pytest.mark.needs_hf


def _reference_logprob(backend, prompt, continuation):
    """Independent reference: score each continuation token with its OWN forward over the
    growing prefix (autoregressive), instead of one batched forward. Must agree with the
    backend's batched implementation up to floating-point tolerance.
    """
    import torch

    p_ids = backend.tokenize(prompt)
    f_ids = backend.tokenize(prompt + continuation)
    k = common_prefix_len(p_ids, f_ids)
    total = 0.0
    toks = []
    with torch.no_grad():
        for j in range(k, len(f_ids)):
            ctx = torch.tensor([f_ids[:j]], device=backend.device)
            logits = backend._model(ctx).logits[0, -1]  # next-token logits given f_ids[:j]
            lp = torch.log_softmax(logits.to(torch.float64), dim=-1)[f_ids[j]]
            total += float(lp.item())
            toks.append(f_ids[j])
    return total, k, toks


def test_matches_independent_reference(hf_backend):
    for prompt, cont in [
        ("Hello world", " foo"),
        ("The quick brown fox", " jumps over"),
        ("import numpy as np\nnp", ".array("),
    ]:
        res = hf_backend.score_continuation(prompt, cont)
        ref, k, toks = _reference_logprob(hf_backend, prompt, cont)
        assert res.n_tokens == len(toks)
        assert abs(res.logprob - ref) < 1e-4, f"{prompt!r}+{cont!r}: {res.logprob} vs {ref}"


def test_single_token_equals_direct_logsoftmax(hf_backend):
    import torch

    prompt = "The capital of France is"
    cont = " Paris"
    res = hf_backend.score_continuation(prompt, cont)
    # Recompute the FIRST continuation token's logprob directly and check it is a component.
    p_ids = hf_backend.tokenize(prompt)
    f_ids = hf_backend.tokenize(prompt + cont)
    k = common_prefix_len(p_ids, f_ids)
    with torch.no_grad():
        logits = hf_backend._model(torch.tensor([f_ids[:k]])).logits[0, -1]
        first_lp = float(torch.log_softmax(logits.to(torch.float64), dim=-1)[f_ids[k]].item())
    assert abs(res.token_logprobs[0] - first_lp) < 1e-4


def test_boundary_merge_flagged_for_dot(hf_backend):
    # Prompt ends in '.'; gpt2 BPE merges the dot into the first continuation token.
    res = hf_backend.score_continuation("import pandas as np\nnp.", "array(")
    p_ids = hf_backend.tokenize("import pandas as np\nnp.")
    f_ids = hf_backend.tokenize("import pandas as np\nnp.array(")
    k = common_prefix_len(p_ids, f_ids)
    assert res.boundary_merge == (k < len(p_ids))
    # score must still equal the independent reference despite the merge
    ref, _, _ = _reference_logprob(hf_backend, "import pandas as np\nnp.", "array(")
    assert abs(res.logprob - ref) < 1e-4


def test_score_many_matches_individual(hf_backend):
    prompt = "import pandas as np\nnp."
    conts = ["array(", "DataFrame(", "arange(", "read_csv("]
    many = hf_backend.score_many(prompt, conts)
    for cont, m in zip(conts, many):
        one = hf_backend.score_continuation(prompt, cont)
        assert abs(m.logprob - one.logprob) < 1e-9
        assert m.token_ids == one.token_ids


def test_logprobs_are_nonpositive(hf_backend):
    res = hf_backend.score_continuation("Hello", " world there")
    for lp in res.token_logprobs:
        assert lp <= 1e-6  # log-probabilities
