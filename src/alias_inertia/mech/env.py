"""Model loading, REPO_ROOT, and global hook-filter for the mech arm."""

from __future__ import annotations

import os

# Absolute path to repo root; computed at import time, no hardcoded paths.
REPO_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Models supported for the mech arm (in ascending parameter count).
MECH_MODELS: tuple[str, ...] = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B")


def get_tokenizer(name: str = MECH_MODELS[0]):
    """Return a HuggingFace tokenizer; CPU-safe, no weight loading."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def names_filter_full(n: str) -> bool:
    """Cache filter for full mech runs: resid_pre/post, hook_z, hook_pattern, ln_final.hook_scale."""
    return (
        "resid_pre" in n
        or "resid_post" in n
        or n.endswith("hook_z")
        or n.endswith("hook_pattern")
        or n == "ln_final.hook_scale"
    )


def load_model(name: str, device: str = "cuda"):
    """Load a HookedTransformer in fp16 with seeds set, in eval mode.

    Uses from_pretrained_no_processing (no LN folding / weight centering): the
    weight-processing pass allocates fp32 copies that exceed available RAM for
    the 1.5B model on the reference machine, and unprocessed weights match the
    HF forward pass exactly. The logit-lens final checkpoint reproduces the true
    final logits under this loading mode (verified: argmax-exact, corr 1.0);
    apply_ln handles the unfolded RMSNorm. Both mech models use the same mode so
    per-head attributions are comparable.

    Requires GPU; not called during build-phase verification.
    """
    import torch  # noqa: PLC0415
    from transformer_lens import HookedTransformer  # noqa: PLC0415

    from alias_inertia.determinism import set_determinism  # noqa: PLC0415

    set_determinism(20260618)
    model = HookedTransformer.from_pretrained_no_processing(
        name, device=device, dtype=torch.float16
    )
    model.eval()
    return model
