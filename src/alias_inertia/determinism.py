"""Reproducibility utilities: deterministic setup, environment fingerprinting, stable hashing.

Everything that affects a result is captured here so a run can be reconstructed exactly:
seeds are pinned, the full software/hardware environment is fingerprinted into the run
manifest, and configs are hashed so any change invalidates downstream caches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _md
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

# Packages whose versions materially affect numerical results; recorded in every manifest.
_FINGERPRINT_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "statsmodels",
    "matplotlib",
    "PyYAML",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "llama-cpp-python",
)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp (single source of truth for run times)."""
    return datetime.now(timezone.utc).isoformat()


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serialisable object.

    Used to hash configs, prompts and cache keys. Key order is normalised so that
    logically-identical objects hash identically regardless of dict insertion order.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def sha256_text(text: str) -> str:
    """Full sha256 hex digest of a string (used for prompt provenance)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _FINGERPRINT_PACKAGES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def environment_fingerprint() -> dict[str, Any]:
    """Capture the software/hardware environment for the run manifest."""
    fp: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
    # Torch / CUDA details (optional; only if torch importable).
    try:
        import torch  # noqa: PLC0415

        fp["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_devices": (
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
        }
    except Exception:  # pragma: no cover - torch optional
        fp["torch"] = {"version": "not-installed"}
    return fp


def set_determinism(seed: int, *, torch_threads: int | None = None) -> dict[str, Any]:
    """Pin all RNGs and request deterministic kernels. Returns what was set (for the manifest).

    Note: PYTHONHASHSEED cannot be changed after interpreter start; callers who need
    hash-randomisation disabled should export PYTHONHASHSEED=0 before launching Python.
    """
    import random

    import numpy as np

    state: dict[str, Any] = {"seed": seed, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>")}

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism-first knobs.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))
            state["torch_num_threads"] = int(torch_threads)
        state["torch_deterministic"] = True
    except Exception:  # pragma: no cover - torch optional
        state["torch_deterministic"] = "torch-not-installed"

    return state
