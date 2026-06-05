"""Backend registry and factory.

``build_backend(name, cfg, cache_dir)`` constructs the requested scoring backend. Backends
are imported lazily so that running the HF test-suite does not require ``llama-cpp-python``,
and running the llama.cpp path does not require ``transformers``/``torch``.
"""

from __future__ import annotations

from .base import Backend, ScoreResult, common_prefix_len

__all__ = ["Backend", "ScoreResult", "common_prefix_len", "build_backend"]


def build_backend(name: str, cfg: dict, *, cache_dir: str | None = None) -> Backend:
    name = name.lower()
    if name == "hf":
        from .hf import HFBackend

        return HFBackend(
            model=cfg.get("model", "sshleifer/tiny-gpt2"),
            revision=cfg.get("revision"),
            device=cfg.get("device", "cpu"),
            dtype=cfg.get("dtype", "float32"),
            add_special_tokens=cfg.get("add_special_tokens", False),
        )
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(
            gguf_path=cfg.get("gguf_path"),
            ollama_model=cfg.get("ollama_model"),
            n_ctx=int(cfg.get("n_ctx", 4096)),
            n_threads=int(cfg.get("n_threads", 8)),
            n_batch=int(cfg.get("n_batch", 512)),
            seed=int(cfg.get("seed", 12345)),
        )
    raise ValueError(f"unknown backend {name!r} (expected hf | llamacpp)")
