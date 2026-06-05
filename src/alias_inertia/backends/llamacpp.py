"""llama.cpp teacher-forced scorer (used for the local Llama-3.1-8B pilot).

Reads the GGUF weights that Ollama already downloaded (no re-download) and scores forced
continuations exactly, with **KV-cache reuse**: the (expensive) prompt is forwarded once,
then each continuation only re-evaluates its few suffix tokens. Validated against a
full-sequence ``logits_all`` forward (agrees to within llama.cpp's CPU batch-dependent FP
noise, ~0.1-0.3 nat absolute) and verified bit-identical run-to-run with fixed
``n_threads``/``n_batch``/``seed``.

Because every continuation for a prompt shares the identical reused base forward, the
prior-pull metric (a difference of log-sum-exps over continuations) is internally
self-consistent and unaffected by that absolute-scale FP offset.
"""

from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np

from .base import Backend, ScoreResult, common_prefix_len


def resolve_ollama_gguf(model_tag: str) -> tuple[str, str]:
    """Locate the GGUF blob for an Ollama model tag (e.g. ``"llama3.1:8b"``).

    Returns ``(gguf_path, blob_digest)``. ``blob_digest`` (the content sha256) is the
    canonical model fingerprint recorded in the manifest.
    """
    models_dir = os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")
    name, _, tag = model_tag.partition(":")
    tag = tag or "latest"
    # Ollama library models live under registry.ollama.ai/library/<name>/<tag>.
    candidates = [
        os.path.join(models_dir, "manifests", "registry.ollama.ai", "library", name, tag),
        os.path.join(models_dir, "manifests", "registry.ollama.ai", name, tag),
    ]
    manifest_path = next((c for c in candidates if os.path.isfile(c)), None)
    if manifest_path is None:
        raise FileNotFoundError(
            f"Ollama manifest for {model_tag!r} not found under {models_dir}. "
            f"Tried: {candidates}"
        )
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    model_layers = [L for L in manifest.get("layers", []) if L.get("mediaType", "").endswith("image.model")]
    if not model_layers:
        raise ValueError(f"no model layer in Ollama manifest {manifest_path}")
    digest = model_layers[0]["digest"]  # e.g. "sha256:667b0c..."
    blob = os.path.join(models_dir, "blobs", digest.replace(":", "-"))
    if not os.path.isfile(blob):
        raise FileNotFoundError(f"GGUF blob {blob} referenced by manifest does not exist")
    return blob, digest


def _logsoftmax(row: np.ndarray) -> np.ndarray:
    x = np.asarray(row, dtype=np.float64)
    m = x.max()
    return x - m - math.log(float(np.exp(x - m).sum()))


class LlamaCppBackend(Backend):
    def __init__(
        self,
        *,
        gguf_path: str | None = None,
        ollama_model: str | None = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
        seed: int = 12345,
    ):
        from llama_cpp import Llama  # noqa: PLC0415
        import llama_cpp  # noqa: PLC0415

        self._llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
        self.blob_digest = None
        if gguf_path is None:
            if ollama_model is None:
                raise ValueError("provide gguf_path or ollama_model")
            gguf_path, self.blob_digest = resolve_ollama_gguf(ollama_model)
        self.gguf_path = gguf_path
        self.ollama_model = ollama_model
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed

        # logits_all=True so per-position logits are available for multi-token continuations.
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            seed=seed,
            logits_all=True,
            verbose=False,
        )
        self._metadata = dict(getattr(self._llm, "metadata", {}) or {})

    @property
    def id(self) -> str:
        tag = self.ollama_model or os.path.basename(self.gguf_path)
        short = (self.blob_digest or "")[:19]
        return f"llamacpp:{tag}@{short}"

    @property
    def max_context(self) -> int:
        return self.n_ctx

    def fingerprint(self) -> dict:
        return {
            "backend": "llamacpp",
            "ollama_model": self.ollama_model,
            "gguf_path": self.gguf_path,
            "blob_digest": self.blob_digest,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "seed": self.seed,
            "logits_all": True,
            "llama_cpp_version": self._llama_cpp_version,
            "gguf_general_architecture": self._metadata.get("general.architecture"),
            "gguf_quantization": self._metadata.get("general.file_type"),
            "gguf_context_length": self._metadata.get(
                f"{self._metadata.get('general.architecture', '')}.context_length"
            ),
            "scoring_method": "teacher_forced_kv_reuse",
            "add_bos": True,
        }

    def cache_fingerprint(self) -> dict:
        # llama.cpp CPU scores are only bit-identical for fixed n_threads/n_batch/seed/n_ctx;
        # all of them (and the GGUF content digest) gate the disk-cache key.
        return {
            "backend": "llamacpp",
            "blob_digest": self.blob_digest,
            "gguf_basename": os.path.basename(self.gguf_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "seed": self.seed,
            "logits_all": True,
            "scoring_method": "teacher_forced_kv_reuse",
        }

    def tokenize(self, text: str) -> list[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        # Greedy (temperature 0) decode; return only the newly generated text.
        out = self._llm.create_completion(
            prompt=prompt, max_tokens=max_new_tokens, temperature=0.0, top_k=1, echo=False
        )
        return out["choices"][0]["text"]

    def close(self) -> None:  # llama.cpp frees on GC; provided for interface symmetry
        try:
            del self._llm
        except Exception:  # pragma: no cover
            pass

    def score_continuation(self, prompt: str, continuation: str) -> ScoreResult:
        return self.score_many(prompt, [continuation])[0]

    def score_many(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        llm = self._llm
        p_ids = self.tokenize(prompt)
        fulls = [self.tokenize(prompt + c) for c in continuations]
        ks = [common_prefix_len(p_ids, f) for f in fulls]

        # base = longest common token prefix of the prompt and EVERY full sequence -> the
        # forward we reuse. Robust to BPE merges of any depth at the prompt boundary.
        base_len = min([len(p_ids)] + ks) if ks else len(p_ids)
        base_len = max(base_len, 1)  # need >=1 token of context to score the next token
        base = p_ids[:base_len]

        if len(base) > self.n_ctx:
            raise ValueError(
                f"prompt ({len(p_ids)} tokens) exceeds n_ctx={self.n_ctx}; raise n_ctx or skip this depth"
            )

        llm.reset()
        llm.eval(base)

        results: list[ScoreResult] = []
        for f, k in zip(fulls, ks):
            boundary = k < len(p_ids)
            if k >= len(f):  # continuation added no tokens
                results.append(ScoreResult(0.0, 0, (), (), boundary_merge=boundary))
                continue
            if len(f) > self.n_ctx:
                raise ValueError(f"prompt+continuation ({len(f)} tokens) exceeds n_ctx={self.n_ctx}")

            # Reuse the base KV cache: rewind the position pointer and evaluate only the suffix.
            llm.n_tokens = base_len
            suffix = f[base_len:]
            if suffix:
                llm.eval(suffix)

            tok_lps: list[float] = []
            for j in range(k, len(f)):
                row = llm.scores[j - 1]  # logits predicting token j, given f[:j]
                tok_lps.append(float(_logsoftmax(row)[f[j]]))

            results.append(
                ScoreResult(
                    logprob=float(sum(tok_lps)),
                    n_tokens=len(tok_lps),
                    token_ids=tuple(f[k:]),
                    token_logprobs=tuple(tok_lps),
                    boundary_merge=boundary,
                )
            )
        return results
