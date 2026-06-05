"""HuggingFace teacher-forced scorer - the RELIABLE reference path.

One forward pass over ``prompt+continuation``; sum ``log_softmax`` at the continuation
positions (identified by longest-common-token-prefix, never by character offset). This is
the backend the test-suite checks against a hand-computed reference, and the recommended
local scorer when llama.cpp is unavailable.
"""

from __future__ import annotations

from typing import Sequence

from .base import Backend, ScoreResult, common_prefix_len


class HFBackend(Backend):
    def __init__(
        self,
        model: str = "sshleifer/tiny-gpt2",
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        add_special_tokens: bool = False,
    ):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.model_name = model
        self.revision = revision
        self.device = device
        self.dtype_name = dtype
        self.add_special_tokens = add_special_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self._model = AutoModelForCausalLM.from_pretrained(model, revision=revision)
        torch_dtype = getattr(torch, dtype)
        self._model = self._model.to(device=device, dtype=torch_dtype)
        self._model.eval()

        # Resolve the model's commit hash for provenance, if available.
        self._commit = getattr(getattr(self._model, "config", None), "_commit_hash", None) or revision

    @property
    def id(self) -> str:
        return f"hf:{self.model_name}@{self._commit or 'main'}"

    @property
    def max_context(self) -> int:
        cfg = self._model.config
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        return 1 << 30  # effectively unbounded

    def fingerprint(self) -> dict:
        import transformers  # noqa: PLC0415

        return {
            "backend": "hf",
            "model": self.model_name,
            "revision": self.revision,
            "commit_hash": self._commit,
            "device": self.device,
            "dtype": self.dtype_name,
            "add_special_tokens": self.add_special_tokens,
            "max_context": self.max_context,
            "transformers_version": transformers.__version__,
            "torch_version": self._torch.__version__,
            "scoring_method": "teacher_forced_full_forward",
        }

    def cache_fingerprint(self) -> dict:
        # Every field here changes a returned log-prob; folded into the disk-cache key.
        return {
            "backend": "hf",
            "model": self.model_name,
            "commit_hash": self._commit,
            "dtype": self.dtype_name,
            "device": self.device,
            "add_special_tokens": self.add_special_tokens,
            "scoring_method": "teacher_forced_full_forward",
        }

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=self.add_special_tokens)["input_ids"]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, prompt: str, max_new_tokens: int = 48) -> str:
        torch = self._torch
        ids = self.tokenizer(prompt, add_special_tokens=self.add_special_tokens, return_tensors="pt")
        ids = {k: v.to(self.device) for k, v in ids.items()}
        n_in = ids["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy / deterministic
                num_beams=1,
                pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            )
        gen_ids = out[0][n_in:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def close(self) -> None:
        """Free GPU memory (call between models in a multi-model run)."""
        try:
            del self._model
            if self.device.startswith("cuda"):
                self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def score_continuation(self, prompt: str, continuation: str) -> ScoreResult:
        return self.score_many(prompt, [continuation])[0]

    def _forward_kept_logits(self, input_ids, keep: int):
        """Forward and return logits for only the last ``keep`` positions ([keep, V]).

        Uses ``logits_to_keep`` (transformers v5) / ``num_logits_to_keep`` (v4) so a long-context
        forward does NOT materialise a [seq_len, vocab] logits tensor (which OOMs at 8k/32k).
        Falls back to full logits when the model does not support the argument.
        """
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                return self._model(input_ids, **{kw: keep}).logits[0]
            except TypeError:
                continue
        return self._model(input_ids).logits[0]  # full logits (fine for short sequences)

    def score_many(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        torch = self._torch
        p_ids = self.tokenize(prompt)
        results: list[ScoreResult] = []
        with torch.no_grad():
            for cont in continuations:
                f_ids = self.tokenize(prompt + cont)
                k = common_prefix_len(p_ids, f_ids)
                if k < 1:
                    raise ValueError(
                        "prompt is not a token-prefix of prompt+continuation; cannot score "
                        f"(prompt={prompt!r}, continuation={cont!r})"
                    )
                if k >= len(f_ids):
                    # Continuation added no new tokens (degenerate). Score is 0 over 0 tokens.
                    results.append(ScoreResult(0.0, 0, (), (), boundary_merge=k < len(p_ids)))
                    continue

                keep = len(f_ids) - k + 1  # logits for positions [k-1 .. len-1] (predict tokens k..len-1)
                input_ids = torch.tensor([f_ids], device=self.device)
                logits = self._forward_kept_logits(input_ids, keep)  # [L_kept, V]
                logprobs = torch.log_softmax(logits.to(torch.float64), dim=-1)
                offset = len(f_ids) - logprobs.shape[0]  # 0 if full logits; k-1 if kept

                tok_lps: list[float] = []
                for j in range(k, len(f_ids)):
                    tok_lps.append(float(logprobs[(j - 1) - offset, f_ids[j]].item()))
                results.append(
                    ScoreResult(
                        logprob=float(sum(tok_lps)),
                        n_tokens=len(tok_lps),
                        token_ids=tuple(f_ids[k:]),
                        token_logprobs=tuple(tok_lps),
                        boundary_merge=k < len(p_ids),
                    )
                )
        return results
