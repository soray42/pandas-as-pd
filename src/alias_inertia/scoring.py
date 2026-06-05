"""Scoring orchestration: on-disk response cache and a caching wrapper around any backend.

The single scoring primitive is the backend's ``score_continuation`` /
``score_many`` (teacher-forced summed log-prob). This module adds:
  * :class:`DiskCache` - content-addressed JSON cache so reruns never recompute / re-query.
  * :class:`CachingScorer` - wraps a backend, caches per (backend, scoring_version, prompt,
    continuation), and lets the backend reuse the prompt forward across cache-misses.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from typing import Sequence

from .backends.base import Backend, ScoreResult
from .determinism import stable_hash

SCORING_VERSION = "1.0"


def scoring_code_hash(backend) -> str:
    """Hash the source of the scoring-relevant modules (this module, metrics, base, and the
    active backend). Folded into the cache key so editing scoring/boundary logic AUTOMATICALLY
    invalidates stale cache entries - no manual ``scoring_version`` bump required.
    """
    from . import metrics  # noqa: PLC0415
    from .backends import base as base_mod  # noqa: PLC0415

    mods = [sys.modules.get(__name__), metrics, base_mod, sys.modules.get(type(backend).__module__)]
    srcs = []
    for m in mods:
        try:
            srcs.append(inspect.getsource(m) if m is not None else "")
        except (OSError, TypeError):  # source not available (e.g. zipapp)
            srcs.append("")
    return stable_hash(srcs, length=12)


class DiskCache:
    """Tiny content-addressed JSON cache (one file per key, sharded by prefix)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        sub = os.path.join(self.root, key[:2])
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{key}.json")

    def get(self, key: str):
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):  # pragma: no cover - corrupt cache entry
            return None

    def put(self, key: str, value) -> None:
        path = self._path(key)
        # Atomic write so an interrupted run never leaves a half-written cache entry.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class CachingScorer:
    """Wrap a backend with a per-(prompt, continuation) disk cache."""

    def __init__(
        self,
        backend: Backend,
        *,
        cache_dir: str | None = None,
        scoring_version: str = SCORING_VERSION,
        enabled: bool = True,
    ):
        self.backend = backend
        self.scoring_version = scoring_version
        self.enabled = enabled and cache_dir is not None
        self._cache = DiskCache(os.path.join(cache_dir, "scores")) if self.enabled else None
        self.stats = {"hits": 0, "misses": 0}

        # The cache key folds in (a) every score-affecting backend setting and (b) a hash of the
        # scoring code, so a cache hit is guaranteed to have been computed under the SAME config
        # and the SAME scoring implementation. backend.id alone (human-readable) is insufficient.
        cfp = getattr(backend, "cache_fingerprint", None)
        self.backend_cache_fp = cfp() if callable(cfp) else {"id": backend.id}
        self.code_hash = scoring_code_hash(backend)
        self.cache_id = stable_hash(
            {"backend": self.backend_cache_fp, "code": self.code_hash, "scoring_version": scoring_version},
            length=24,
        )

    # passthroughs ------------------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        return self.backend.tokenize(text)

    def count_tokens(self, text: str) -> int:
        return self.backend.count_tokens(text)

    def fingerprint(self) -> dict:
        return self.backend.fingerprint()

    @property
    def id(self) -> str:
        return self.backend.id

    # scoring ----------------------------------------------------------------------
    def _key(self, prompt: str, cont: str) -> str:
        # cache_id already encodes backend cache-fingerprint + scoring-code hash + scoring_version.
        return stable_hash([self.cache_id, prompt, cont], length=32)

    def score_many(self, prompt: str, continuations: Sequence[str]) -> list[ScoreResult]:
        continuations = list(continuations)
        results: list[ScoreResult | None] = [None] * len(continuations)
        miss_idx: list[int] = []

        if self._cache is not None:
            for i, cont in enumerate(continuations):
                hit = self._cache.get(self._key(prompt, cont))
                if hit is not None:
                    results[i] = ScoreResult.from_json(hit)
                    self.stats["hits"] += 1
                else:
                    miss_idx.append(i)
        else:
            miss_idx = list(range(len(continuations)))

        if miss_idx:
            miss_conts = [continuations[i] for i in miss_idx]
            computed = self.backend.score_many(prompt, miss_conts)
            for i, res in zip(miss_idx, computed):
                results[i] = res
                self.stats["misses"] += 1
                if self._cache is not None:
                    self._cache.put(self._key(prompt, continuations[i]), res.to_json())

        return [r for r in results]  # type: ignore[return-value]

    def score_continuation(self, prompt: str, continuation: str) -> ScoreResult:
        return self.score_many(prompt, [continuation])[0]
