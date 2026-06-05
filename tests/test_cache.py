"""Cache-key correctness (reproducibility): a cached score is only reused when the backend's
score-affecting settings AND the scoring code are identical. Regression test for the review
finding that the key used backend.id (which omits dtype / n_threads / n_batch / seed / ...)."""

from __future__ import annotations

import math

from alias_inertia.scoring import CachingScorer


def test_cache_hit_avoids_recompute(tmp_path, stub_backend_factory):
    be = stub_backend_factory({"a(": math.log(0.5)})
    sc = CachingScorer(be, cache_dir=str(tmp_path), enabled=True)
    sc.score_continuation("p", "a(")
    assert be.calls == 1
    sc.score_continuation("p", "a(")  # served from cache
    assert be.calls == 1
    assert sc.stats == {"hits": 1, "misses": 1}


def test_score_affecting_param_changes_cache_id(tmp_path, stub_backend_factory):
    be1 = stub_backend_factory({"a(": math.log(0.5)}, params={"n_threads": 8})
    be2 = stub_backend_factory({"a(": math.log(0.5)}, params={"n_threads": 4})
    sc1 = CachingScorer(be1, cache_dir=str(tmp_path), enabled=True)
    sc2 = CachingScorer(be2, cache_dir=str(tmp_path), enabled=True)
    # Different score-affecting params -> different cache_id -> different key (no stale reuse).
    assert sc1.cache_id != sc2.cache_id
    assert sc1._key("p", "a(") != sc2._key("p", "a(")


def test_same_config_shares_cache_id(tmp_path, stub_backend_factory):
    be1 = stub_backend_factory({"a(": -1.0}, params={"n_threads": 8})
    be2 = stub_backend_factory({"a(": -1.0}, params={"n_threads": 8})
    sc1 = CachingScorer(be1, cache_dir=str(tmp_path), enabled=True)
    sc2 = CachingScorer(be2, cache_dir=str(tmp_path), enabled=True)
    assert sc1.cache_id == sc2.cache_id  # identical config + code -> shared cache


def test_scoring_version_in_cache_id(tmp_path, stub_backend_factory):
    be = stub_backend_factory({"a(": -1.0})
    a = CachingScorer(be, cache_dir=str(tmp_path), scoring_version="1.0", enabled=True)
    b = CachingScorer(be, cache_dir=str(tmp_path), scoring_version="2.0", enabled=True)
    assert a.cache_id != b.cache_id  # manual cache-buster still works


def test_code_hash_is_recorded(tmp_path, stub_backend_factory):
    be = stub_backend_factory({"a(": -1.0})
    sc = CachingScorer(be, cache_dir=str(tmp_path), enabled=True)
    assert isinstance(sc.code_hash, str) and len(sc.code_hash) > 0
