"""The repeat-question cache: eviction, expiry, and what makes a key."""

from __future__ import annotations

import time

from bse_nlq.cache import TTLCache, key_for


def test_keys_are_unambiguous_across_part_boundaries():
    """("ab","c") and ("a","bc") must not collide -- a question ending in a
    date fragment could otherwise be served an answer for a different one."""
    assert key_for("ab", "c") != key_for("a", "bc")
    assert key_for("q", "2026-09-21") == key_for("q", "2026-09-21")


def test_entries_expire():
    cache = TTLCache[str](max_entries=4, ttl_seconds=0.05)
    cache.put("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.06)
    assert cache.get("k") is None
    assert len(cache) == 0, "an expired entry is dropped on the way past"


def test_least_recently_used_is_evicted_first():
    cache = TTLCache[str](max_entries=2, ttl_seconds=60)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.get("a")                 # 'a' is now the most recently used
    cache.put("c", "3")
    assert cache.get("a") == "1"
    assert cache.get("b") is None
    assert cache.get("c") == "3"


def test_zero_ttl_or_size_disables_it_without_a_branch_at_the_call_site():
    for cache in (TTLCache[str](max_entries=0, ttl_seconds=60),
                  TTLCache[str](max_entries=8, ttl_seconds=0)):
        assert not cache.enabled
        cache.put("k", "v")
        assert cache.get("k") is None


def test_stats_track_hit_rate():
    cache = TTLCache[str](max_entries=4, ttl_seconds=60)
    cache.put("k", "v")
    cache.get("k")
    cache.get("missing")
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5
