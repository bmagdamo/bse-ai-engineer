"""A small TTL + LRU cache, used to answer a repeated question for free.

Prompt caching already makes the *tokens* cheap; this removes the round trip
entirely. Two model calls and a scan over ~850k ticket rows is a lot to spend
on a question someone asked ninety seconds ago -- and in a shared dashboard,
the same three questions are most of the traffic.

The cache is generic in its value type and imports nothing from the agent, so
the dependency runs one way: the agent knows about the cache, never the
reverse. What makes an entry safe to reuse is the *key*, built by the caller
out of everything an answer depends on -- see `NLQAgent._cache_key`.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


def key_for(*parts: str) -> str:
    """A stable key from the parts an answer depends on.

    Joined with a separator that cannot occur in the parts, so ("ab", "c")
    and ("a", "bc") cannot collide, then hashed to keep keys a fixed size
    however long a question is.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TTLCache[V]:
    """Least-recently-used, with a per-entry expiry.

    Thread-safe because the Streamlit app serves every session from one cached
    agent, so two script-runner threads can reach the same instance at once.

    A zero `ttl_seconds` or `max_entries` disables it: `get` always misses and
    `put` stores nothing, so caching can be turned off by configuration
    without the agent growing a branch for it.
    """

    def __init__(self, max_entries: int = 128, ttl_seconds: float = 300.0):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: str) -> V | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] <= time.monotonic():
                # An expired entry is dropped on the way past rather than by a
                # sweeper thread: the only entries worth evicting are the ones
                # somebody actually looked for.
                self._entries.pop(key, None)
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry[1]

    def put(self, key: str, value: V) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(hits=self._hits, misses=self._misses)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
