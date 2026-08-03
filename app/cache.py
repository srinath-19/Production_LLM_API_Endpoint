"""
Response Caching Layer
In-memory LRU cache with TTL for LLM response deduplication.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Optional


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live) and an LRU size bound.

    Two independent limits keep memory flat:
      - ttl_seconds: how long an entry stays fresh
      - max_size:    hard ceiling on entry count; the least-recently-used
                     entry is evicted once the ceiling is reached

    Without the size bound the cache grows forever, because expired entries
    are only noticed when that exact key is requested again. A key that is
    never asked for a second time would never be reclaimed.

    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        # OrderedDict preserves insertion/access order, which gives us LRU
        # semantics: the leftmost item is the least recently used.
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _make_key(self, query: str) -> str:
        """Create a cache key from the normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    # 'What is Python?' and 'what is python?'

    def _is_expired(self, entry: dict, now: float) -> bool:
        return now - entry["timestamp"] >= self.ttl

    def _purge_expired(self) -> None:
        """
        Drop every entry past its TTL.

        Entries are inserted in time order and `set` refreshes the timestamp,
        but `get` also moves entries to the end on a hit, so ordering is by
        access rather than age. We therefore scan the whole cache instead of
        stopping at the first fresh entry.
        """
        now = time.time()
        expired = [k for k, v in self._cache.items() if self._is_expired(v, now)]
        for key in expired:
            del self._cache[key]
            self._expirations += 1

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry, time.time()):
                # Mark as most recently used so it survives the next eviction.
                self._cache.move_to_end(key)
                self._hits += 1
                return entry["response"]
            # Expired - remove it
            del self._cache[key]
            self._expirations += 1

        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Cache a response, evicting the least-recently-used entry if full."""
        key = self._make_key(query)

        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }
        self._cache.move_to_end(key)

        # Reclaim dead entries before evicting live ones - an expired entry is
        # always the better thing to drop.
        if len(self._cache) > self.max_size:
            self._purge_expired()

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # pop the least recently used
            self._evictions += 1

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "evictions": self._evictions,
            "expirations": self._expirations,
        }
