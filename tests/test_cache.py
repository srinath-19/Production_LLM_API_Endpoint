"""
Tests for the caching layer.
Fast, deterministic, no external dependencies.
"""

import time
from app.cache import ResponseCache


class TestResponseCache:
    """Test the response cache."""

    def setup_method(self):
        self.cache = ResponseCache(ttl_seconds=2)

    def test_cache_miss_returns_none(self):
        assert self.cache.get("unknown query") is None

    def test_cache_hit_returns_response(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("What is Python?")
        assert result == "A programming language."

    def test_case_insensitive_matching(self):
        self.cache.set("What is Python?", "A programming language.")
        result = self.cache.get("what is python?")
        assert result == "A programming language."

    def test_ttl_expiration(self):
        self.cache = ResponseCache(ttl_seconds=1)
        self.cache.set("query", "response")
        assert self.cache.get("query") == "response"
        time.sleep(1.5)
        assert self.cache.get("query") is None

    def test_stats_tracking(self):
        self.cache.get("miss1")
        self.cache.get("miss2")
        self.cache.set("hit", "value")
        self.cache.get("hit")

        stats = self.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["cached_entries"] == 1


class TestCacheEviction:
    """The size bound is what actually keeps memory flat."""

    def test_never_exceeds_max_size(self):
        cache = ResponseCache(ttl_seconds=300, max_size=3)
        for i in range(50):
            cache.set(f"query {i}", f"response {i}")
        assert cache.stats["cached_entries"] == 3

    def test_evicts_least_recently_used_first(self):
        cache = ResponseCache(ttl_seconds=300, max_size=3)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")  # pushes out "a", the oldest

        assert cache.get("a") is None
        assert cache.get("d") == "4"

    def test_reading_an_entry_protects_it_from_eviction(self):
        cache = ResponseCache(ttl_seconds=300, max_size=3)
        cache.set("hot", "value")
        for i in range(5):
            cache.set(f"cold {i}", "junk")
            cache.get("hot")  # keep it recently-used

        assert cache.get("hot") == "value"

    def test_eviction_counter(self):
        cache = ResponseCache(ttl_seconds=300, max_size=2)
        for i in range(5):
            cache.set(f"q{i}", "v")
        assert cache.stats["evictions"] == 3

    def test_expired_entries_reclaimed_before_live_ones(self):
        cache = ResponseCache(ttl_seconds=1, max_size=3)
        cache.set("old1", "v")
        cache.set("old2", "v")
        time.sleep(1.2)  # both now stale

        cache.set("new1", "v")
        cache.set("new2", "v")
        cache.set("new3", "v")

        # The two stale entries should be purged rather than evicting fresh ones.
        assert cache.get("new1") == "v"
        assert cache.stats["expirations"] >= 2
