"""Behaviour of the process-wide shared caches.

These back the cross-session sharing that makes a second user's session cheap,
so the bound, the TTL and the build-once-under-contention guarantee all matter.
"""

from __future__ import annotations

import threading

import pytest

from leds._cache import SharedLRU


def test_builds_once_then_serves_from_cache():
    calls = []
    cache = SharedLRU(4)

    for _ in range(3):
        assert cache.get("k", lambda: calls.append(1) or "v") == "v"

    assert len(calls) == 1
    assert (cache.hits, cache.misses) == (2, 1)


def test_caches_a_none_result():
    """A cached ``None`` must not look like a miss."""
    calls = []
    cache = SharedLRU(4)

    cache.get("k", lambda: calls.append(1))
    cache.get("k", lambda: calls.append(1))

    assert len(calls) == 1


def test_evicts_least_recently_used():
    cache = SharedLRU(2)
    cache.get("a", lambda: "a")
    cache.get("b", lambda: "b")
    cache.get("a", lambda: pytest.fail("should still be cached"))  # 'a' is now newest

    cache.get("c", lambda: "c")  # evicts 'b', the least recently used

    assert len(cache._entries) == 2
    rebuilt = []
    cache.get("b", lambda: rebuilt.append(1) or "b")
    assert rebuilt == [1]


def test_entries_expire_after_the_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("leds._cache.time.monotonic", lambda: now[0])
    cache = SharedLRU(4, ttl=60)
    calls = []

    cache.get("k", lambda: calls.append(1) or "v")
    now[0] += 59
    cache.get("k", lambda: calls.append(1) or "v")
    assert len(calls) == 1, "still inside the TTL"

    now[0] += 2
    cache.get("k", lambda: calls.append(1) or "v")
    assert len(calls) == 2, "expired, so rebuilt"


def test_concurrent_callers_build_once():
    """Two sessions asking for the same cold entry must not both build it."""
    cache = SharedLRU(4)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        started.set()
        release.wait(timeout=5)
        return "v"

    first = threading.Thread(target=lambda: cache.get("k", slow))
    first.start()
    started.wait(timeout=5)

    results = []
    second = threading.Thread(target=lambda: results.append(cache.get("k", slow)))
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert calls == [1], "the second caller waited instead of rebuilding"
    assert results == ["v"]


def test_a_failed_build_is_not_cached():
    cache = SharedLRU(4)

    with pytest.raises(RuntimeError):
        cache.get("k", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cache.get("k", lambda: "v") == "v"
    assert not cache._building, "the in-flight lock must be released"


def _wait_for_background():
    from leds import _cache  # noqa: PLC0415

    _cache._LATER.submit(lambda: None).result(timeout=10)


def test_build_later_builds_once_in_the_background():
    calls = []
    cache = SharedLRU(4)

    cache.build_later("k", lambda: calls.append(1) or "v")
    cache.build_later("k", lambda: calls.append(1) or "v")  # already queued
    _wait_for_background()
    cache.build_later("k", lambda: calls.append(1) or "v")  # already built

    assert cache.peek("k") == "v"
    assert len(calls) == 1


def test_build_later_caches_a_failure():
    cache = SharedLRU(4)

    def boom():
        msg = "bad file"
        raise OSError(msg)

    cache.build_later("k", boom)
    _wait_for_background()

    assert cache.peek("k") == {"error": "OSError: bad file"}
