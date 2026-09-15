"""
RAGShield rate limiter tests.
"""

from __future__ import annotations

import threading

import pytest

from src.rate_limiter import RateLimiter


def test_first_request_is_allowed():
    limiter = RateLimiter()

    result = limiter.check(
        "client-a",
        "query",
        limit=3,
        window_seconds=60,
        now=100.0,
    )

    assert result.allowed is True
    assert result.limit == 3
    assert result.remaining == 2
    assert result.retry_after_seconds == 0


def test_limit_is_enforced():
    limiter = RateLimiter()

    for _ in range(3):

        result = limiter.check(
            "client-a",
            "query",
            limit=3,
            window_seconds=60,
            now=100.0,
        )

        assert result.allowed is True

    blocked = limiter.check(
        "client-a",
        "query",
        limit=3,
        window_seconds=60,
        now=100.0,
    )

    assert blocked.allowed is False
    assert blocked.limit == 3
    assert blocked.remaining == 0
    assert blocked.retry_after_seconds > 0


def test_window_resets():
    limiter = RateLimiter()

    for _ in range(3):

        limiter.check(
            "client-a",
            "query",
            limit=3,
            window_seconds=60,
            now=100.0,
        )

    blocked = limiter.check(
        "client-a",
        "query",
        limit=3,
        window_seconds=60,
        now=100.0,
    )

    assert blocked.allowed is False

    after_window = limiter.check(
        "client-a",
        "query",
        limit=3,
        window_seconds=60,
        now=160.0,
    )

    assert after_window.allowed is True
    assert after_window.remaining == 2


def test_clients_are_isolated():
    limiter = RateLimiter()

    for _ in range(2):

        limiter.check(
            "client-a",
            "query",
            limit=2,
            window_seconds=60,
            now=100.0,
        )

    blocked_a = limiter.check(
        "client-a",
        "query",
        limit=2,
        window_seconds=60,
        now=100.0,
    )

    allowed_b = limiter.check(
        "client-b",
        "query",
        limit=2,
        window_seconds=60,
        now=100.0,
    )

    assert blocked_a.allowed is False
    assert allowed_b.allowed is True


def test_scopes_are_isolated():
    limiter = RateLimiter()

    for _ in range(2):

        limiter.check(
            "client-a",
            "query",
            limit=2,
            window_seconds=60,
            now=100.0,
        )

    blocked_query = limiter.check(
        "client-a",
        "query",
        limit=2,
        window_seconds=60,
        now=100.0,
    )

    allowed_attack = limiter.check(
        "client-a",
        "attack",
        limit=2,
        window_seconds=60,
        now=100.0,
    )

    assert blocked_query.allowed is False
    assert allowed_attack.allowed is True


def test_reset_client():
    limiter = RateLimiter()

    limiter.check(
        "client-a",
        "query",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    blocked = limiter.check(
        "client-a",
        "query",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    assert blocked.allowed is False

    limiter.reset(
        client_key="client-a"
    )

    allowed = limiter.check(
        "client-a",
        "query",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    assert allowed.allowed is True


def test_reset_scope():
    limiter = RateLimiter()

    limiter.check(
        "client-a",
        "query",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    limiter.check(
        "client-a",
        "attack",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    limiter.reset(
        scope="query"
    )

    query_result = limiter.check(
        "client-a",
        "query",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    attack_result = limiter.check(
        "client-a",
        "attack",
        limit=1,
        window_seconds=60,
        now=100.0,
    )

    assert query_result.allowed is True
    assert attack_result.allowed is False


def test_capacity_is_bounded():
    limiter = RateLimiter(
        max_clients=2
    )

    limiter.check(
        "client-a",
        "query",
        limit=10,
        window_seconds=60,
        now=100.0,
    )

    limiter.check(
        "client-b",
        "query",
        limit=10,
        window_seconds=60,
        now=101.0,
    )

    limiter.check(
        "client-c",
        "query",
        limit=10,
        window_seconds=60,
        now=102.0,
    )

    assert limiter.size() <= 2


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        RateLimiter(
            max_clients=0
        )

    with pytest.raises(ValueError):
        RateLimiter(
            cleanup_interval_seconds=0
        )

    limiter = RateLimiter()

    with pytest.raises(ValueError):
        limiter.check(
            "client",
            "query",
            limit=0,
            window_seconds=60,
        )

    with pytest.raises(ValueError):
        limiter.check(
            "client",
            "query",
            limit=1,
            window_seconds=0,
        )

    with pytest.raises(ValueError):
        limiter.check(
            "",
            "query",
            limit=1,
            window_seconds=60,
        )

    with pytest.raises(ValueError):
        limiter.check(
            "client",
            "",
            limit=1,
            window_seconds=60,
        )


def test_concurrent_requests_are_thread_safe():
    limiter = RateLimiter()

    results = []

    results_lock = threading.Lock()

    def worker():

        result = limiter.check(
            "same-client",
            "query",
            limit=10,
            window_seconds=60,
        )

        with results_lock:
            results.append(
                result.allowed
            )

    threads = [
        threading.Thread(
            target=worker
        )
        for _ in range(50)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(results) == 50

    assert sum(results) == 10

    assert limiter.size() == 1