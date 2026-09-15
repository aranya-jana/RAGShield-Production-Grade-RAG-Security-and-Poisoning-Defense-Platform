"""
API-level tests for RAGShield rate limiting.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api import (
    RATE_LIMITS,
    enforce_rate_limit,
    rate_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """
    Prevent rate-limit state from leaking between tests.
    """

    rate_limiter.reset()

    yield

    rate_limiter.reset()


def _principal(username: str = "test-user"):
    """
    Minimal principal compatible with the rate-limit dependency.
    """

    class Principal:
        pass

    principal = Principal()
    principal.username = username

    return principal


def test_rate_limit_configuration_exists():
    assert "query" in RATE_LIMITS
    assert "attack" in RATE_LIMITS
    assert "setup" in RATE_LIMITS

    assert RATE_LIMITS["query"]["limit"] == 10
    assert RATE_LIMITS["attack"]["limit"] == 20
    assert RATE_LIMITS["setup"]["limit"] == 5


def test_unknown_rate_limit_scope_is_rejected():
    with pytest.raises(ValueError):
        enforce_rate_limit(
            "does-not-exist"
        )


def test_query_rate_limit_blocks_after_limit():
    dependency = enforce_rate_limit(
        "query"
    )

    principal = _principal()

    for _ in range(
        RATE_LIMITS["query"]["limit"]
    ):
        result = dependency(
            principal=principal
        )

        assert result is principal

    with pytest.raises(HTTPException) as exc_info:
        dependency(
            principal=principal
        )

    exc = exc_info.value

    assert exc.status_code == 429
    assert (
        exc.detail
        == "Rate limit exceeded. Please retry later."
    )

    assert (
        exc.headers["X-RateLimit-Limit"]
        == "10"
    )

    assert (
        exc.headers["X-RateLimit-Remaining"]
        == "0"
    )

    assert (
        int(exc.headers["Retry-After"])
        > 0
    )


def test_attack_rate_limit_is_independent():
    query_dependency = enforce_rate_limit(
        "query"
    )

    attack_dependency = enforce_rate_limit(
        "attack"
    )

    principal = _principal()

    for _ in range(
        RATE_LIMITS["query"]["limit"]
    ):
        query_dependency(
            principal=principal
        )

    with pytest.raises(HTTPException):
        query_dependency(
            principal=principal
        )

    # The attack scope has its own bucket.
    result = attack_dependency(
        principal=principal
    )

    assert result is principal


def test_different_users_have_independent_limits():
    dependency = enforce_rate_limit(
        "query"
    )

    user_a = _principal(
        "user-a"
    )

    user_b = _principal(
        "user-b"
    )

    for _ in range(
        RATE_LIMITS["query"]["limit"]
    ):
        dependency(
            principal=user_a
        )

    with pytest.raises(HTTPException):
        dependency(
            principal=user_a
        )

    # user-b has a separate bucket.
    result = dependency(
        principal=user_b
    )

    assert result is user_b


def test_missing_username_is_rejected():
    dependency = enforce_rate_limit(
        "query"
    )

    class Principal:
        username = None

    with pytest.raises(HTTPException) as exc_info:
        dependency(
            principal=Principal()
        )

    assert exc_info.value.status_code == 401


def test_rate_limiter_does_not_store_request_content():
    dependency = enforce_rate_limit(
        "query"
    )

    principal = _principal(
        "security-user"
    )

    dependency(
        principal=principal
    )

    # Only the generated user/scope bucket should exist.
    assert rate_limiter.size() == 1

    # Verify no request body or token-like content is present.
    for key in rate_limiter._buckets:
        assert "security-user" in key
        assert "password" not in key.lower()
        assert "token" not in key.lower()
        assert "query=" not in key.lower()


def test_setup_has_stricter_limit_than_query():
    assert (
        RATE_LIMITS["setup"]["limit"]
        < RATE_LIMITS["query"]["limit"]
    )


def test_rate_limit_dependency_returns_authenticated_principal():
    dependency = enforce_rate_limit(
        "query"
    )

    principal = _principal()

    result = dependency(
        principal=principal
    )

    assert result is principal