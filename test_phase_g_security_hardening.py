"""
Phase G — production security hardening regression tests.

These tests cover the backend behaviors hardened in Phase G:
- current account state is re-checked when bearer tokens are validated
- login attempts are throttled before PBKDF2 password verification
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from src import api
from src.auth import TokenManager, UserStore
from src.rate_limiter import RateLimiter


def _request_with_client(host: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/auth/login",
        "headers": [],
        "client": (host, 12345),
        "server": ("localhost", 8000),
        "scheme": "http",
    }
    return Request(scope)


def test_api_token_validation_rechecks_current_user_state(monkeypatch):
    """A token must stop working after its account is disabled."""

    store = UserStore()
    user = store.create_user(
        username="phase-g-user",
        password="CorrectHorseBatteryStaple!",
        roles=("user",),
        document_scopes=("*",),
    )
    manager = TokenManager(
        secret="phase-g-test-secret-" + "x" * 40,
    )
    token = manager.issue(user, now=int(time.time()))

    monkeypatch.setattr(api, "user_store", store)
    monkeypatch.setattr(api, "token_manager", manager)

    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials=token,
    )

    principal = api.get_current_principal(credentials)
    assert principal.username == "phase-g-user"

    store.disable_user("phase-g-user")

    with pytest.raises(HTTPException) as exc_info:
        api.get_current_principal(credentials)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired authentication token."


def test_login_rate_limit_blocks_before_repeated_authentication_attempts(monkeypatch):
    """The unauthenticated login throttle must return 429 after the limit."""

    limiter = RateLimiter(
        max_clients=100,
        cleanup_interval_seconds=60,
    )

    audit_events = []
    monkeypatch.setattr(api, "rate_limiter", limiter)
    monkeypatch.setattr(
        api,
        "add_audit_event",
        lambda **kwargs: audit_events.append(kwargs),
    )

    original_ip_limit = api.RATE_LIMITS["login_ip"]
    original_username_limit = api.RATE_LIMITS["login_username"]

    api.RATE_LIMITS["login_ip"] = {
        "limit": 1,
        "window_seconds": 60,
    }
    api.RATE_LIMITS["login_username"] = {
        "limit": 10,
        "window_seconds": 60,
    }

    try:
        request = _request_with_client("192.0.2.10")

        api._enforce_login_rate_limits(
            request,
            "phase-g-user",
        )

        with pytest.raises(HTTPException) as exc_info:
            api._enforce_login_rate_limits(
                request,
                "phase-g-user",
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"]
        assert audit_events
        assert audit_events[-1]["event_type"] == "RATE LIMIT VIOLATION"
        assert audit_events[-1]["telemetry_metadata"]["scope"] == "login_ip"
    finally:
        api.RATE_LIMITS["login_ip"] = original_ip_limit
        api.RATE_LIMITS["login_username"] = original_username_limit
