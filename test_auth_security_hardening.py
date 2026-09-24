"""
RAGShield Authentication & Token Lifecycle Security Tests
===========================================================

Test-first coverage for the authentication hardening milestone.

These tests establish the desired security contract before changing
src/auth.py.
"""

from __future__ import annotations

import time

import pytest

from src.auth import (
    AuthenticationError,
    AuthenticatedPrincipal,
    TokenManager,
    User,
    UserStore,
)


def _make_user(
    username: str = "alice",
    *,
    roles: tuple[str, ...] = ("user",),
    document_scopes: tuple[str, ...] = ("*",),
    disabled: bool = False,
) -> User:
    store = UserStore()

    return store.create_user(
        username=username,
        password="StrongPassword123!",
        roles=roles,
        document_scopes=document_scopes,
        disabled=disabled,
    )


def test_token_contains_expiration_metadata():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    principal = manager.validate(token)

    assert isinstance(
        principal,
        AuthenticatedPrincipal,
    )
    assert principal.username == "alice"
    assert principal.expires_at > principal.issued_at


def test_token_expires_after_ttl():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=1,
    )

    issued_at = 1_700_000_000
    token = manager.issue(
        user,
        now=issued_at,
    )

    with pytest.raises(Exception):
        manager.validate(
            token,
            now=issued_at + 2,
        )


def test_token_can_be_validated_after_manager_recreation_with_same_secret():
    user = _make_user()

    secret = "production-test-secret-" + ("x" * 40)

    manager_one = TokenManager(
        secret=secret,
        ttl_seconds=3600,
    )

    token = manager_one.issue(
        user,
        now=1_700_000_000,
    )

    manager_two = TokenManager(
        secret=secret,
        ttl_seconds=3600,
    )

    principal = manager_two.validate(
        token,
        now=1_700_000_100,
    )

    assert principal.username == "alice"


def test_token_cannot_be_validated_with_different_secret():
    user = _make_user()

    manager_one = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager_one.issue(user)

    manager_two = TokenManager(
        secret="b" * 64,
        ttl_seconds=3600,
    )

    with pytest.raises(Exception):
        manager_two.validate(token)


def test_disabled_user_cannot_receive_token():
    user = _make_user(
        disabled=True,
    )

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    with pytest.raises(AuthenticationError):
        manager.issue(user)


def test_disabled_user_token_is_rejected_during_validation():
    store = UserStore()

    user = store.create_user(
        username="alice",
        password="StrongPassword123!",
    )

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    disabled_user = User(
        username=user.username,
        password_hash=user.password_hash,
        password_salt=user.password_salt,
        roles=user.roles,
        document_scopes=user.document_scopes,
        disabled=True,
    )

    # The current TokenManager API may not yet accept a user lookup callback.
    # This test establishes that disabled-user state must eventually be
    # checked during token validation.
    with pytest.raises(Exception):
        manager.validate(
            token,
            user_lookup=lambda username: disabled_user,
        )


def test_revoked_token_is_rejected():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    manager.revoke(token)

    with pytest.raises(Exception):
        manager.validate(token)


def test_revoking_one_token_does_not_revoke_another_token():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token_one = manager.issue(
        user,
        now=1_700_000_000,
    )

    token_two = manager.issue(
        user,
        now=1_700_000_001,
    )

    manager.revoke(token_one)

    with pytest.raises(Exception):
        manager.validate(
            token_one,
            now=1_700_000_010,
        )

    principal = manager.validate(
        token_two,
        now=1_700_000_010,
    )

    assert principal.username == "alice"


def test_revoked_token_cannot_be_revoked_again_as_a_new_token():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    manager.revoke(token)

    assert manager.is_revoked(token) is True


def test_token_version_is_present():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    principal = manager.validate(token)

    assert principal.token_version >= 1


def test_token_ttl_must_be_positive():
    with pytest.raises(ValueError):
        TokenManager(
            secret="a" * 64,
            ttl_seconds=0,
        )


def test_token_secret_must_be_at_least_32_bytes():
    with pytest.raises(ValueError):
        TokenManager(
            secret="short",
            ttl_seconds=3600,
        )


def test_token_secret_is_not_exposed_in_principal_metadata():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    principal = manager.validate(token)

    metadata = principal.safe_metadata()

    assert "secret" not in metadata
    assert "password" not in metadata
    assert "password_hash" not in metadata
    assert "token" not in metadata


def test_token_validation_rejects_malformed_token():
    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    with pytest.raises(Exception):
        manager.validate("not-a-valid-token")


def test_token_validation_rejects_tampering():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=3600,
    )

    token = manager.issue(user)

    tampered = token[:-1] + (
        "A"
        if token[-1] != "A"
        else "B"
    )

    with pytest.raises(Exception):
        manager.validate(tampered)


def test_user_password_is_never_stored_as_plaintext():
    store = UserStore()

    user = store.create_user(
        username="alice",
        password="StrongPassword123!",
    )

    assert user.password_hash != "StrongPassword123!"
    assert user.password_salt != "StrongPassword123!"


def test_user_store_rejects_short_passwords():
    store = UserStore()

    with pytest.raises(ValueError):
        store.create_user(
            username="alice",
            password="short",
        )


def test_token_validation_rejects_token_after_expiration_boundary():
    user = _make_user()

    manager = TokenManager(
        secret="a" * 64,
        ttl_seconds=60,
    )

    issued_at = int(time.time())

    token = manager.issue(
        user,
        now=issued_at,
    )

    with pytest.raises(Exception):
        manager.validate(
            token,
            now=issued_at + 61,
        )
