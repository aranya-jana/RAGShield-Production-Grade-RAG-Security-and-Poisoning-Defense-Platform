import pytest

from src.auth import (
    AuthenticationError,
    AuthorizationError,
    PasswordHasher,
    RBAC,
    TokenManager,
    TokenValidationError,
    UserStore,
    safe_principal_metadata,
    safe_user_metadata,
)


def test_password_hash_is_salted_and_verifies():
    hasher = PasswordHasher()
    password_hash_1, salt_1 = hasher.hash_password("CorrectHorseBattery!")
    password_hash_2, salt_2 = hasher.hash_password("CorrectHorseBattery!")

    assert password_hash_1 != password_hash_2
    assert salt_1 != salt_2
    assert hasher.verify_password(
        "CorrectHorseBattery!",
        password_hash_1,
        salt_1,
    )
    assert not hasher.verify_password(
        "wrong-password",
        password_hash_1,
        salt_1,
    )


def test_short_password_is_rejected():
    with pytest.raises(ValueError):
        PasswordHasher().hash_password("short")


def test_user_authentication():
    store = UserStore()
    user = store.create_user(
        "Alice",
        "CorrectHorseBattery!",
        roles=("user",),
        document_scopes=("doc-1",),
    )

    assert user.username == "alice"
    authenticated = store.authenticate(
        "ALICE",
        "CorrectHorseBattery!",
    )
    assert authenticated.username == "alice"

    with pytest.raises(AuthenticationError):
        store.authenticate("alice", "wrong-password")


def test_disabled_user_cannot_authenticate():
    store = UserStore()
    store.create_user(
        "disabled",
        "CorrectHorseBattery!",
        roles=("user",),
        disabled=True,
    )

    with pytest.raises(AuthenticationError):
        store.authenticate(
            "disabled",
            "CorrectHorseBattery!",
        )


def test_duplicate_user_is_rejected():
    store = UserStore()
    store.create_user("alice", "CorrectHorseBattery!")

    with pytest.raises(ValueError):
        store.create_user("alice", "AnotherCorrectPassword!")


def test_token_round_trip():
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
        roles=("security_analyst",),
        document_scopes=("doc-1", "doc-2"),
    )
    tokens = TokenManager(
        secret="x" * 48,
        ttl_seconds=3600,
    )

    token = tokens.issue(user, now=1_000)
    principal = tokens.validate(token, now=1_001)

    assert principal.username == "alice"
    assert principal.roles == frozenset({"security_analyst"})
    assert principal.document_scopes == frozenset({"doc-1", "doc-2"})


def test_tampered_token_is_rejected():
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
    )
    tokens = TokenManager(secret="x" * 48)
    token = tokens.issue(user, now=1_000)

    parts = token.split(".")
    parts[1] = parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B")
    tampered = ".".join(parts)

    with pytest.raises(TokenValidationError):
        tokens.validate(tampered, now=1_001)


def test_expired_token_is_rejected():
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
    )
    tokens = TokenManager(
        secret="x" * 48,
        ttl_seconds=60,
    )
    token = tokens.issue(user, now=1_000)

    with pytest.raises(TokenValidationError):
        tokens.validate(token, now=1_061)


def test_future_token_is_rejected():
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
    )
    tokens = TokenManager(secret="x" * 48)
    token = tokens.issue(user, now=2_000)

    with pytest.raises(TokenValidationError):
        tokens.validate(token, now=1_000)


def make_principal(roles=("user",), scopes=("*",)):
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
        roles=roles,
        document_scopes=scopes,
    )
    token_manager = TokenManager(secret="x" * 48)
    token = token_manager.issue(user, now=1_000)
    return token_manager.validate(token, now=1_001)


def test_rbac_user_permissions():
    principal = make_principal(("user",))
    rbac = RBAC()

    rbac.require_permission(principal, "query")
    rbac.require_permission(principal, "read_audit")

    with pytest.raises(AuthorizationError):
        rbac.require_permission(principal, "run_red_team")


def test_rbac_security_analyst_permissions():
    principal = make_principal(("security_analyst",))
    rbac = RBAC()

    rbac.require_permission(principal, "run_red_team")

    with pytest.raises(AuthorizationError):
        rbac.require_permission(principal, "manage_users")


def test_rbac_admin_permissions():
    principal = make_principal(("admin",))
    rbac = RBAC()

    rbac.require_permission(principal, "manage_users")
    rbac.require_permission(principal, "manage_documents")
    rbac.require_permission(principal, "reset_corpus")


def test_document_scope_allows_specific_document():
    principal = make_principal(
        ("user",),
        ("doc-1",),
    )
    rbac = RBAC()

    rbac.require_document_access(principal, "doc-1")

    with pytest.raises(AuthorizationError):
        rbac.require_document_access(principal, "doc-2")


def test_admin_bypasses_document_scope():
    principal = make_principal(
        ("admin",),
        ("doc-1",),
    )
    RBAC().require_document_access(principal, "secret-doc")


def test_wildcard_scope_allows_documents():
    principal = make_principal(
        ("user",),
        ("*",),
    )
    RBAC().require_document_access(principal, "any-doc")


def test_safe_metadata_contains_no_password_or_token():
    store = UserStore()
    user = store.create_user(
        "alice",
        "CorrectHorseBattery!",
        roles=("user",),
    )

    safe_user = safe_user_metadata(user)
    assert "password_hash" not in safe_user
    assert "password_salt" not in safe_user

    principal = make_principal()
    safe_principal = safe_principal_metadata(principal)
    assert "token" not in safe_principal
