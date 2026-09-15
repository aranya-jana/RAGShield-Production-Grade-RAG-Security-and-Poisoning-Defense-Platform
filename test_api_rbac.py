"""
API RBAC integration tests for RAGShield.

These tests verify:
- Public endpoints remain accessible without authentication.
- Protected endpoints reject missing authentication.
- Invalid bearer tokens are rejected.
- Normal users can query and read audit data.
- Normal users cannot run red-team attacks.
- Normal users cannot manage/setup the corpus.
- Security analysts can run red-team attacks.
- Administrators can clear the audit log.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.auth import (
    RBAC,
    TokenManager,
    UserStore,
)


@pytest.fixture()
def auth_context(monkeypatch):
    """
    Create isolated authentication/RBAC state for each test.

    UserStore.create_user() accepts roles directly. The returned User
    objects are immutable because their roles are stored as frozensets,
    so roles must be supplied during user creation.
    """
    user_store = UserStore()

    user_store.create_user(
        username="user1",
        password="StrongPassword123!",
        roles=("user",),
        document_scopes=("*",),
    )

    user_store.create_user(
        username="analyst1",
        password="StrongPassword123!",
        roles=("security_analyst",),
        document_scopes=("*",),
    )

    user_store.create_user(
        username="admin1",
        password="StrongPassword123!",
        roles=("admin",),
        document_scopes=("*",),
    )

    token_manager = TokenManager(
    secret="test-ragshield-rbac-secret-32-bytes",
    )

    rbac = RBAC()

    # Replace the API module's authentication objects with isolated
    # test instances.
    monkeypatch.setattr(
        "src.api.user_store",
        user_store,
    )

    monkeypatch.setattr(
        "src.api.token_manager",
        token_manager,
    )

    monkeypatch.setattr(
        "src.api.rbac",
        rbac,
    )

    return {
        "user_store": user_store,
        "token_manager": token_manager,
        "rbac": rbac,
    }


@pytest.fixture()
def client():
    """Create a FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def _token(context, username: str, role: str) -> str:
    """
    Issue a token for a test user.

    TokenManager.issue() accepts the complete User object.
    """
    user = context["user_store"].get_user(username)

    if user is None:
        raise AssertionError(
            f"Test user '{username}' was not created."
        )

    if role not in user.roles:
        raise AssertionError(
            f"Test user '{username}' does not have expected role "
            f"'{role}'. Actual roles: {sorted(user.roles)}"
        )

    return context["token_manager"].issue(user)


def _auth_header(token: str) -> dict:
    """Return an HTTP Bearer authorization header."""
    return {
        "Authorization": f"Bearer {token}",
    }


def test_health_endpoint_is_public(client):
    """The health endpoint must not require authentication."""
    response = client.get("/health")

    assert response.status_code == 200


def test_query_requires_authentication(client):
    """Unauthenticated users must not access the query endpoint."""
    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
        },
    )

    assert response.status_code == 401


def test_audit_requires_authentication(client):
    """Unauthenticated users must not read audit data."""
    response = client.get("/audit")

    assert response.status_code == 401


def test_clear_audit_requires_authentication(client):
    """Unauthenticated users must not clear audit data."""
    response = client.delete("/audit")

    assert response.status_code == 401


def test_invalid_token_is_rejected(client):
    """Malformed bearer tokens must be rejected."""
    response = client.post(
        "/query",
        headers={
            "Authorization": "Bearer invalid-token",
        },
        json={
            "query": "What is RAG?",
        },
    )

    assert response.status_code == 401


def test_malformed_authorization_header_is_rejected(client):
    """Malformed authorization headers must be rejected."""
    response = client.post(
        "/query",
        headers={
            "Authorization": "NotBearer some-token",
        },
        json={
            "query": "What is RAG?",
        },
    )

    assert response.status_code == 401


def test_user_can_query(client, auth_context):
    """A normal user should have query permission."""
    token = _token(
        auth_context,
        "user1",
        "user",
    )

    response = client.post(
        "/query",
        headers=_auth_header(token),
        json={
            "query": "What is RAG?",
        },
    )

    # The request must not be rejected by authentication/RBAC.
    assert response.status_code not in {401, 403}


def test_user_can_read_audit(client, auth_context):
    """A normal user should have read_audit permission."""
    token = _token(
        auth_context,
        "user1",
        "user",
    )

    response = client.get(
        "/audit",
        headers=_auth_header(token),
    )

    assert response.status_code == 200


def test_user_cannot_run_red_team_attack(client, auth_context):
    """A normal user must not have red-team execution permission."""
    token = _token(
        auth_context,
        "user1",
        "user",
    )

    response = client.post(
        "/attack",
        headers=_auth_header(token),
        json={
            "attack_id": "PI-001",
        },
    )

    assert response.status_code == 403


def test_user_cannot_setup_corpus(client, auth_context):
    """A normal user must not manage the document corpus."""
    token = _token(
        auth_context,
        "user1",
        "user",
    )

    response = client.post(
        "/setup",
        headers=_auth_header(token),
        json={},
    )

    assert response.status_code == 403


def test_security_analyst_can_run_red_team_attack(
    client,
    auth_context,
):
    """A security analyst should be allowed to run red-team attacks."""
    token = _token(
        auth_context,
        "analyst1",
        "security_analyst",
    )

    response = client.post(
        "/attack",
        headers=_auth_header(token),
        json={
            "attack_id": "PI-001",
        },
    )

    # Authentication/RBAC must allow the request through.
    # The endpoint may still produce an application-level response
    # depending on the attack/environment.
    assert response.status_code not in {401, 403}


def test_admin_can_clear_audit(
    client,
    auth_context,
):
    """An administrator should be allowed to clear audit data."""
    token = _token(
        auth_context,
        "admin1",
        "admin",
    )

    response = client.delete(
        "/audit",
        headers=_auth_header(token),
    )

    assert response.status_code == 200