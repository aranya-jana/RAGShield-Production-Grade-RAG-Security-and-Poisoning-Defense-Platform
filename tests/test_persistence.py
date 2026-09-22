from pathlib import Path

from src.auth import TokenManager, User, UserStore
from src.database import Database


def test_user_store_persists_users(tmp_path: Path) -> None:
    database_path = tmp_path / "users.db"

    store_one = UserStore(
        database=Database(str(database_path)),
    )

    created = store_one.create_user(
        "PersistTest",
        "TestPassword123!",
        roles=("user", "security_analyst"),
        document_scopes=("docs/a", "docs/b"),
    )

    store_two = UserStore(
        database=Database(str(database_path)),
    )

    authenticated = store_two.authenticate(
        "persisttest",
        "TestPassword123!",
    )

    assert authenticated == created
    assert authenticated.username == "persisttest"
    assert authenticated.roles == frozenset(
        {"user", "security_analyst"}
    )
    assert authenticated.document_scopes == frozenset(
        {"docs/a", "docs/b"}
    )
    assert authenticated.disabled is False

    store_two.disable_user("persisttest")

    store_three = UserStore(
        database=Database(str(database_path)),
    )

    assert store_three.get_user("persisttest").disabled is True


def test_token_revocation_persists_across_managers(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "tokens.db"
    secret = "A" * 32

    database_one = Database(str(database_path))
    manager_one = TokenManager(
        secret=secret,
        ttl_seconds=3600,
        database=database_one,
    )

    user = User(
        username="token-test",
        password_hash="hash",
        password_salt="salt",
    )

    token = manager_one.issue(user)
    manager_one.revoke(token)

    database_two = Database(str(database_path))
    manager_two = TokenManager(
        secret=secret,
        ttl_seconds=3600,
        database=database_two,
    )

    assert manager_one.is_revoked(token) is True
    assert manager_two.is_revoked(token) is True


def test_token_manager_requires_configured_secret_when_production_mode(monkeypatch) -> None:
    monkeypatch.delenv("RAGSHIELD_AUTH_SECRET", raising=False)
    monkeypatch.setenv("RAGSHIELD_DEV_ADMIN_PROVISIONING", "false")

    try:
        TokenManager()
    except ValueError as exc:
        assert "RAGSHIELD_AUTH_SECRET" in str(exc)
    else:
        raise AssertionError("TokenManager should require a configured auth secret in production mode")
