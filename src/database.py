"""RAGShield persistent SQLite database layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DATABASE_PATH = "./data/ragshield.db"


class Database:
    """Small SQLite persistence layer for RAGShield application state."""

    def __init__(
        self,
        path: str = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with foreign-key enforcement enabled."""
        connection = sqlite3.connect(
            self.path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        return connection

    def initialize(self) -> None:
        """Create the persistent application schema if it does not exist."""
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    document_scopes TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    token_fingerprint TEXT PRIMARY KEY,
                    revoked_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_revoked_tokens_revoked_at
                    ON revoked_tokens(revoked_at);
                """
            )

    def execute(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> None:
        """Execute one write statement and commit it."""
        with self.connect() as connection:
            connection.execute(
                sql,
                parameters,
            )

    def read_one(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> sqlite3.Row | None:
        """Read one database row."""
        with self.connect() as connection:
            return connection.execute(
                sql,
                parameters,
            ).fetchone()

    def read_all(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> list[sqlite3.Row]:
        """Read all rows returned by a query."""
        with self.connect() as connection:
            return connection.execute(
                sql,
                parameters,
            ).fetchall()

    def insert_user(
        self,
        username: str,
        password_hash: str,
        password_salt: str,
        roles: str,
        document_scopes: str,
        disabled: bool,
    ) -> None:
        """Persist a new user account."""
        self.execute(
            """
            INSERT INTO users (
                username,
                password_hash,
                password_salt,
                roles,
                document_scopes,
                disabled
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                password_hash,
                password_salt,
                roles,
                document_scopes,
                int(disabled),
            ),
        )

    def get_user(
        self,
        username: str,
    ) -> sqlite3.Row | None:
        """Return one persisted user by normalized username."""
        return self.read_one(
            """
            SELECT
                username,
                password_hash,
                password_salt,
                roles,
                document_scopes,
                disabled
            FROM users
            WHERE username = ?
            """,
            (username,),
        )

    def list_users(self) -> list[sqlite3.Row]:
        """Return all persisted users."""
        return self.read_all(
            """
            SELECT
                username,
                password_hash,
                password_salt,
                roles,
                document_scopes,
                disabled
            FROM users
            ORDER BY username
            """
        )

    def set_user_disabled(
        self,
        username: str,
        disabled: bool,
    ) -> bool:
        """Set the disabled state for an existing user."""
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET disabled = ?
                WHERE username = ?
                """,
                (
                    int(disabled),
                    username,
                ),
            )
            return cursor.rowcount > 0

    def revoke_token(
        self,
        token_fingerprint: str,
        revoked_at: int,
    ) -> None:
        """Persist a token fingerprint without storing the bearer token."""
        self.execute(
            """
            INSERT OR IGNORE INTO revoked_tokens (
                token_fingerprint,
                revoked_at
            )
            VALUES (?, ?)
            """,
            (
                token_fingerprint,
                revoked_at,
            ),
        )

    def is_token_revoked(
        self,
        token_fingerprint: str,
    ) -> bool:
        """Return whether a token fingerprint is persistently revoked."""
        row = self.read_one(
            """
            SELECT 1
            FROM revoked_tokens
            WHERE token_fingerprint = ?
            LIMIT 1
            """,
            (token_fingerprint,),
        )
        return row is not None
