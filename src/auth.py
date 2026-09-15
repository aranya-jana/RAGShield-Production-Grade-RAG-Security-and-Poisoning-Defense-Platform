"""
RAGShield authentication and role-based access control.

This module provides a dependency-light security core for the FastAPI layer:
- PBKDF2 password hashing with per-user salts
- HMAC-SHA256 signed bearer tokens
- token expiry validation
- role-based permissions
- document-level authorization helpers
- safe audit metadata (never returns passwords or raw tokens)

The module intentionally has no FastAPI dependency so it can be unit-tested
independently and reused by API and non-API callers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Mapping, Optional


AUTH_VERSION = 1
DEFAULT_TOKEN_TTL_SECONDS = 3600
PBKDF2_ITERATIONS = 310_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 12


class AuthenticationError(ValueError):
    """Raised when authentication credentials are invalid."""


class AuthorizationError(PermissionError):
    """Raised when an authenticated principal lacks permission."""


class TokenValidationError(AuthenticationError):
    """Raised when a bearer token is invalid or expired."""


@dataclass(frozen=True)
class User:
    """Authenticated application user."""

    username: str
    password_hash: str
    password_salt: str
    roles: FrozenSet[str] = field(default_factory=frozenset)
    document_scopes: FrozenSet[str] = field(default_factory=frozenset)
    disabled: bool = False

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def can_access_document(self, document_id: str) -> bool:
        if "admin" in self.roles:
            return True
        if "*" in self.document_scopes:
            return True
        return document_id in self.document_scopes


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Identity extracted from a validated token."""

    username: str
    roles: FrozenSet[str]
    document_scopes: FrozenSet[str]
    issued_at: int
    expires_at: int

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def can_access_document(self, document_id: str) -> bool:
        if "admin" in self.roles:
            return True
        if "*" in self.document_scopes:
            return True
        return document_id in self.document_scopes


class PasswordHasher:
    """Local password hashing using PBKDF2-HMAC-SHA256."""

    def __init__(self, iterations: int = PBKDF2_ITERATIONS):
        if iterations < 100_000:
            raise ValueError("PBKDF2 iteration count is too low")
        self.iterations = iterations

    def hash_password(self, password: str) -> tuple[str, str]:
        self._validate_password(password)
        salt = secrets.token_bytes(SALT_BYTES)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
        )
        return (
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        )

    def verify_password(
        self,
        password: str,
        password_hash: str,
        password_salt: str,
    ) -> bool:
        if not isinstance(password, str):
            return False

        try:
            salt = self._decode(password_salt)
            expected = self._decode(password_hash)
        except (ValueError, TypeError):
            return False

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
        )
        return hmac.compare_digest(actual, expected)

    @staticmethod
    def _decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    @staticmethod
    def _validate_password(password: str) -> None:
        if not isinstance(password, str):
            raise ValueError("Password must be a string")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must contain at least {MIN_PASSWORD_LENGTH} characters"
            )


class UserStore:
    """In-memory user store suitable for the current local RAGShield deployment.

    A production multi-instance deployment should replace this with a database
    backed implementation while preserving this interface.
    """

    def __init__(self, hasher: Optional[PasswordHasher] = None):
        self.hasher = hasher or PasswordHasher()
        self._users: Dict[str, User] = {}

    def create_user(
        self,
        username: str,
        password: str,
        roles: Iterable[str] = ("user",),
        document_scopes: Iterable[str] = ("*",),
        disabled: bool = False,
    ) -> User:
        username = self._normalize_username(username)
        if username in self._users:
            raise ValueError("User already exists")

        normalized_roles = frozenset(
            str(role).strip().lower()
            for role in roles
            if str(role).strip()
        )
        if not normalized_roles:
            raise ValueError("At least one role is required")

        password_hash, password_salt = self.hasher.hash_password(password)

        user = User(
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
            roles=normalized_roles,
            document_scopes=frozenset(
                str(scope).strip()
                for scope in document_scopes
                if str(scope).strip()
            ),
            disabled=bool(disabled),
        )
        self._users[username] = user
        return user

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(self._normalize_username(username))

    def authenticate(self, username: str, password: str) -> User:
        username = self._normalize_username(username)
        user = self._users.get(username)

        # Do not reveal whether the username exists.
        if user is None or user.disabled:
            raise AuthenticationError("Invalid username or password")

        if not self.hasher.verify_password(
            password,
            user.password_hash,
            user.password_salt,
        ):
            raise AuthenticationError("Invalid username or password")

        return user

    @staticmethod
    def _normalize_username(username: str) -> str:
        if not isinstance(username, str):
            raise ValueError("Username must be a string")
        username = username.strip().lower()
        if not username or len(username) > 128:
            raise ValueError("Invalid username")
        return username


class TokenManager:
    """Creates and validates compact HMAC-SHA256 bearer tokens."""

    def __init__(
        self,
        secret: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
    ):
        secret = secret or os.getenv("RAGSHIELD_AUTH_SECRET")
        if not secret:
            # Explicitly allow development use, but never silently use a
            # predictable/default secret.
            secret = secrets.token_urlsafe(48)

        if len(secret.encode("utf-8")) < 32:
            raise ValueError(
                "RAGSHIELD_AUTH_SECRET must contain at least 32 bytes"
            )

        if ttl_seconds <= 0:
            raise ValueError("Token TTL must be positive")

        self._secret = secret.encode("utf-8")
        self.ttl_seconds = int(ttl_seconds)

    def issue(self, user: User, now: Optional[int] = None) -> str:
        issued_at = int(time.time() if now is None else now)
        expires_at = issued_at + self.ttl_seconds

        payload = {
            "v": AUTH_VERSION,
            "sub": user.username,
            "roles": sorted(user.roles),
            "scopes": sorted(user.document_scopes),
            "iat": issued_at,
            "exp": expires_at,
        }
        encoded_payload = self._encode_json(payload)
        signature = self._sign(encoded_payload)
        return f"rs1.{encoded_payload}.{signature}"

    def validate(
        self,
        token: str,
        now: Optional[int] = None,
    ) -> AuthenticatedPrincipal:
        if not isinstance(token, str):
            raise TokenValidationError("Invalid bearer token")

        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "rs1":
            raise TokenValidationError("Invalid bearer token")

        _, encoded_payload, supplied_signature = parts
        expected_signature = self._sign(encoded_payload)

        if not hmac.compare_digest(
            supplied_signature,
            expected_signature,
        ):
            raise TokenValidationError("Invalid bearer token")

        try:
            payload = self._decode_json(encoded_payload)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise TokenValidationError("Invalid bearer token") from None

        if payload.get("v") != AUTH_VERSION:
            raise TokenValidationError("Unsupported token version")

        username = payload.get("sub")
        roles = payload.get("roles")
        scopes = payload.get("scopes")
        issued_at = payload.get("iat")
        expires_at = payload.get("exp")

        if (
            not isinstance(username, str)
            or not isinstance(roles, list)
            or not isinstance(scopes, list)
            or not isinstance(issued_at, int)
            or not isinstance(expires_at, int)
        ):
            raise TokenValidationError("Invalid bearer token")

        current_time = int(time.time() if now is None else now)

        if expires_at <= current_time:
            raise TokenValidationError("Bearer token expired")

        if issued_at > current_time + 30:
            raise TokenValidationError("Bearer token issued in the future")

        return AuthenticatedPrincipal(
            username=username,
            roles=frozenset(str(role) for role in roles),
            document_scopes=frozenset(str(scope) for scope in scopes),
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def _sign(self, encoded_payload: str) -> str:
        digest = hmac.new(
            self._secret,
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return self._b64(digest)

    @staticmethod
    def _encode_json(payload: Mapping[str, object]) -> str:
        raw = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return TokenManager._b64(raw)

    @staticmethod
    def _decode_json(encoded: str) -> dict:
        raw = base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        )
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Token payload must be an object")
        return value

    @staticmethod
    def _b64(raw: bytes) -> str:
        return (
            base64.urlsafe_b64encode(raw)
            .decode("ascii")
            .rstrip("=")
        )


class RBAC:
    """Central role/permission policy for RAGShield."""

    PERMISSIONS = {
        "user": frozenset(
            {
                "query",
                "read_audit",
            }
        ),
        "security_analyst": frozenset(
            {
                "query",
                "read_audit",
                "run_red_team",
                "read_security_events",
            }
        ),
        "admin": frozenset(
            {
                "query",
                "read_audit",
                "run_red_team",
                "read_security_events",
                "manage_users",
                "manage_documents",
                "reset_corpus",
            }
        ),
    }

    def permissions_for(self, principal: AuthenticatedPrincipal) -> FrozenSet[str]:
        permissions = set()
        for role in principal.roles:
            permissions.update(self.PERMISSIONS.get(role, ()))
        return frozenset(permissions)

    def require_permission(
        self,
        principal: AuthenticatedPrincipal,
        permission: str,
    ) -> None:
        if permission not in self.permissions_for(principal):
            raise AuthorizationError(
                f"Permission denied: {permission}"
            )

    def require_role(
        self,
        principal: AuthenticatedPrincipal,
        role: str,
    ) -> None:
        if role not in principal.roles:
            raise AuthorizationError(
                f"Role required: {role}"
            )

    def require_document_access(
        self,
        principal: AuthenticatedPrincipal,
        document_id: str,
    ) -> None:
        if not principal.can_access_document(document_id):
            raise AuthorizationError(
                "Document access denied"
            )


def safe_user_metadata(user: User) -> dict:
    """Return audit/UI-safe user information."""
    return {
        "username": user.username,
        "roles": sorted(user.roles),
        "document_scope_count": len(user.document_scopes),
        "disabled": user.disabled,
    }


def safe_principal_metadata(principal: AuthenticatedPrincipal) -> dict:
    """Return audit/UI-safe principal information."""
    return {
        "username": principal.username,
        "roles": sorted(principal.roles),
        "expires_at": principal.expires_at,
        "document_scope_count": len(principal.document_scopes),
    }
