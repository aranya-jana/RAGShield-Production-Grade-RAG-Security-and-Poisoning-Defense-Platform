"""
RAGShield Authentication, Token Lifecycle & RBAC
================================================

Production-oriented local authentication primitives for RAGShield.

Security goals:
- Salted password hashing using PBKDF2-HMAC-SHA256.
- Signed, versioned authentication tokens.
- Explicit token expiration.
- Configurable token signing secret.
- Individual token revocation.
- Disabled-user enforcement.
- Role-based access control.
- Document-scope authorization.
- Safe audit metadata without passwords, password hashes, secrets, or tokens.

This module intentionally has no external network dependency.

For local development, TokenManager may generate an ephemeral signing secret
when no configured secret is available. For production, set:

    RAGSHIELD_AUTH_SECRET

to a random secret containing at least 32 UTF-8 bytes.

Example:

    $env:RAGSHIELD_AUTH_SECRET = "<strong-random-secret>"
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Set,
)

from src.database import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_VERSION = 1
TOKEN_VERSION = 1

DEFAULT_TOKEN_TTL_SECONDS = 3600

PBKDF2_ITERATIONS = 310_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 12

MIN_TOKEN_SECRET_BYTES = 32

TOKEN_ALGORITHM = "HS256"

AUTH_SECRET_ENV = "RAGSHIELD_AUTH_SECRET"

def _get_bool_env(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthenticationError(Exception):
    """Raised when user authentication fails."""


class AuthorizationError(Exception):
    """Raised when an authenticated principal lacks permission."""


class TokenValidationError(AuthenticationError):
    """Raised when a token is malformed, invalid, expired, or revoked."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class User:
    """
    Stored user account.

    Password material is represented only by a PBKDF2 password hash and salt.
    """

    username: str
    password_hash: str
    password_salt: str
    roles: FrozenSet[str] = field(
        default_factory=frozenset
    )
    document_scopes: FrozenSet[str] = field(
        default_factory=frozenset
    )
    disabled: bool = False


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """
    Identity represented by a successfully validated authentication token.
    """

    username: str
    roles: FrozenSet[str]
    document_scopes: FrozenSet[str]
    issued_at: int = 0
    expires_at: int = 0
    token_version: int = TOKEN_VERSION
    token_id: str = ""

    def safe_metadata(self) -> Dict[str, object]:
        """
        Return metadata safe for logs/API telemetry.

        Deliberately excludes:
        - token
        - password
        - password_hash
        - password_salt
        - signing secret
        """
        return {
            "username": self.username,
            "roles": sorted(self.roles),
            "document_scopes": sorted(
                self.document_scopes
            ),
            "issued_at": int(self.issued_at),
            "expires_at": int(self.expires_at),
            "token_version": int(self.token_version),
            "token_id": self.token_id,
        }


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class PasswordHasher:
    """PBKDF2-HMAC-SHA256 password hashing."""

    def __init__(
        self,
        iterations: int = PBKDF2_ITERATIONS,
        salt_bytes: int = SALT_BYTES,
        min_password_length: int = MIN_PASSWORD_LENGTH,
    ) -> None:
        if iterations < 100_000:
            raise ValueError(
                "PBKDF2 iterations must be at least 100000."
            )

        if salt_bytes < 16:
            raise ValueError(
                "Salt must be at least 16 bytes."
            )

        if min_password_length < 8:
            raise ValueError(
                "Minimum password length must be at least 8."
            )

        self.iterations = iterations
        self.salt_bytes = salt_bytes
        self.min_password_length = min_password_length

    def hash_password(
        self,
        password: str,
    ) -> tuple[str, str]:
        """Hash a password with a fresh random salt."""
        if not isinstance(password, str):
            raise TypeError(
                "password must be a string."
            )

        if len(password) < self.min_password_length:
            raise ValueError(
                "Password does not meet minimum length requirements."
            )

        salt = secrets.token_bytes(
            self.salt_bytes
        )

        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
        )

        return (
            base64.urlsafe_b64encode(
                password_hash
            ).decode("ascii"),
            base64.urlsafe_b64encode(
                salt
            ).decode("ascii"),
        )

    def verify_password(
        self,
        password: str,
        password_hash: str,
        password_salt: str,
    ) -> bool:
        """Verify a password using constant-time comparison."""
        if not isinstance(password, str):
            return False

        try:
            expected_hash = (
                base64.urlsafe_b64decode(
                    password_hash.encode("ascii")
                )
            )

            salt = (
                base64.urlsafe_b64decode(
                    password_salt.encode("ascii")
                )
            )

            actual_hash = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                self.iterations,
            )

            return hmac.compare_digest(
                actual_hash,
                expected_hash,
            )

        except (
            ValueError,
            TypeError,
            UnicodeError,
        ):
            return False


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


class UserStore:
    """User account store with optional persistent SQLite backing."""

    def __init__(
        self,
        password_hasher: Optional[PasswordHasher] = None,
        database: Optional[Database] = None,
    ) -> None:
        self.password_hasher = (
            password_hasher
            or PasswordHasher()
        )
        self.database = database

        self._users: Dict[str, User] = {}

    @staticmethod
    def _normalize_username(
        username: str,
    ) -> str:
        if not isinstance(username, str):
            raise TypeError(
                "username must be a string."
            )

        normalized = username.strip().lower()

        if not normalized:
            raise ValueError(
                "username cannot be empty."
            )

        return normalized

    @staticmethod
    def _normalize_roles(
        roles: Iterable[str],
    ) -> FrozenSet[str]:
        normalized = frozenset(
            str(role).strip().lower()
            for role in roles
            if str(role).strip()
        )

        return normalized

    @staticmethod
    def _normalize_scopes(
        scopes: Iterable[str],
    ) -> FrozenSet[str]:
        normalized = frozenset(
            str(scope).strip()
            for scope in scopes
            if str(scope).strip()
        )

        return normalized

    @staticmethod
    def _serialize_values(
        values: Iterable[str],
    ) -> str:
        return json.dumps(
            sorted(
                str(value)
                for value in values
            ),
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize_values(
        value: str,
    ) -> tuple[str, ...]:
        parsed = json.loads(value)

        if not isinstance(parsed, list):
            raise ValueError(
                "Persisted user collection must be a JSON array."
            )

        return tuple(
            str(item)
            for item in parsed
        )

    @classmethod
    def _user_from_row(
        cls,
        row: Mapping[str, object],
    ) -> User:
        return User(
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            password_salt=str(row["password_salt"]),
            roles=cls._normalize_roles(
                cls._deserialize_values(
                    str(row["roles"])
                )
            ),
            document_scopes=cls._normalize_scopes(
                cls._deserialize_values(
                    str(row["document_scopes"])
                )
            ),
            disabled=bool(row["disabled"]),
        )

    def create_user(
        self,
        username: str,
        password: str,
        roles: Iterable[str] = ("user",),
        document_scopes: Iterable[str] = ("*",),
        disabled: bool = False,
    ) -> User:
        """Create a new local user account."""
        normalized_username = (
            self._normalize_username(
                username
            )
        )

        if self.database is not None:
            if self.database.get_user(
                normalized_username
            ) is not None:
                raise ValueError(
                    f"User already exists: {normalized_username}"
                )
        elif normalized_username in self._users:
            raise ValueError(
                f"User already exists: {normalized_username}"
            )

        password_hash, password_salt = (
            self.password_hasher.hash_password(
                password
            )
        )

        normalized_roles = (
            self._normalize_roles(
                roles
            )
        )

        normalized_scopes = (
            self._normalize_scopes(
                document_scopes
            )
        )

        user = User(
            username=normalized_username,
            password_hash=password_hash,
            password_salt=password_salt,
            roles=normalized_roles,
            document_scopes=normalized_scopes,
            disabled=bool(disabled),
        )

        if self.database is not None:
            self.database.insert_user(
                username=user.username,
                password_hash=user.password_hash,
                password_salt=user.password_salt,
                roles=self._serialize_values(
                    user.roles
                ),
                document_scopes=self._serialize_values(
                    user.document_scopes
                ),
                disabled=user.disabled,
            )
        else:
            self._users[
                normalized_username
            ] = user

        return user

    def get_user(
        self,
        username: str,
    ) -> User:
        """Return a user or raise AuthenticationError."""
        normalized_username = (
            self._normalize_username(
                username
            )
        )

        if self.database is not None:
            row = self.database.get_user(
                normalized_username
            )

            if row is None:
                raise AuthenticationError(
                    "Invalid username or password."
                )

            return self._user_from_row(row)

        user = self._users.get(
            normalized_username
        )

        if user is None:
            raise AuthenticationError(
                "Invalid username or password."
            )

        return user

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> User:
        """Authenticate credentials and enforce disabled-user policy."""
        user = self.get_user(username)

        if user.disabled:
            raise AuthenticationError(
                "User account is disabled."
            )

        if not self.password_hasher.verify_password(
            password,
            user.password_hash,
            user.password_salt,
        ):
            raise AuthenticationError(
                "Invalid username or password."
            )

        return user

    def _set_disabled(
        self,
        username: str,
        disabled: bool,
    ) -> User:
        normalized_username = (
            self._normalize_username(
                username
            )
        )

        if self.database is not None:
            if not self.database.set_user_disabled(
                normalized_username,
                disabled,
            ):
                raise AuthenticationError(
                    "User not found."
                )

            return self.get_user(
                normalized_username
            )

        user = self._users.get(
            normalized_username
        )

        if user is None:
            raise AuthenticationError(
                "User not found."
            )

        updated_user = User(
            username=user.username,
            password_hash=user.password_hash,
            password_salt=user.password_salt,
            roles=user.roles,
            document_scopes=user.document_scopes,
            disabled=disabled,
        )

        self._users[
            normalized_username
        ] = updated_user

        return updated_user

    def disable_user(
        self,
        username: str,
    ) -> User:
        """Disable an existing user account."""
        return self._set_disabled(
            username,
            True,
        )

    def enable_user(
        self,
        username: str,
    ) -> User:
        """Enable an existing user account."""
        return self._set_disabled(
            username,
            False,
        )

    def all_users(self) -> tuple[User, ...]:
        """Return users without exposing mutable internal storage."""
        if self.database is not None:
            return tuple(
                self._user_from_row(row)
                for row in self.database.list_users()
            )

        return tuple(
            self._users.values()
        )


# ---------------------------------------------------------------------------
# Token manager
# ---------------------------------------------------------------------------


class TokenManager:
    """
    Signed authentication token manager.

    Token format:

        base64url(header).base64url(payload).base64url(signature)

    Header:

        {
            "alg": "HS256",
            "typ": "RAGShieldToken",
            "ver": 1
        }

    Payload contains:
        - username
        - roles
        - document scopes
        - issued-at
        - expiration
        - token id
        - token version

    For production, provide a stable secret via the constructor or
    RAGSHIELD_AUTH_SECRET. This allows tokens to remain valid across process
    restarts.

    Token revocation is persisted when a database is configured, allowing
    revoked tokens to remain revoked across process restarts and API workers
    that share the same database.
    """

    def __init__(
        self,
        secret: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
        database: Optional[Database] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                "Token TTL must be greater than zero."
            )

        configured_secret = (
            secret
            if secret is not None
            else os.getenv(
                AUTH_SECRET_ENV
            )
        )

        if configured_secret is None:
            if not _get_bool_env(
                "RAGSHIELD_DEV_ADMIN_PROVISIONING",
                True,
            ):
                raise ValueError(
                    "RAGSHIELD_AUTH_SECRET must be configured "
                    "when development admin provisioning is disabled."
                )

            # Local-development fallback. This deliberately does not provide
            # restart persistence; production deployments must configure
            # RAGSHIELD_AUTH_SECRET.
            configured_secret = (
                secrets.token_urlsafe(48)
            )

        if not isinstance(
            configured_secret,
            str,
        ):
            raise TypeError(
                "Token secret must be a string."
            )

        secret_bytes = (
            configured_secret.encode(
                "utf-8"
            )
        )

        if len(secret_bytes) < MIN_TOKEN_SECRET_BYTES:
            raise ValueError(
                "Token secret must contain at least "
                "32 UTF-8 bytes."
            )

        self._secret = secret_bytes
        self.ttl_seconds = int(
            ttl_seconds
        )

        self.database = database

        self._revoked_tokens: Set[str] = set()
        self._revocation_lock = (
            threading.RLock()
        )

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _b64encode(
        value: bytes,
    ) -> str:
        return (
            base64.urlsafe_b64encode(
                value
            )
            .decode("ascii")
            .rstrip("=")
        )

    @staticmethod
    def _b64decode(
        value: str,
    ) -> bytes:
        padding = "=" * (
            (-len(value)) % 4
        )

        return base64.urlsafe_b64decode(
            value + padding
        )

    @staticmethod
    def _canonical_json(
        value: Mapping[str, object],
    ) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _sign(
        self,
        signing_input: bytes,
    ) -> bytes:
        return hmac.new(
            self._secret,
            signing_input,
            hashlib.sha256,
        ).digest()

    def _signature_matches(
        self,
        signing_input: bytes,
        provided_signature: bytes,
    ) -> bool:
        expected_signature = (
            self._sign(
                signing_input
            )
        )

        return hmac.compare_digest(
            expected_signature,
            provided_signature,
        )

    # ------------------------------------------------------------------
    # Token issuance
    # ------------------------------------------------------------------

    def issue(
        self,
        user: User,
        now: Optional[int] = None,
    ) -> str:
        """
        Issue a signed token for an active user.

        Disabled users are rejected before token creation.
        """
        if not isinstance(
            user,
            User,
        ):
            raise TypeError(
                "user must be a User instance."
            )

        if user.disabled:
            raise AuthenticationError(
                "Disabled users cannot receive authentication tokens."
            )

        issued_at = (
            int(time.time())
            if now is None
            else int(now)
        )

        expires_at = (
            issued_at
            + self.ttl_seconds
        )

        token_id = secrets.token_urlsafe(
            24
        )

        header = {
            "alg": TOKEN_ALGORITHM,
            "typ": "RAGShieldToken",
            "ver": TOKEN_VERSION,
        }

        payload = {
            "auth_version": AUTH_VERSION,
            "token_version": TOKEN_VERSION,
            "username": user.username,
            "roles": sorted(
                user.roles
            ),
            "document_scopes": sorted(
                user.document_scopes
            ),
            "iat": issued_at,
            "exp": expires_at,
            "jti": token_id,
        }

        header_b64 = self._b64encode(
            self._canonical_json(
                header
            )
        )

        payload_b64 = self._b64encode(
            self._canonical_json(
                payload
            )
        )

        signing_input = (
            f"{header_b64}.{payload_b64}"
            .encode("ascii")
        )

        signature_b64 = self._b64encode(
            self._sign(
                signing_input
            )
        )

        return (
            f"{header_b64}."
            f"{payload_b64}."
            f"{signature_b64}"
        )

    # ------------------------------------------------------------------
    # Token revocation
    # ------------------------------------------------------------------

    def revoke(
        self,
        token: str,
    ) -> None:
        """
        Revoke one exact token.

        The token itself is stored only as an HMAC-derived identifier to avoid
        retaining the bearer token in cleartext in the revocation set.
        """
        self._validate_token_shape(
            token
        )

        token_fingerprint = self._token_fingerprint(
            token
        )

        with self._revocation_lock:
            self._revoked_tokens.add(
                token_fingerprint
            )

            if self.database is not None:
                self.database.revoke_token(
                    token_fingerprint,
                    int(time.time()),
                )

    def is_revoked(
        self,
        token: str,
    ) -> bool:
        """Return whether the supplied token has been revoked."""
        if not isinstance(
            token,
            str,
        ):
            return False

        token_fingerprint = (
            self._token_fingerprint(
                token
            )
        )

        with self._revocation_lock:
            if token_fingerprint in self._revoked_tokens:
                return True

            if self.database is not None:
                return self.database.is_token_revoked(
                    token_fingerprint
                )

            return False

    def _token_fingerprint(
        self,
        token: str,
    ) -> str:
        return hmac.new(
            self._secret,
            token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_token_shape(
        token: str,
    ) -> None:
        if not isinstance(
            token,
            str,
        ):
            raise TokenValidationError(
                "Token must be a string."
            )

        parts = token.split(".")

        if len(parts) != 3:
            raise TokenValidationError(
                "Malformed authentication token."
            )

        if any(
            not part
            for part in parts
        ):
            raise TokenValidationError(
                "Malformed authentication token."
            )

    def validate(
        self,
        token: str,
        now: Optional[int] = None,
        user_lookup: Optional[
            Callable[[str], User]
        ] = None,
    ) -> AuthenticatedPrincipal:
        """
        Validate a token and return an authenticated principal.

        ``user_lookup`` is optional for compatibility with the existing
        RAGShield API. When supplied, current user state is checked so a
        disabled account cannot continue using an otherwise-valid token.

        The lookup callback must return a User or raise an exception.
        """
        self._validate_token_shape(
            token
        )

        if self.is_revoked(token):
            raise TokenValidationError(
                "Authentication token has been revoked."
            )

        parts = token.split(".")

        header_b64 = parts[0]
        payload_b64 = parts[1]
        signature_b64 = parts[2]

        try:
            header_bytes = (
                self._b64decode(
                    header_b64
                )
            )

            payload_bytes = (
                self._b64decode(
                    payload_b64
                )
            )

            signature_bytes = (
                self._b64decode(
                    signature_b64
                )
            )

            header = json.loads(
                header_bytes.decode(
                    "utf-8"
                )
            )

            payload = json.loads(
                payload_bytes.decode(
                    "utf-8"
                )

            )

        except (
            ValueError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise TokenValidationError(
                "Malformed authentication token."
            ) from exc

        if not isinstance(
            header,
            dict,
        ):
            raise TokenValidationError(
                "Invalid token header."
            )

        if not isinstance(
            payload,
            dict,
        ):
            raise TokenValidationError(
                "Invalid token payload."
            )

        if header.get("alg") != TOKEN_ALGORITHM:
            raise TokenValidationError(
                "Unsupported token signing algorithm."
            )

        if header.get("typ") != "RAGShieldToken":
            raise TokenValidationError(
                "Invalid token type."
            )

        if int(
            header.get(
                "ver",
                -1,
            )
        ) != TOKEN_VERSION:
            raise TokenValidationError(
                "Unsupported token version."
            )

        if int(
            payload.get(
                "auth_version",
                -1,
            )
        ) != AUTH_VERSION:
            raise TokenValidationError(
                "Unsupported authentication version."
            )

        if int(
            payload.get(
                "token_version",
                -1,
            )
        ) != TOKEN_VERSION:
            raise TokenValidationError(
                "Unsupported token version."
            )

        signing_input = (
            f"{header_b64}.{payload_b64}"
            .encode("ascii")
        )

        if not self._signature_matches(
            signing_input,
            signature_bytes,
        ):
            raise TokenValidationError(
                "Invalid authentication token signature."
            )

        try:
            username = str(
                payload["username"]
            )

            issued_at = int(
                payload["iat"]
            )

            expires_at = int(
                payload["exp"]
            )

            token_id = str(
                payload["jti"]
            )

            roles = frozenset(
                str(role)
                for role in payload.get(
                    "roles",
                    [],
                )
            )

            document_scopes = frozenset(
                str(scope)
                for scope in payload.get(
                    "document_scopes",
                    [],
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise TokenValidationError(
                "Invalid authentication token payload."
            ) from exc

        if not username:
            raise TokenValidationError(
                "Token username is empty."
            )

        if not token_id:
            raise TokenValidationError(
                "Token identifier is missing."
            )

        if expires_at <= issued_at:
            raise TokenValidationError(
                "Token expiration is invalid."
            )

        current_time = (
            int(time.time())
            if now is None
            else int(now)
        )

        if issued_at > current_time:
            raise TokenValidationError(
                "Authentication token was issued in the future."
            )

        if current_time >= expires_at:
            raise TokenValidationError(
                "Authentication token has expired."
            )

        # Optional current-user validation.
        #
        # This is important because a token may have been issued while the
        # account was active and subsequently disabled.
        if user_lookup is not None:
            try:
                current_user = user_lookup(
                    username
                )
            except AuthenticationError:
                raise
            except Exception as exc:
                raise TokenValidationError(
                    "Unable to validate current user state."
                ) from exc

            if not isinstance(
                current_user,
                User,
            ):
                raise TokenValidationError(
                    "User lookup returned an invalid user."
                )

            if current_user.disabled:
                raise TokenValidationError(
                    "User account is disabled."
                )

            # Refresh authorization state from the current account instead
            # of trusting potentially stale roles/scopes in the token.
            roles = current_user.roles
            document_scopes = (
                current_user.document_scopes
            )

        return AuthenticatedPrincipal(
            username=username,
            roles=roles,
            document_scopes=document_scopes,
            issued_at=issued_at,
            expires_at=expires_at,
            token_version=int(
                payload.get(
                    "token_version",
                    TOKEN_VERSION,
                )
            ),
            token_id=token_id,
        )


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------


class RBAC:
    """
    Role-based access-control policy.

    Roles:
        user
        security_analyst
        admin

    Permissions:
        query
        read_audit
        read_security_events
        run_red_team
        manage_users
        manage_documents
        reset_corpus
    """

    ROLE_PERMISSIONS = {
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
                "read_security_events",
                "run_red_team",
                "manage_users",
                "manage_documents",
                "reset_corpus",
            }
        ),
    }

    def permissions_for(
        self,
        principal: AuthenticatedPrincipal,
    ) -> FrozenSet[str]:
        """Return the union of permissions granted by all principal roles."""
        if not isinstance(
            principal,
            AuthenticatedPrincipal,
        ):
            raise AuthorizationError(
                "Invalid authenticated principal."
            )

        permissions = set()

        for role in principal.roles:
            permissions.update(
                self.ROLE_PERMISSIONS.get(
                    role,
                    frozenset(),
                )
            )

        return frozenset(
            permissions
        )

    def has_permission(
        self,
        principal: AuthenticatedPrincipal,
        permission: str,
    ) -> bool:
        """Return whether a principal has a permission."""
        if not isinstance(
            permission,
            str,
        ):
            return False

        normalized_permission = (
            permission.strip().lower()
        )

        return (
            normalized_permission
            in self.permissions_for(
                principal
            )
        )

    def require_permission(
        self,
        principal: AuthenticatedPrincipal,
        permission: str,
    ) -> None:
        """Require a specific permission."""
        if not self.has_permission(
            principal,
            permission,
        ):
            raise AuthorizationError(
                f"Permission denied: {permission}"
            )

    def has_role(
        self,
        principal: AuthenticatedPrincipal,
        role: str,
    ) -> bool:
        """Return whether a principal has a role."""
        if not isinstance(
            role,
            str,
        ):
            return False

        return (
            role.strip().lower()
            in principal.roles
        )

    def require_role(
        self,
        principal: AuthenticatedPrincipal,
        role: str,
    ) -> None:
        """Require a specific role."""
        if not self.has_role(
            principal,
            role,
        ):
            raise AuthorizationError(
                f"Role denied: {role}"
            )

    def require_document_access(
        self,
        principal: AuthenticatedPrincipal,
        document_id: str,
    ) -> None:
        """
        Require access to a document.

        Administrators bypass document scopes.
        A wildcard scope grants access to all documents.
        """
        if not isinstance(
            principal,
            AuthenticatedPrincipal,
        ):
            raise AuthorizationError(
                "Invalid authenticated principal."
            )

        if not isinstance(
            document_id,
            str,
        ):
            raise AuthorizationError(
                "Invalid document identifier."
            )

        normalized_document_id = (
            document_id.strip()
        )

        if not normalized_document_id:
            raise AuthorizationError(
                "Document identifier cannot be empty."
            )

        if "admin" in principal.roles:
            return

        if "*" in principal.document_scopes:
            return

        if (
            normalized_document_id
            not in principal.document_scopes
        ):
            raise AuthorizationError(
                "Document access denied."
            )


# ---------------------------------------------------------------------------
# Safe audit metadata
# ---------------------------------------------------------------------------


def safe_user_metadata(
    user: User,
) -> Dict[str, object]:
    """
    Return user metadata safe for audit/API responses.

    Never exposes password hashes or salts.
    """
    if not isinstance(
        user,
        User,
    ):
        raise TypeError(
            "user must be a User instance."
        )

    return {
        "username": user.username,
        "roles": sorted(
            user.roles
        ),
        "document_scopes": sorted(
            user.document_scopes
        ),
        "disabled": bool(
            user.disabled
        ),
    }


def safe_principal_metadata(
    principal: AuthenticatedPrincipal,
) -> Dict[str, object]:
    """
    Return principal metadata safe for audit/API responses.

    Never exposes bearer tokens or password material.
    """
    if not isinstance(
        principal,
        AuthenticatedPrincipal,
    ):
        raise TypeError(
            "principal must be an AuthenticatedPrincipal."
        )

    return principal.safe_metadata()


__all__ = [
    "AUTH_VERSION",
    "TOKEN_VERSION",
    "DEFAULT_TOKEN_TTL_SECONDS",
    "PBKDF2_ITERATIONS",
    "SALT_BYTES",
    "MIN_PASSWORD_LENGTH",
    "MIN_TOKEN_SECRET_BYTES",
    "AUTH_SECRET_ENV",
    "AuthenticationError",
    "AuthorizationError",
    "TokenValidationError",
    "User",
    "AuthenticatedPrincipal",
    "PasswordHasher",
    "UserStore",
    "TokenManager",
    "RBAC",
    "safe_user_metadata",
    "safe_principal_metadata",
]
