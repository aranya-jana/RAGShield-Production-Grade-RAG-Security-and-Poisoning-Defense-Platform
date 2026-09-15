"""RAGShield security telemetry and request-correlation primitives.

Deterministic, local-only helpers for API security events.

Security telemetry deliberately records metadata rather than request bodies,
bearer tokens, passwords, attack payloads, or query contents.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Optional


TELEMETRY_VERSION = "1.0"
_REQUEST_ID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")


@dataclass(frozen=True)
class RequestSecurityContext:
    request_id: str
    method: str
    path: str


_context: ContextVar[Optional[RequestSecurityContext]] = ContextVar(
    "ragshield_request_security_context",
    default=None,
)


def new_request_id() -> str:
    """Return a fresh UUID4 request correlation identifier."""
    return str(uuid.uuid4())


def begin_request(
    *,
    request_id: Optional[str] = None,
    method: str = "",
    path: str = "",
) -> str:
    """Set the current request security context and return its request ID."""
    value = request_id or new_request_id()
    if not _REQUEST_ID_PATTERN.fullmatch(value):
        raise ValueError("request_id must be a UUID string.")

    _context.set(
        RequestSecurityContext(
            request_id=value,
            method=str(method),
            path=str(path),
        )
    )
    return value


def end_request() -> None:
    """Clear the current request context."""
    _context.set(None)


def get_request_context() -> Optional[RequestSecurityContext]:
    """Return the current request security context."""
    return _context.get()


def get_request_id() -> Optional[str]:
    """Return the current request ID, if one exists."""
    context = get_request_context()
    return context.request_id if context else None


def safe_content_metadata(value: Optional[str]) -> Dict[str, object]:
    """Return non-sensitive metadata for text without retaining the text."""
    if value is None:
        return {}

    text = str(value)
    return {
        "content_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "content_length": len(text),
    }


def security_event_metadata(
    *,
    category: str,
    reason_code: str,
    severity: str = "WARNING",
    username: Optional[str] = None,
    permission: Optional[str] = None,
    scope: Optional[str] = None,
    limit: Optional[int] = None,
    remaining: Optional[int] = None,
    retry_after_seconds: Optional[int] = None,
    content: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a safe, structured security-event metadata dictionary."""
    context = get_request_context()

    metadata: Dict[str, object] = {
        "telemetry_version": TELEMETRY_VERSION,
        "category": str(category),
        "reason_code": str(reason_code),
        "severity": str(severity),
    }

    if context:
        metadata.update(
            {
                "request_id": context.request_id,
                "method": context.method,
                "path": context.path,
            }
        )

    if username:
        metadata["username"] = str(username)

    if permission:
        metadata["permission"] = str(permission)

    if scope:
        metadata["scope"] = str(scope)

    if limit is not None:
        metadata["limit"] = int(limit)

    if remaining is not None:
        metadata["remaining"] = int(remaining)

    if retry_after_seconds is not None:
        metadata["retry_after_seconds"] = int(
            retry_after_seconds
        )

    if content is not None:
        metadata.update(
            {
                "content": safe_content_metadata(content),
            }
        )

    if extra:
        # Caller-supplied extras are intentionally restricted to scalar
        # metadata and must never contain raw request material.
        for key, value in extra.items():
            if key in {
                "token",
                "password",
                "password_hash",
                "password_salt",
                "secret",
                "authorization",
                "query",
                "payload",
                "body",
            }:
                continue
            metadata[str(key)] = value

    return metadata


__all__ = [
    "TELEMETRY_VERSION",
    "RequestSecurityContext",
    "new_request_id",
    "begin_request",
    "end_request",
    "get_request_context",
    "get_request_id",
    "safe_content_metadata",
    "security_event_metadata",
]
