"""
RAGShield HTTP Security Middleware.

Provides defensive HTTP response headers for the FastAPI application.

The middleware is intentionally framework-light and deterministic:
- no network access
- no external security service
- no sensitive request/response logging
- API-friendly Content Security Policy
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add defensive HTTP security headers to every response.

    These headers reduce exposure to:
    - MIME-type sniffing
    - clickjacking
    - referrer leakage
    - unnecessary browser capabilities
    - unsafe resource execution
    - caching of sensitive API responses
    """

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"

        response.headers["Referrer-Policy"] = "no-referrer"

        response.headers[
            "Permissions-Policy"
        ] = "camera=(), microphone=(), geolocation=()"

        response.headers[
            "Content-Security-Policy"
        ] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'none'; "
            "form-action 'none'"
        )

        response.headers[
            "Cross-Origin-Resource-Policy"
        ] = "same-site"

        response.headers[
            "X-Permitted-Cross-Domain-Policies"
        ] = "none"

        # RAGShield API responses may contain security findings,
        # retrieved document information, or model-generated content.
        # Prevent browsers/intermediaries from caching them.
        if request.url.path in {
            "/query",
            "/attack",
            "/setup",
            "/audit",
        }:
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"

        return response