"""RAGShield HTTP Security Middleware.

Provides defensive HTTP response headers and request correlation IDs for the
FastAPI application.

The middleware is intentionally framework-light and deterministic:
- no network access
- no external security service
- no sensitive request/response logging
- API-friendly Content Security Policy
- per-request correlation ID propagation
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.security_telemetry import begin_request, end_request


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add defensive HTTP security headers and request IDs."""

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        request_id = str(uuid.uuid4())
        begin_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)

            response.headers["X-Request-ID"] = request_id
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

        finally:
            end_request()
