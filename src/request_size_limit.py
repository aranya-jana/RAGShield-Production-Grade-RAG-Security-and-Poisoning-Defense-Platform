from __future__ import annotations

from typing import Callable, Dict, Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestTooLargeError(Exception):
    """Raised when an HTTP request exceeds its configured size limit."""


class RequestSizeLimitMiddleware:
    """
    ASGI middleware that rejects oversized HTTP request bodies.

    The middleware performs an early Content-Length check when available and
    also counts streamed request-body bytes so chunked requests cannot bypass
    the configured limit.

    Raw request bodies are never logged or persisted.
    """

    DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MiB

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        path_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be greater than zero")

        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path_limits = dict(path_limits or {})

        for path, limit in self.path_limits.items():
            if not isinstance(path, str) or not path:
                raise ValueError("path limits must use non-empty string paths")
            if limit <= 0:
                raise ValueError(
                    f"body limit for {path!r} must be greater than zero"
                )

    def _limit_for_path(self, path: str) -> int:
        """
        Return the most specific configured limit for a request path.

        Exact path matches take precedence over the global limit.
        """
        return self.path_limits.get(path, self.max_body_bytes)

    @staticmethod
    def _content_length(headers: list[tuple[bytes, bytes]]) -> Optional[int]:
        for name, value in headers:
            if name.lower() == b"content-length":
                try:
                    return int(value.decode("ascii").strip())
                except (ValueError, UnicodeDecodeError):
                    return None
        return None

    @staticmethod
    async def _send_413(send: Send, limit: int) -> None:
        body = b'{"detail":"Request body too large"}'

        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": headers,
            }
        )

        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for_path(scope.get("path", ""))

        content_length = self._content_length(scope.get("headers", []))

        if content_length is not None and content_length > limit:
            await self._send_413(send, limit)
            return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes

            message = await receive()

            if message["type"] != "http.request":
                return message

            body = message.get("body", b"")

            if body:
                received_bytes += len(body)

                if received_bytes > limit:
                    raise RequestTooLargeError(
                        f"Request body exceeds configured limit of {limit} bytes"
                    )

            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLargeError:
            await self._send_413(send, limit)