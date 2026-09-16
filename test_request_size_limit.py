import asyncio

import pytest

from src.request_size_limit import (
    RequestSizeLimitMiddleware,
    RequestTooLargeError,
)


def test_rejects_content_length_above_global_limit():
    sent = []

    async def app(scope, receive, send):
        raise AssertionError("application should not be called")

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=100,
    )

    async def send(message):
        sent.append(message)

    async def receive():
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    scope = {
        "type": "http",
        "path": "/test",
        "headers": [(b"content-length", b"101")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413
    assert b"too large" in sent[1]["body"].lower()


def test_allows_content_length_at_limit():
    called = []

    async def app(scope, receive, send):
        called.append(True)

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=100,
    )

    async def send(message):
        pass

    async def receive():
        return {
            "type": "http.request",
            "body": b"x" * 100,
            "more_body": False,
        }

    scope = {
        "type": "http",
        "path": "/test",
        "headers": [(b"content-length", b"100")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert called == [True]


def test_path_specific_limit_is_stricter():
    sent = []
    called = []

    async def app(scope, receive, send):
        called.append(True)

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=1000,
        path_limits={"/query": 100},
    )

    async def send(message):
        sent.append(message)

    async def receive():
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    scope = {
        "type": "http",
        "path": "/query",
        "headers": [(b"content-length", b"101")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert called == []
    assert sent[0]["status"] == 413


def test_streamed_body_cannot_bypass_limit():
    sent = []
    calls = 0

    async def app(scope, receive, send):
        nonlocal calls

        calls += 1

        first = await receive()
        assert first["type"] == "http.request"

        second = await receive()
        assert second["type"] == "http.request"

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=100,
    )

    async def send(message):
        sent.append(message)

    chunks = iter(
        [
            {
                "type": "http.request",
                "body": b"x" * 75,
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"x" * 26,
                "more_body": False,
            },
        ]
    )

    async def receive():
        return next(chunks)

    scope = {
        "type": "http",
        "path": "/test",
        "headers": [],
    }

    asyncio.run(middleware(scope, receive, send))

    assert calls == 1
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert sent[1]["type"] == "http.response.body"
    assert b"too large" in sent[1]["body"].lower()


def test_non_http_scope_passes_through():
    called = []

    async def app(scope, receive, send):
        called.append(scope["type"])

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=100,
    )

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        pass

    asyncio.run(
        middleware(
            {"type": "lifespan"},
            receive,
            send,
        )
    )

    assert called == ["lifespan"]


def test_invalid_content_length_does_not_crash():
    called = []

    async def app(scope, receive, send):
        called.append(True)

    middleware = RequestSizeLimitMiddleware(
        app,
        max_body_bytes=100,
    )

    async def receive():
        return {
            "type": "http.request",
            "body": b"hello",
            "more_body": False,
        }

    async def send(message):
        pass

    scope = {
        "type": "http",
        "path": "/test",
        "headers": [(b"content-length", b"not-a-number")],
    }

    asyncio.run(middleware(scope, receive, send))

    assert called == [True]