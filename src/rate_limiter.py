"""
RAGShield API rate limiting.

Thread-safe, dependency-free, in-process rate limiting.

This implementation is appropriate for a single-process deployment or
local security POC.

For multi-worker or horizontally scaled production deployments, the
limiter should eventually be backed by a shared store such as Redis.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RateLimitDecision:
    """Result returned by a rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int = 0


@dataclass
class _Bucket:
    """Internal fixed-window request bucket."""

    window_started_at: float
    request_count: int
    last_seen_at: float


class RateLimiter:
    """
    Thread-safe fixed-window rate limiter.

    State is maintained independently for each:

        scope + client

    For example:

        query:user:alice
        attack:user:alice

    No authentication tokens, passwords, query contents, or attack
    payloads are stored.
    """

    def __init__(
        self,
        *,
        max_clients: int = 10_000,
        cleanup_interval_seconds: int = 60,
    ) -> None:

        if max_clients <= 0:
            raise ValueError(
                "max_clients must be greater than zero."
            )

        if cleanup_interval_seconds <= 0:
            raise ValueError(
                "cleanup_interval_seconds must be greater than zero."
            )

        self.max_clients = int(max_clients)

        self.cleanup_interval_seconds = int(
            cleanup_interval_seconds
        )

        self._buckets: Dict[str, _Bucket] = {}

        self._lock = threading.Lock()

        self._last_cleanup_at = time.monotonic()

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def check(
        self,
        client_key: str,
        scope: str,
        *,
        limit: int,
        window_seconds: int,
        now: Optional[float] = None,
    ) -> RateLimitDecision:
        """
        Consume one request from the client's rate-limit bucket.

        Args:
            client_key:
                Opaque identifier for the client.

            scope:
                Endpoint/resource scope.

            limit:
                Maximum requests permitted in the window.

            window_seconds:
                Length of the rate-limit window.

            now:
                Optional monotonic timestamp used by tests.

        Returns:
            RateLimitDecision.
        """

        if not client_key:
            raise ValueError(
                "client_key must not be empty."
            )

        if not scope:
            raise ValueError(
                "scope must not be empty."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be greater than zero."
            )

        current = (
            time.monotonic()
            if now is None
            else float(now)
        )

        bucket_key = (
            f"{scope}:{client_key}"
        )

        with self._lock:

            self._maybe_cleanup(
                current=current,
                window_seconds=window_seconds,
            )

            bucket = self._buckets.get(
                bucket_key
            )

            if bucket is None:

                self._enforce_capacity()

                bucket = _Bucket(
                    window_started_at=current,
                    request_count=0,
                    last_seen_at=current,
                )

                self._buckets[
                    bucket_key
                ] = bucket

            elapsed = (
                current
                - bucket.window_started_at
            )

            if elapsed >= window_seconds:

                bucket.window_started_at = current

                bucket.request_count = 0

            bucket.last_seen_at = current

            if bucket.request_count >= limit:

                retry_after = max(
                    1,
                    int(
                        window_seconds
                        - (
                            current
                            - bucket.window_started_at
                        )
                    )
                    + 1,
                )

                return RateLimitDecision(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            bucket.request_count += 1

            remaining = max(
                0,
                limit - bucket.request_count,
            )

            return RateLimitDecision(
                allowed=True,
                limit=limit,
                remaining=remaining,
                retry_after_seconds=0,
            )

    def reset(
        self,
        client_key: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> None:
        """
        Reset limiter state.

        With no arguments, all state is cleared.

        With client_key, only that client's buckets are removed.

        With scope, only that scope's buckets are removed.
        """

        with self._lock:

            if (
                client_key is None
                and scope is None
            ):
                self._buckets.clear()
                return

            keys_to_remove = []

            for key in self._buckets:

                key_scope, key_client = (
                    key.split(":", 1)
                )

                if (
                    client_key is not None
                    and key_client != client_key
                ):
                    continue

                if (
                    scope is not None
                    and key_scope != scope
                ):
                    continue

                keys_to_remove.append(key)

            for key in keys_to_remove:

                self._buckets.pop(
                    key,
                    None,
                )

    def size(self) -> int:
        """Return the number of active buckets."""

        with self._lock:
            return len(self._buckets)

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _maybe_cleanup(
        self,
        *,
        current: float,
        window_seconds: int,
    ) -> None:
        """
        Opportunistically remove stale buckets.

        No background cleanup thread is required.
        """

        if (
            current - self._last_cleanup_at
            < self.cleanup_interval_seconds
        ):
            return

        stale_after = max(
            self.cleanup_interval_seconds,
            window_seconds,
        )

        cutoff = (
            current - stale_after
        )

        stale_keys = [
            key
            for key, bucket in self._buckets.items()
            if bucket.last_seen_at < cutoff
        ]

        for key in stale_keys:

            self._buckets.pop(
                key,
                None,
            )

        self._last_cleanup_at = current

    def _enforce_capacity(self) -> None:
        """
        Keep limiter memory bounded.

        The least recently used bucket is evicted when capacity is
        reached.
        """

        if (
            len(self._buckets)
            < self.max_clients
        ):
            return

        oldest_key = min(
            self._buckets,
            key=lambda key:
                self._buckets[
                    key
                ].last_seen_at,
        )

        self._buckets.pop(
            oldest_key,
            None,
        )


__all__ = [
    "RateLimitDecision",
    "RateLimiter",
]