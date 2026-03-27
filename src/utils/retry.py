"""
Async retry utility — used by every network call in the bot.

Usage:
    result = await async_retry(
        fn=lambda: client.call(),
        attempts=3,
        backoff_base=2.0,
        max_backoff=16.0,
        label="ollama_decision",
    )
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, label: str, attempts: int, last_exc: Exception) -> None:
        super().__init__(f"{label}: exhausted {attempts} attempts — last error: {last_exc}")
        self.label = label
        self.attempts = attempts
        self.last_exc = last_exc


async def async_retry(
    fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    attempts: int = 3,
    backoff_base: float = 2.0,
    max_backoff: float = 16.0,
    label: str = "async_retry",
    reraise_types: tuple[type[Exception], ...] = (),
) -> T:
    """
    Retry an async callable with exponential backoff.

    Args:
        fn:             Zero-argument async callable to retry.
        attempts:       Total number of attempts (1 = no retry).
        backoff_base:   Initial sleep duration in seconds (doubles each attempt).
        max_backoff:    Maximum sleep duration cap.
        label:          Log label for diagnostics.
        reraise_types:  Exception types to re-raise immediately without retrying.

    Returns:
        The return value of fn() on success.

    Raises:
        RetryExhausted: If all attempts fail.
        Any exception matching reraise_types: re-raised immediately.
    """
    last_exc: Exception | None = None
    delay = backoff_base

    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except tuple(reraise_types) as exc:  # type: ignore[misc]
            raise exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts:
                log.warning(
                    "retry_attempt_failed",
                    label=label,
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                    retry_in_seconds=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_backoff)
            else:
                log.error(
                    "retry_exhausted",
                    label=label,
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                )

    raise RetryExhausted(label=label, attempts=attempts, last_exc=last_exc)  # type: ignore[arg-type]
