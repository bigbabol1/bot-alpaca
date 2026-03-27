"""Tests for async_retry utility — all retry behavior covered in one file."""
from __future__ import annotations

import asyncio

import pytest

from src.utils.retry import RetryExhausted, async_retry


async def test_retry_succeeds_first_attempt():
    """Happy path: succeeds on first attempt, no retry."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    result = await async_retry(fn, attempts=3, label="test")
    assert result == "ok"
    assert calls == 1


async def test_retry_succeeds_on_second_attempt():
    """Fails once, then succeeds — returns value."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ValueError("transient error")
        return "success"

    result = await async_retry(fn, attempts=3, backoff_base=0.001, label="test")
    assert result == "success"
    assert calls == 2


async def test_retry_exhausted_raises():
    """All attempts fail → RetryExhausted raised."""
    async def fn():
        raise ConnectionError("always fails")

    with pytest.raises(RetryExhausted) as exc_info:
        await async_retry(fn, attempts=3, backoff_base=0.001, label="test_exhausted")

    assert exc_info.value.attempts == 3
    assert exc_info.value.label == "test_exhausted"
    assert isinstance(exc_info.value.last_exc, ConnectionError)


async def test_retry_reraise_types_immediate():
    """Exception in reraise_types bypasses retry and raises immediately."""
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt("stop immediately")

    with pytest.raises(KeyboardInterrupt):
        await async_retry(fn, attempts=5, reraise_types=(KeyboardInterrupt,), label="test")

    assert calls == 1   # no retries


async def test_retry_single_attempt_no_sleep():
    """attempts=1 → raises immediately, no backoff sleep."""
    import time
    t0 = time.monotonic()

    async def fn():
        raise RuntimeError("fail")

    with pytest.raises(RetryExhausted):
        await async_retry(fn, attempts=1, backoff_base=10.0, label="test")

    elapsed = time.monotonic() - t0
    assert elapsed < 1.0   # no sleep happened


async def test_retry_backoff_caps_at_max():
    """Backoff doubles but is capped at max_backoff."""
    delays_seen = []
    original_sleep = asyncio.sleep

    async def mock_sleep(secs):
        delays_seen.append(secs)
        await original_sleep(0)

    async def fn():
        raise RuntimeError("fail")

    import unittest.mock as mock
    with mock.patch("src.utils.retry.asyncio.sleep", side_effect=mock_sleep):
        with pytest.raises(RetryExhausted):
            await async_retry(
                fn,
                attempts=5,
                backoff_base=2.0,
                max_backoff=5.0,
                label="test",
            )

    # Delays: 2.0, 4.0, 5.0 (capped), 5.0 (capped) — 4 sleeps for 5 attempts
    assert delays_seen[0] == 2.0
    assert delays_seen[1] == 4.0
    assert all(d <= 5.0 for d in delays_seen)
