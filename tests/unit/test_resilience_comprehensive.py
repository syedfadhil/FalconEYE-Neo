"""Unit tests for retry and circuit breaker resilience helpers."""

import asyncio

import pytest

from falconeye.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
)
from falconeye.infrastructure.resilience.retry import RetryConfig, retry_with_backoff


@pytest.mark.asyncio
async def test_retry_async_retries_correct_number_with_exponential_backoff(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    attempts = 0

    @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.5, exponential_base=2, max_delay=10, jitter=0, retryable_exceptions=(ConnectionError,)))
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert await flaky() == "ok"
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retry_async_exhausts_and_reraises_retryable_exception(monkeypatch):
    sleeps = []

    async def instant_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)
    attempts = 0

    @retry_with_backoff(RetryConfig(max_retries=1, initial_delay=0.25, jitter=0, retryable_exceptions=(TimeoutError,)))
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        await always_fails()
    assert attempts == 2
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_non_retryable_exception(monkeypatch):
    async def fake_sleep(delay):
        raise AssertionError("sleep should not be called")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    attempts = 0

    @retry_with_backoff(RetryConfig(max_retries=3, retryable_exceptions=(ConnectionError,)))
    async def bad_request():
        nonlocal attempts
        attempts += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        await bad_request()
    assert attempts == 1


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_failure_threshold_and_blocks_calls():
    breaker = CircuitBreaker("svc", CircuitBreakerConfig(failure_threshold=2, timeout=60, exclude_exceptions=()))
    attempts = 0

    @breaker.protect
    async def failing():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await failing()
    assert breaker.state is CircuitBreakerState.CLOSED
    with pytest.raises(RuntimeError):
        await failing()
    assert breaker.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitBreakerError):
        await failing()
    assert attempts == 2


@pytest.mark.asyncio
async def test_circuit_breaker_resets_after_cooldown_and_success_threshold(monkeypatch):
    breaker = CircuitBreaker("svc", CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout=5, exclude_exceptions=()))
    now = 1000.0
    monkeypatch.setattr("falconeye.infrastructure.resilience.circuit_breaker.time.time", lambda: now)

    @breaker.protect
    async def failing():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await failing()
    assert breaker.state is CircuitBreakerState.OPEN

    now = 1006.0
    assert breaker.state is CircuitBreakerState.HALF_OPEN

    @breaker.protect
    async def healthy():
        return "ok"

    assert await healthy() == "ok"
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    assert await healthy() == "ok"
    assert breaker.state is CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_reopens_immediately(monkeypatch):
    breaker = CircuitBreaker("svc", CircuitBreakerConfig(failure_threshold=1, timeout=1, exclude_exceptions=()))
    now = 1.0
    monkeypatch.setattr("falconeye.infrastructure.resilience.circuit_breaker.time.time", lambda: now)

    @breaker.protect
    async def fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await fail()
    now = 3.0
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    with pytest.raises(RuntimeError):
        await fail()
    assert breaker.state is CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_excluded_exceptions_do_not_trip_circuit():
    breaker = CircuitBreaker("svc", CircuitBreakerConfig(failure_threshold=1, exclude_exceptions=(ValueError,)))

    @breaker.protect
    async def invalid_input():
        raise ValueError("caller bug")

    with pytest.raises(ValueError):
        await invalid_input()
    assert breaker.state is CircuitBreakerState.CLOSED
