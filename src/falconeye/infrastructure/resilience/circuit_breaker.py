"""
Circuit Breaker pattern implementation.

Prevents cascade failures by stopping calls to a failing service
after a threshold is reached, then allowing periodic test calls.
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any
from functools import wraps
import threading

from ..logging import FalconEyeLogger, logging_context


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in HALF_OPEN to close circuit
        timeout: Seconds to wait before trying HALF_OPEN (recovery test)
        exclude_exceptions: Exceptions that don't count as failures
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    exclude_exceptions: tuple = (ValueError, TypeError)


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascade failures.

    Tracks failures and automatically opens the circuit (stops requests)
    when failure threshold is reached. After a timeout, enters HALF_OPEN
    state to test if the service has recovered.

    Thread-safe for use across concurrent requests.

    Example:
        >>> breaker = CircuitBreaker(name="llm_service")
        >>>
        >>> @breaker.protect
        ... async def call_llm(prompt):
        ...     return await llm.analyze(prompt)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the protected service (for logging)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config if config else CircuitBreakerConfig()

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

        self.logger = FalconEyeLogger.get_instance()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if (self._state == CircuitBreakerState.OPEN and
                self._last_failure_time and
                time.time() - self._last_failure_time >= self.config.timeout):
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0

                with logging_context(operation="circuit_breaker_half_open"):
                    self.logger.info(
                        f"Circuit breaker entering HALF_OPEN state: {self.name}",
                        extra={
                            "circuit_breaker": self.name,
                            "state": self._state.value,
                            "timeout_seconds": self.config.timeout
                        }
                    )

            return self._state

    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    # Close the circuit
                    self._state = CircuitBreakerState.CLOSED
                    self._success_count = 0

                    with logging_context(operation="circuit_breaker_closed"):
                        self.logger.info(
                            f"Circuit breaker closed (recovered): {self.name}",
                            extra={
                                "circuit_breaker": self.name,
                                "state": self._state.value,
                                "success_threshold": self.config.success_threshold
                            }
                        )

    def _record_failure(self, exception: Exception):
        """Record a failed call."""
        # Don't count excluded exceptions as failures
        if isinstance(exception, self.config.exclude_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            # If in HALF_OPEN, immediately open on failure
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                self._failure_count = 0

                with logging_context(operation="circuit_breaker_reopened"):
                    self.logger.warning(
                        f"Circuit breaker reopened (recovery failed): {self.name}",
                        extra={
                            "circuit_breaker": self.name,
                            "state": self._state.value,
                            "error_type": type(exception).__name__
                        }
                    )

            # If in CLOSED, check if we should open
            elif (self._state == CircuitBreakerState.CLOSED and
                  self._failure_count >= self.config.failure_threshold):
                self._state = CircuitBreakerState.OPEN

                with logging_context(operation="circuit_breaker_opened"):
                    self.logger.error(
                        f"Circuit breaker opened (failure threshold exceeded): {self.name}",
                        extra={
                            "circuit_breaker": self.name,
                            "state": self._state.value,
                            "failure_count": self._failure_count,
                            "failure_threshold": self.config.failure_threshold,
                            "timeout_seconds": self.config.timeout
                        }
                    )

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect an async function with circuit breaker.

        Args:
            func: Async function to protect

        Returns:
            Protected function

        Raises:
            CircuitBreakerError: If circuit is open
        """
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Check circuit state
            current_state = self.state

            if current_state == CircuitBreakerState.OPEN:
                with logging_context(operation="circuit_breaker_blocked"):
                    self.logger.warning(
                        f"Circuit breaker blocked call: {self.name}",
                        extra={
                            "circuit_breaker": self.name,
                            "state": current_state.value,
                            "function": func.__name__
                        }
                    )

                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN for {self.name}. "
                    f"Service unavailable. Will retry after {self.config.timeout}s."
                )

            # Execute the function
            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result

            except Exception as e:
                self._record_failure(e)
                raise

        return wrapper

    def protect_sync(self, func: Callable) -> Callable:
        """
        Decorator to protect a synchronous function with circuit breaker.

        Args:
            func: Synchronous function to protect

        Returns:
            Protected function

        Raises:
            CircuitBreakerError: If circuit is open
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_state = self.state

            if current_state == CircuitBreakerState.OPEN:
                with logging_context(operation="circuit_breaker_blocked"):
                    self.logger.warning(
                        f"Circuit breaker blocked call: {self.name}",
                        extra={
                            "circuit_breaker": self.name,
                            "state": current_state.value,
                            "function": func.__name__
                        }
                    )

                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN for {self.name}. "
                    f"Service unavailable. Will retry after {self.config.timeout}s."
                )

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result

            except Exception as e:
                self._record_failure(e)
                raise

        return wrapper

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

            with logging_context(operation="circuit_breaker_reset"):
                self.logger.info(
                    f"Circuit breaker manually reset: {self.name}",
                    extra={
                        "circuit_breaker": self.name,
                        "state": self._state.value
                    }
                )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocking requests."""
    pass
