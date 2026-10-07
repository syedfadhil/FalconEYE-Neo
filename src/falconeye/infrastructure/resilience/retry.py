"""
Retry logic with exponential backoff.

Provides decorator-based retry mechanism for handling transient failures
in external service calls (LLM, vector store, etc.).
"""

import asyncio
import time
from typing import Optional, Callable, Any, Type, Tuple
from dataclasses import dataclass
from functools import wraps

from ..logging import FalconEyeLogger, logging_context


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff (typically 2)
        jitter: Add random jitter to prevent thundering herd (0.0 to 1.0)
        retryable_exceptions: Tuple of exception types that should trigger retry
    """
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(RetryConfig(max_retries=5))
        ... async def call_llm(prompt):
        ...     return await llm_service.analyze(prompt)
    """
    if config is None:
        config = RetryConfig()

    logger = FalconEyeLogger.get_instance()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    # First attempt or retry
                    if attempt > 0:
                        # Calculate delay with exponential backoff
                        delay = min(
                            config.initial_delay * (config.exponential_base ** (attempt - 1)),
                            config.max_delay
                        )

                        # Add jitter to prevent thundering herd
                        if config.jitter > 0:
                            import random
                            jitter_amount = delay * config.jitter * random.random()
                            delay = delay + jitter_amount

                        with logging_context(operation="retry_backoff"):
                            logger.warning(
                                f"Retrying {func.__name__} (attempt {attempt + 1}/{config.max_retries + 1})",
                                extra={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_attempts": config.max_retries + 1,
                                    "delay_seconds": round(delay, 2),
                                    "last_error": type(last_exception).__name__ if last_exception else None
                                }
                            )

                        await asyncio.sleep(delay)

                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Log successful retry if this wasn't the first attempt
                    if attempt > 0:
                        with logging_context(operation="retry_success"):
                            logger.info(
                                f"Retry successful for {func.__name__}",
                                extra={
                                    "function": func.__name__,
                                    "successful_attempt": attempt + 1,
                                    "total_attempts": attempt + 1
                                }
                            )

                    return result

                except config.retryable_exceptions as e:
                    last_exception = e

                    # If this was the last attempt, log and re-raise
                    if attempt == config.max_retries:
                        with logging_context(operation="retry_exhausted"):
                            logger.error(
                                f"All retry attempts exhausted for {func.__name__}",
                                exc_info=True,
                                extra={
                                    "function": func.__name__,
                                    "total_attempts": attempt + 1,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)
                                }
                            )
                        raise

                    # Otherwise, continue to next retry
                    continue

                except Exception as e:
                    # Non-retryable exception, log and re-raise immediately
                    with logging_context(operation="retry_non_retryable"):
                        logger.error(
                            f"Non-retryable exception in {func.__name__}",
                            exc_info=True,
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        )
                    raise

            # Should never reach here, but just in case
            raise last_exception if last_exception else RuntimeError("Unexpected retry state")

        return wrapper
    return decorator


def retry_with_backoff_sync(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying synchronous functions with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff_sync(RetryConfig(max_retries=3))
        ... def fetch_data(url):
        ...     return requests.get(url)
    """
    if config is None:
        config = RetryConfig()

    logger = FalconEyeLogger.get_instance()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = min(
                            config.initial_delay * (config.exponential_base ** (attempt - 1)),
                            config.max_delay
                        )

                        if config.jitter > 0:
                            import random
                            jitter_amount = delay * config.jitter * random.random()
                            delay = delay + jitter_amount

                        with logging_context(operation="retry_backoff"):
                            logger.warning(
                                f"Retrying {func.__name__} (attempt {attempt + 1}/{config.max_retries + 1})",
                                extra={
                                    "function": func.__name__,
                                    "attempt": attempt + 1,
                                    "max_attempts": config.max_retries + 1,
                                    "delay_seconds": round(delay, 2)
                                }
                            )

                        time.sleep(delay)

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        with logging_context(operation="retry_success"):
                            logger.info(
                                f"Retry successful for {func.__name__}",
                                extra={
                                    "function": func.__name__,
                                    "successful_attempt": attempt + 1,
                                    "total_attempts": attempt + 1
                                }
                            )

                    return result

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        with logging_context(operation="retry_exhausted"):
                            logger.error(
                                f"All retry attempts exhausted for {func.__name__}",
                                exc_info=True,
                                extra={
                                    "function": func.__name__,
                                    "total_attempts": attempt + 1,
                                    "error_type": type(e).__name__
                                }
                            )
                        raise

                    continue

                except Exception as e:
                    with logging_context(operation="retry_non_retryable"):
                        logger.error(
                            f"Non-retryable exception in {func.__name__}",
                            exc_info=True,
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error_type": type(e).__name__
                            }
                        )
                    raise

            raise last_exception if last_exception else RuntimeError("Unexpected retry state")

        return wrapper
    return decorator
