"""
Logging context management for FalconEYE.

Provides thread-local storage for correlation IDs and metrics that
automatically propagate to all log messages within a context.

Design:
- Thread-local storage ensures thread safety
- Context manager interface for automatic cleanup
- Automatic merging of context into log extra fields

Example:
    >>> with logging_context(command_id="cmd-123", project_id="proj-456"):
    ...     logger.info("Processing started")  # Automatically includes IDs
    ...     process_files()  # All logs in call stack include IDs
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict
from copy import deepcopy


class LogContext:
    """
    Thread-local storage for logging context.

    Provides a thread-safe way to store and retrieve context information
    (like correlation IDs, project IDs, metrics) that should be included
    in all log messages within the current execution context.

    Thread-local storage ensures that each thread has its own independent
    context, preventing context leakage between concurrent operations.

    Example:
        >>> LogContext.set("command_id", "cmd-123")
        >>> LogContext.set("project_id", "proj-456")
        >>> context = LogContext.get_context()
        >>> # {'command_id': 'cmd-123', 'project_id': 'proj-456'}
    """

    _local = threading.local()

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """
        Get current thread's logging context.

        Returns:
            Dictionary of context fields for current thread
        """
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return deepcopy(cls._local.context)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a single context field.

        Args:
            key: Context field name
            value: Context field value
        """
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        cls._local.context[key] = value

    @classmethod
    def update(cls, fields: Dict[str, Any]) -> None:
        """
        Update multiple context fields at once.

        Args:
            fields: Dictionary of fields to add/update in context
        """
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        cls._local.context.update(fields)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a specific context field.

        Args:
            key: Context field name
            default: Default value if field not found

        Returns:
            Field value or default
        """
        if not hasattr(cls._local, 'context'):
            return default
        return cls._local.context.get(key, default)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all context fields for current thread.
        """
        if hasattr(cls._local, 'context'):
            cls._local.context = {}

    @classmethod
    def remove(cls, *keys: str) -> None:
        """
        Remove specific context fields.

        Args:
            *keys: Field names to remove
        """
        if not hasattr(cls._local, 'context'):
            return

        for key in keys:
            cls._local.context.pop(key, None)


@contextmanager
def logging_context(**fields):
    """
    Context manager for automatic logging context management.

    Sets context fields on entry and removes them on exit (even if exception occurs).
    Preserves fields that were set before entering the context.

    Args:
        **fields: Context fields to set (e.g., command_id="cmd-123")

    Example:
        >>> with logging_context(command_id="cmd-123", project_id="proj-456"):
        ...     logger.info("Processing")  # Includes cmd-123 and proj-456
        ...     do_work()  # All logs include these IDs
        ... # Context automatically cleaned up

    Nested contexts:
        >>> with logging_context(command_id="outer"):
        ...     with logging_context(operation="inner"):
        ...         # Both command_id and operation are in context
        ...         pass
        ...     # Only command_id remains
        ... # All cleared
    """
    # Set the context fields
    LogContext.update(fields)

    try:
        yield
    finally:
        # Remove only the fields we added
        LogContext.remove(*fields.keys())
