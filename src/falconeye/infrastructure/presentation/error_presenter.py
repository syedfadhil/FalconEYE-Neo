"""
ErrorPresenter - User-friendly error message generation.

Transforms technical exceptions into actionable, user-friendly messages.
Supports verbose mode for technical details.
"""

import traceback
from typing import Tuple, List

from ...domain.exceptions import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


class ErrorPresenter:
    """
    Presents errors to users with helpful messages and actions.

    Provides two modes:
    - Normal: User-friendly message with actionable suggestions
    - Verbose: Technical details including stack trace
    """

    @staticmethod
    def present(error: Exception, verbose: bool = False) -> str:
        """
        Present error with user-friendly message.

        Args:
            error: Exception to present
            verbose: Show technical details (stack trace, error type)

        Returns:
            Formatted error message
        """
        # Get friendly message and suggestions
        message, suggestions = ErrorPresenter._get_friendly_message(error)

        # Format output
        if verbose:
            return ErrorPresenter._format_verbose(error, message, suggestions)
        else:
            return ErrorPresenter._format_friendly(message, suggestions)

    @staticmethod
    def _get_friendly_message(error: Exception) -> Tuple[str, List[str]]:
        """
        Get friendly message and actionable suggestions for error.

        Args:
            error: Exception to analyze

        Returns:
            Tuple of (message, suggestions)
        """
        error_str = str(error)

        # OllamaConnectionError
        if isinstance(error, OllamaConnectionError):
            return (
                "Could not connect to Ollama service",
                [
                    "Check if Ollama is running: `docker ps` or `ps aux | grep ollama`",
                    "Start Ollama: `docker start ollama` or `ollama serve`",
                    "Check connection: `curl http://localhost:11434/api/tags`",
                    "Verify the base_url in config.yaml matches your Ollama server"
                ]
            )

        # OllamaModelNotFoundError
        if isinstance(error, OllamaModelNotFoundError):
            # Try to extract model name from error message
            model_name = "the required model"
            if "'" in error_str:
                parts = error_str.split("'")
                if len(parts) >= 2:
                    model_name = parts[1]

            return (
                f"Model '{model_name}' not found in Ollama",
                [
                    f"Pull the model: `ollama pull {model_name}`",
                    "List available models: `ollama list`",
                    "Check model configuration in config.yaml"
                ]
            )

        # OllamaTimeoutError
        if isinstance(error, OllamaTimeoutError):
            return (
                "Ollama request timed out",
                [
                    "Increase timeout in config.yaml: `llm.timeout: 300`",
                    "Check Ollama performance: `docker stats ollama`",
                    "Verify Ollama has sufficient resources (CPU, memory)",
                    "Consider using a smaller/faster model"
                ]
            )

        # FileNotFoundError
        if isinstance(error, FileNotFoundError):
            file_path = str(error).replace("File not found: ", "").replace("[Errno 2] No such file or directory: ", "").strip("'\"")
            return (
                f"File not found: {file_path}",
                [
                    "Check the file path is correct",
                    "Ensure the file exists: `ls -la <path>`",
                    "Check for typos in the path"
                ]
            )

        # PermissionError
        if isinstance(error, PermissionError):
            path = str(error).replace("Permission denied: ", "").replace("[Errno 13] Permission denied: ", "").strip("'\"")
            return (
                f"Permission denied: {path}",
                [
                    f"Check file permissions: `ls -la {path}`",
                    "Ensure you have read access to the file/directory",
                    "Try running with appropriate permissions (avoid sudo unless necessary)"
                ]
            )

        # UnicodeDecodeError
        if isinstance(error, UnicodeDecodeError):
            return (
                "Could not decode file (invalid encoding)",
                [
                    "File may not be valid UTF-8 text",
                    "File might be binary or use a different encoding",
                    "Skip this file or convert it to UTF-8"
                ]
            )

        # KeyboardInterrupt
        if isinstance(error, KeyboardInterrupt):
            return (
                "Operation cancelled by user",
                []
            )

        # Database lock error (SQLite)
        if "database is locked" in error_str.lower():
            return (
                "Database is locked (another process may be using it)",
                [
                    "Close any other FalconEYE processes",
                    "Wait a few seconds and try again",
                    "Check for stale lock files in the data directory"
                ]
            )

        # Generic exception
        error_type = type(error).__name__
        error_msg = error_str if error_str else "No details available"

        return (
            f"An error occurred: {error_type}",
            [
                f"Error details: {error_msg}",
                "Run with --verbose for more information",
                "Check the log file for details"
            ]
        )

    @staticmethod
    def _format_friendly(message: str, suggestions: List[str]) -> str:
        """
        Format user-friendly error message.

        Args:
            message: Main error message
            suggestions: List of actionable suggestions

        Returns:
            Formatted string
        """
        output = [f"Error: {message}"]

        if suggestions:
            output.append("")
            output.append("Suggestions:")
            for suggestion in suggestions:
                output.append(f"  • {suggestion}")

        return "\n".join(output)

    @staticmethod
    def _format_verbose(error: Exception, message: str, suggestions: List[str]) -> str:
        """
        Format verbose error message with technical details.

        Args:
            error: Original exception
            message: User-friendly message
            suggestions: Actionable suggestions

        Returns:
            Formatted string with full details
        """
        output = [f"Error: {message}"]

        if suggestions:
            output.append("")
            output.append("Suggestions:")
            for suggestion in suggestions:
                output.append(f"  • {suggestion}")

        output.append("")
        output.append("Technical Details:")
        output.append(f"  Error Type: {type(error).__name__}")
        output.append(f"  Error Message: {str(error)}")

        # Add cause chain if present
        if error.__cause__:
            output.append(f"  Caused by: {type(error.__cause__).__name__}: {str(error.__cause__)}")

        # Add stack trace
        output.append("")
        output.append("Traceback:")
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        for line in tb_lines:
            # Indent traceback lines
            for sub_line in line.rstrip().split('\n'):
                output.append(f"  {sub_line}")

        return "\n".join(output)
