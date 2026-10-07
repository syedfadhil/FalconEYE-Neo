"""Base formatter interface."""

from abc import ABC, abstractmethod
from ...domain.models.security import SecurityReview, SecurityFinding


class OutputFormatter(ABC):
    """
    Base class for output formatters.

    Formatters convert security review results into various output formats
    (console, JSON, SARIF, etc.) for different use cases.
    """

    @abstractmethod
    def format_review(self, review: SecurityReview) -> str:
        """
        Format a complete security review.

        Args:
            review: SecurityReview to format

        Returns:
            Formatted string
        """
        pass

    @abstractmethod
    def format_finding(self, finding: SecurityFinding) -> str:
        """
        Format a single security finding.

        Args:
            finding: SecurityFinding to format

        Returns:
            Formatted string
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get file extension for this format.

        Returns:
            File extension (e.g., ".json", ".sarif")
        """
        pass