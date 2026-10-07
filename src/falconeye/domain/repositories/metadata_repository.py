"""Metadata repository interface (Port)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.structural import StructuralMetadata


class MetadataRepository(ABC):
    """
    Port for structural metadata operations.

    Stores AST-extracted metadata for enhanced AI context.
    """

    @abstractmethod
    async def store_metadata(
        self,
        metadata: StructuralMetadata,
    ) -> None:
        """
        Store structural metadata for a file.

        Args:
            metadata: Structural metadata to store
        """
        pass

    @abstractmethod
    async def get_metadata(
        self,
        file_path: str,
    ) -> Optional[StructuralMetadata]:
        """
        Retrieve metadata for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Structural metadata or None if not found
        """
        pass

    @abstractmethod
    async def get_function_calls_graph(
        self,
        target_function: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get function call graph.

        Args:
            target_function: Optional filter for specific function

        Returns:
            Dictionary mapping files to their function calls
        """
        pass

    @abstractmethod
    async def get_dependency_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Get dependency graph showing imports and dependencies.

        Returns:
            Dictionary mapping files to their dependencies
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics about the codebase.

        Returns:
            Dictionary with statistics (function count, imports, etc.)
        """
        pass

    @abstractmethod
    async def search_functions(
        self,
        function_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by name.

        Args:
            function_name: Function name to search for

        Returns:
            List of matching functions with file locations
        """
        pass