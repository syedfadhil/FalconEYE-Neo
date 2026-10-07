"""
Index registry repository interface.

This repository manages the index registry which tracks:
- Project metadata (git info, last scan time, etc.)
- File metadata (checksums, modification times, embedding IDs, etc.)

The registry enables smart re-indexing by detecting file changes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..value_objects.project_metadata import FileMetadata, ProjectMetadata


class IndexRegistryRepository(ABC):
    """
    Abstract repository for managing the index registry.

    The index registry stores metadata about indexed projects and files
    to enable smart re-indexing and change detection.
    """

    # ===== Project Management =====

    @abstractmethod
    def save_project(self, project: ProjectMetadata) -> None:
        """
        Save or update project metadata.

        Args:
            project: Project metadata to save
        """
        pass

    @abstractmethod
    def get_project(self, project_id: str) -> Optional[ProjectMetadata]:
        """
        Get project metadata by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project metadata if found, None otherwise
        """
        pass

    @abstractmethod
    def get_all_projects(self) -> List[ProjectMetadata]:
        """
        Get metadata for all indexed projects.

        Returns:
            List of all project metadata
        """
        pass

    @abstractmethod
    def delete_project(self, project_id: str) -> bool:
        """
        Delete project and all associated file metadata.

        Args:
            project_id: Project identifier

        Returns:
            True if project was deleted, False if not found
        """
        pass

    @abstractmethod
    def project_exists(self, project_id: str) -> bool:
        """
        Check if a project exists in the registry.

        Args:
            project_id: Project identifier

        Returns:
            True if project exists, False otherwise
        """
        pass

    # ===== File Management =====

    @abstractmethod
    def save_file(self, file_meta: FileMetadata) -> None:
        """
        Save or update file metadata.

        Args:
            file_meta: File metadata to save
        """
        pass

    @abstractmethod
    def save_files_batch(self, file_metas: List[FileMetadata]) -> None:
        """
        Save or update multiple file metadata entries efficiently.

        Args:
            file_metas: List of file metadata to save
        """
        pass

    @abstractmethod
    def get_file(self, project_id: str, file_path: Path) -> Optional[FileMetadata]:
        """
        Get file metadata by project ID and file path.

        Args:
            project_id: Project identifier
            file_path: Absolute file path

        Returns:
            File metadata if found, None otherwise
        """
        pass

    @abstractmethod
    def get_all_files(self, project_id: str) -> List[FileMetadata]:
        """
        Get all file metadata for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of all file metadata for the project
        """
        pass

    @abstractmethod
    def get_files_by_status(
        self, project_id: str, status: str
    ) -> List[FileMetadata]:
        """
        Get files by status (active, deleted, modified).

        Args:
            project_id: Project identifier
            status: File status to filter by

        Returns:
            List of file metadata matching the status
        """
        pass

    @abstractmethod
    def delete_file(self, project_id: str, file_path: Path) -> bool:
        """
        Delete file metadata.

        Args:
            project_id: Project identifier
            file_path: Absolute file path

        Returns:
            True if file was deleted, False if not found
        """
        pass

    @abstractmethod
    def delete_files_batch(self, project_id: str, file_paths: List[Path]) -> int:
        """
        Delete multiple file metadata entries efficiently.

        Args:
            project_id: Project identifier
            file_paths: List of file paths to delete

        Returns:
            Number of files deleted
        """
        pass

    @abstractmethod
    def mark_file_deleted(self, project_id: str, file_path: Path) -> bool:
        """
        Mark a file as deleted without removing metadata.

        Args:
            project_id: Project identifier
            file_path: Absolute file path

        Returns:
            True if file was marked, False if not found
        """
        pass

    # ===== Query Operations =====

    @abstractmethod
    def get_file_paths(self, project_id: str) -> Set[Path]:
        """
        Get all file paths for a project.

        Args:
            project_id: Project identifier

        Returns:
            Set of all file paths in the project
        """
        pass

    @abstractmethod
    def get_files_metadata_dict(self, project_id: str) -> Dict[Path, FileMetadata]:
        """
        Get all file metadata as a dictionary for efficient lookup.

        Args:
            project_id: Project identifier

        Returns:
            Dict mapping file paths to metadata
        """
        pass

    @abstractmethod
    def get_project_stats(self, project_id: str) -> Dict[str, int]:
        """
        Get statistics for a project.

        Args:
            project_id: Project identifier

        Returns:
            Dict with keys: total_files, total_chunks, active_files, deleted_files
        """
        pass

    # ===== Bulk Operations =====

    @abstractmethod
    def clear_project_files(self, project_id: str) -> int:
        """
        Clear all file metadata for a project (keep project metadata).

        Args:
            project_id: Project identifier

        Returns:
            Number of files cleared
        """
        pass
