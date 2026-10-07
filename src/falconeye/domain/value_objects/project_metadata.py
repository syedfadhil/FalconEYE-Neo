"""
Project and file metadata value objects.

This module contains value objects for tracking project and file metadata
used in the index registry for smart re-indexing and project isolation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ProjectType(str, Enum):
    """Type of project for indexing strategy."""

    GIT = "git"
    NON_GIT = "non-git"


@dataclass(frozen=True)
class ProjectMetadata:
    """
    Metadata about an indexed project.

    This tracks project-level information for smart re-indexing
    and project isolation.
    """

    project_id: str
    """Unique identifier for the project."""

    project_name: str
    """Human-readable project name."""

    project_root: Path
    """Absolute path to project root directory."""

    project_type: ProjectType
    """Type of project (git or non-git)."""

    git_remote_url: Optional[str] = None
    """Git remote URL if this is a git repository."""

    last_indexed_commit: Optional[str] = None
    """Last git commit hash when project was indexed (git repos only)."""

    last_full_scan: datetime = field(default_factory=datetime.now)
    """Timestamp of last full scan."""

    total_files: int = 0
    """Total number of files indexed."""

    total_chunks: int = 0
    """Total number of code chunks created."""

    languages: List[str] = field(default_factory=list)
    """List of programming languages detected in project."""

    created_at: datetime = field(default_factory=datetime.now)
    """When this project was first indexed."""

    updated_at: datetime = field(default_factory=datetime.now)
    """When this project metadata was last updated."""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_root": str(self.project_root),
            "project_type": self.project_type.value,
            "git_remote_url": self.git_remote_url,
            "last_indexed_commit": self.last_indexed_commit,
            "last_full_scan": self.last_full_scan.isoformat(),
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "languages": self.languages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMetadata":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            project_root=Path(data["project_root"]),
            project_type=ProjectType(data["project_type"]),
            git_remote_url=data.get("git_remote_url"),
            last_indexed_commit=data.get("last_indexed_commit"),
            last_full_scan=datetime.fromisoformat(data["last_full_scan"]),
            total_files=data.get("total_files", 0),
            total_chunks=data.get("total_chunks", 0),
            languages=data.get("languages", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class FileStatus(str, Enum):
    """Status of an indexed file."""

    ACTIVE = "active"
    """File is present and indexed."""

    DELETED = "deleted"
    """File was deleted from project."""

    MODIFIED = "modified"
    """File has been modified since last index."""


@dataclass(frozen=True)
class FileMetadata:
    """
    Metadata about an indexed file.

    This tracks file-level information for change detection
    and smart re-indexing.
    """

    project_id: str
    """ID of the project this file belongs to."""

    file_path: Path
    """Absolute path to the file."""

    relative_path: Path
    """Path relative to project root."""

    language: str
    """Programming language of the file."""

    file_checksum: str
    """SHA256 checksum of file content."""

    file_size: int
    """File size in bytes."""

    file_mtime: float
    """File modification time (Unix timestamp)."""

    git_commit_hash: Optional[str] = None
    """Git commit hash when file was indexed (if git repo)."""

    git_file_hash: Optional[str] = None
    """Git's internal hash for this file (if git repo)."""

    indexed_at: datetime = field(default_factory=datetime.now)
    """When this file was indexed."""

    chunk_count: int = 0
    """Number of chunks created from this file."""

    embedding_ids: List[str] = field(default_factory=list)
    """IDs of embeddings in vector store."""

    status: FileStatus = FileStatus.ACTIVE
    """Current status of the file."""

    last_scanned: datetime = field(default_factory=datetime.now)
    """When we last checked this file for changes."""

    @property
    def project_root(self) -> Path:
        """Get project root by subtracting relative path from file path."""
        # file_path = project_root / relative_path
        # So: project_root = file_path.parent...parent (depth of relative_path)
        depth = len(self.relative_path.parts)
        root = self.file_path
        for _ in range(depth):
            root = root.parent
        return root

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "project_id": self.project_id,
            "file_path": str(self.file_path),
            "relative_path": str(self.relative_path),
            "language": self.language,
            "file_checksum": self.file_checksum,
            "file_size": self.file_size,
            "file_mtime": self.file_mtime,
            "git_commit_hash": self.git_commit_hash,
            "git_file_hash": self.git_file_hash,
            "indexed_at": self.indexed_at.isoformat(),
            "chunk_count": self.chunk_count,
            "embedding_ids": self.embedding_ids,
            "status": self.status.value,
            "last_scanned": self.last_scanned.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            file_path=Path(data["file_path"]),
            relative_path=Path(data["relative_path"]),
            language=data["language"],
            file_checksum=data["file_checksum"],
            file_size=data["file_size"],
            file_mtime=data["file_mtime"],
            git_commit_hash=data.get("git_commit_hash"),
            git_file_hash=data.get("git_file_hash"),
            indexed_at=datetime.fromisoformat(data["indexed_at"]),
            chunk_count=data.get("chunk_count", 0),
            embedding_ids=data.get("embedding_ids", []),
            status=FileStatus(data.get("status", "active")),
            last_scanned=datetime.fromisoformat(data["last_scanned"]),
        )

    def has_changed(self, current_mtime: float, current_size: int) -> bool:
        """
        Quick check if file might have changed based on mtime and size.

        This is ~99% accurate and very fast (no file read required).
        """
        return self.file_mtime != current_mtime or self.file_size != current_size
