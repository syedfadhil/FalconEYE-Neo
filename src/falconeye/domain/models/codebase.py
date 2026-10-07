"""Codebase domain models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass
class CodeFile:
    """Value object representing a source code file."""
    path: Path
    relative_path: str
    content: str
    language: str
    size_bytes: int
    line_count: int

    @classmethod
    def create(
        cls,
        path: Path,
        relative_path: str,
        content: str,
        language: str,
    ) -> "CodeFile":
        """Factory method to create a code file."""
        return cls(
            path=path,
            relative_path=relative_path,
            content=content,
            language=language,
            size_bytes=len(content.encode('utf-8')),
            line_count=len(content.splitlines()),
        )

    @property
    def extension(self) -> str:
        """Get file extension."""
        return self.path.suffix


@dataclass
class Codebase:
    """
    Aggregate root representing a codebase to be analyzed.

    The codebase is the main entity that security reviews operate on.
    """
    id: UUID
    root_path: Path
    language: str
    files: List[CodeFile] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        root_path: Path,
        language: str,
        excluded_patterns: Optional[List[str]] = None,
    ) -> "Codebase":
        """Factory method to create a codebase."""
        if not root_path.exists():
            from ..exceptions import InvalidCodebaseError
            raise InvalidCodebaseError(f"Path does not exist: {root_path}")

        if not root_path.is_dir():
            from ..exceptions import InvalidCodebaseError
            raise InvalidCodebaseError(f"Path is not a directory: {root_path}")

        return cls(
            id=uuid4(),
            root_path=root_path,
            language=language,
            excluded_patterns=excluded_patterns or [],
        )

    def add_file(self, file: CodeFile) -> None:
        """Add a file to the codebase."""
        self.files.append(file)

    @property
    def total_files(self) -> int:
        """Get total number of files."""
        return len(self.files)

    @property
    def total_lines(self) -> int:
        """Get total lines of code."""
        return sum(f.line_count for f in self.files)

    @property
    def total_size_bytes(self) -> int:
        """Get total size in bytes."""
        return sum(f.size_bytes for f in self.files)

    @property
    def all_languages(self) -> List[str]:
        """
        Get all unique languages present in the codebase files.
        
        Returns:
            List of unique language names sorted by file count (descending)
        """
        if not self.files:
            return [self.language]
        
        # Count files per language
        from collections import Counter
        language_counts = Counter(f.language for f in self.files)
        
        # Sort by count (descending)
        sorted_languages = sorted(
            language_counts.keys(),
            key=lambda lang: language_counts[lang],
            reverse=True
        )
        
        return sorted_languages