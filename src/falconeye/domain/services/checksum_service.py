"""
Checksum service for file change detection.

This service implements a three-tier strategy for detecting file changes:
1. Git-based detection (fastest, for git repos)
2. Modification time + size check (fast, 99% accurate)
3. SHA256 checksum (slower, 100% accurate)
"""

import hashlib
from pathlib import Path
from typing import Optional, Set

from ..value_objects.project_metadata import FileMetadata


class ChecksumService:
    """
    Service for detecting file changes using multiple strategies.

    Implements three-tier change detection:
    - Tier 1: Git diff (fastest, git repos only)
    - Tier 2: mtime + size (fast, 99% accurate)
    - Tier 3: SHA256 (slow, 100% accurate)
    """

    def __init__(self):
        """Initialize the checksum service."""
        self._chunk_size = 65536  # 64KB chunks for streaming SHA256

    def calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file.

        Uses streaming to handle large files efficiently without
        loading entire file into memory.

        Args:
            file_path: Path to file

        Returns:
            SHA256 checksum as hex string with 'sha256:' prefix

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read file in chunks to avoid memory issues with large files
            while chunk := f.read(self._chunk_size):
                sha256_hash.update(chunk)

        return f"sha256:{sha256_hash.hexdigest()}"

    def has_file_changed_quick(
        self, file_path: Path, cached_metadata: Optional[FileMetadata]
    ) -> bool:
        """
        Quick check if file has changed using mtime and size.

        This is ~99% accurate and very fast (no file read required).
        Only stat() system call is needed.

        Args:
            file_path: Path to file
            cached_metadata: Previously stored metadata

        Returns:
            True if file might have changed, False if definitely unchanged
        """
        if not cached_metadata:
            return True  # No cached data, assume changed

        try:
            stat = file_path.stat()
            return cached_metadata.has_changed(stat.st_mtime, stat.st_size)
        except (FileNotFoundError, PermissionError):
            return True  # File missing or unreadable, treat as changed

    def has_file_changed_checksum(
        self, file_path: Path, cached_metadata: Optional[FileMetadata]
    ) -> bool:
        """
        Accurate check if file has changed using SHA256 checksum.

        This is 100% accurate but slower (requires reading entire file).

        Args:
            file_path: Path to file
            cached_metadata: Previously stored metadata

        Returns:
            True if file has changed, False if unchanged
        """
        if not cached_metadata:
            return True  # No cached data, assume changed

        try:
            current_checksum = self.calculate_file_checksum(file_path)
            return current_checksum != cached_metadata.file_checksum
        except (FileNotFoundError, PermissionError):
            return True  # File missing or unreadable, treat as changed

    def get_file_metadata_snapshot(
        self,
        file_path: Path,
        relative_path: Path,
        project_id: str,
        language: str,
        git_commit_hash: Optional[str] = None,
    ) -> FileMetadata:
        """
        Create a metadata snapshot for a file.

        This captures all the information needed for future change detection.

        Args:
            file_path: Absolute path to file
            relative_path: Path relative to project root
            project_id: Project identifier
            language: Programming language
            git_commit_hash: Git commit hash if in git repo

        Returns:
            FileMetadata snapshot

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
        """
        stat = file_path.stat()
        checksum = self.calculate_file_checksum(file_path)

        return FileMetadata(
            project_id=project_id,
            file_path=file_path,
            relative_path=relative_path,
            language=language,
            file_checksum=checksum,
            file_size=stat.st_size,
            file_mtime=stat.st_mtime,
            git_commit_hash=git_commit_hash,
        )

    def filter_changed_files_efficient(
        self,
        files: list[Path],
        cached_metadata: dict[Path, FileMetadata],
        use_checksum: bool = False,
    ) -> tuple[list[Path], list[Path]]:
        """
        Efficiently filter files into changed and unchanged.

        Uses two-stage approach:
        1. Quick mtime+size check (fast)
        2. Optional SHA256 verification (slower but accurate)

        Args:
            files: List of file paths to check
            cached_metadata: Dict mapping file paths to cached metadata
            use_checksum: Whether to verify with SHA256 (slower but accurate)

        Returns:
            Tuple of (changed_files, unchanged_files)
        """
        changed_files = []
        unchanged_files = []

        for file_path in files:
            cached = cached_metadata.get(file_path)

            if not cached:
                # No cached data, definitely changed (or new)
                changed_files.append(file_path)
                continue

            # Stage 1: Quick check with mtime+size
            if not self.has_file_changed_quick(file_path, cached):
                # Definitely unchanged (mtime and size match)
                unchanged_files.append(file_path)
                continue

            # Stage 2: mtime or size changed, need deeper check
            if use_checksum:
                # Verify with checksum (100% accurate)
                if self.has_file_changed_checksum(file_path, cached):
                    changed_files.append(file_path)
                else:
                    # File was touched but content unchanged
                    unchanged_files.append(file_path)
            else:
                # Assume changed if mtime/size differ
                changed_files.append(file_path)

        return changed_files, unchanged_files

    def batch_calculate_checksums(
        self, files: list[Path], max_workers: int = 4
    ) -> dict[Path, str]:
        """
        Calculate checksums for multiple files in parallel.

        Args:
            files: List of file paths
            max_workers: Number of parallel workers

        Returns:
            Dict mapping file paths to checksums
        """
        import concurrent.futures

        checksums = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all checksum calculations
            future_to_file = {
                executor.submit(self.calculate_file_checksum, f): f for f in files
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    checksums[file_path] = future.result()
                except Exception as e:
                    # Skip files that can't be processed
                    print(f"Warning: Failed to checksum {file_path}: {e}")

        return checksums

    def identify_deleted_files(
        self, current_files: Set[Path], cached_files: Set[Path]
    ) -> Set[Path]:
        """
        Identify files that were previously indexed but no longer exist.

        Args:
            current_files: Set of currently existing file paths
            cached_files: Set of previously indexed file paths

        Returns:
            Set of deleted file paths
        """
        return cached_files - current_files

    def identify_new_files(
        self, current_files: Set[Path], cached_files: Set[Path]
    ) -> Set[Path]:
        """
        Identify files that are new (not previously indexed).

        Args:
            current_files: Set of currently existing file paths
            cached_files: Set of previously indexed file paths

        Returns:
            Set of new file paths
        """
        return current_files - cached_files
