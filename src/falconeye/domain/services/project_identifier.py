"""
Project identification service.

This service identifies projects using a hybrid approach:
1. Git repositories: Use git remote URL + repo name
2. Non-git directories: Use path hash + directory name
3. Explicit override: Allow user to specify project ID
"""

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from ..value_objects.project_metadata import ProjectType


class ProjectIdentifier:
    """
    Service for identifying projects and generating unique project IDs.

    This uses a hybrid approach to ensure:
    - Same git repo → same project_id (regardless of location)
    - Different repos with same name → different project_ids
    - Non-git projects → deterministic path-based IDs
    - User can override for edge cases (monorepos, etc.)
    """

    def __init__(self):
        """Initialize the project identifier service."""
        pass

    def identify_project(
        self, path: Path, explicit_id: Optional[str] = None
    ) -> Tuple[str, str, ProjectType, Optional[str]]:
        """
        Identify a project and generate its unique ID.

        Args:
            path: Path to the project directory
            explicit_id: Optional explicit project ID override

        Returns:
            Tuple of (project_id, project_name, project_type, git_remote_url)

        Examples:
            Git repo:
                identify_project(Path("/path/to/myrepo"))
                → ("myrepo_a1b2c3d4", "myrepo", ProjectType.GIT, "github.com/user/myrepo")

            Non-git:
                identify_project(Path("/path/to/myproject"))
                → ("myproject_x9y8z7w6", "myproject", ProjectType.NON_GIT, None)

            Explicit:
                identify_project(Path("/path/to/monorepo/frontend"), explicit_id="frontend")
                → ("frontend", "frontend", ProjectType.GIT, "github.com/user/monorepo")
        """
        # Priority 1: Explicit user override
        if explicit_id:
            sanitized_id = self._sanitize_project_id(explicit_id)
            git_root = self._find_git_root(path)
            if git_root:
                remote_url = self._get_git_remote_url(git_root)
                return (sanitized_id, explicit_id, ProjectType.GIT, remote_url)
            return (sanitized_id, explicit_id, ProjectType.NON_GIT, None)

        # Priority 2: Git repository
        git_root = self._find_git_root(path)
        if git_root:
            repo_name = git_root.name
            remote_url = self._get_git_remote_url(git_root)

            if remote_url:
                # Hash remote URL to ensure uniqueness
                # This ensures same repo cloned to different locations has same ID
                url_hash = self._hash_string(remote_url)[:8]
                project_id = f"{self._sanitize_project_id(repo_name)}_{url_hash}"
                return (project_id, repo_name, ProjectType.GIT, remote_url)
            else:
                # Git repo but no remote (local repo)
                # Use repo name only
                project_id = self._sanitize_project_id(repo_name)
                return (project_id, repo_name, ProjectType.GIT, None)

        # Priority 3: Non-git directory
        dir_name = path.resolve().name
        path_hash = self._hash_string(str(path.resolve()))[:8]
        project_id = f"{self._sanitize_project_id(dir_name)}_{path_hash}"
        return (project_id, dir_name, ProjectType.NON_GIT, None)

    def _find_git_root(self, path: Path) -> Optional[Path]:
        """
        Find the git repository root containing the given path.

        Args:
            path: Path to search from

        Returns:
            Path to git root if found, None otherwise
        """
        current = path.resolve()

        # Traverse up the directory tree
        while current != current.parent:
            git_dir = current / ".git"
            if git_dir.exists():
                return current
            current = current.parent

        return None

    def _get_git_remote_url(self, git_root: Path) -> Optional[str]:
        """
        Get the git remote URL for a repository.

        Args:
            git_root: Path to git repository root

        Returns:
            Remote URL if found, None otherwise
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(git_root), "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                url = result.stdout.strip()
                # Normalize URL (remove .git suffix, convert SSH to HTTPS format)
                return self._normalize_git_url(url)

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Git not available or command failed
            return None

    def _normalize_git_url(self, url: str) -> str:
        """
        Normalize git URL to consistent format.

        Converts:
            git@github.com:user/repo.git → github.com/user/repo
            https://github.com/user/repo.git → github.com/user/repo
            https://github.com/user/repo → github.com/user/repo

        Args:
            url: Git remote URL

        Returns:
            Normalized URL
        """
        # Remove .git suffix
        if url.endswith(".git"):
            url = url[:-4]

        # Convert SSH format to HTTPS-like format
        # git@github.com:user/repo → github.com/user/repo
        ssh_pattern = r"^git@([^:]+):(.+)$"
        match = re.match(ssh_pattern, url)
        if match:
            host, path = match.groups()
            return f"{host}/{path}"

        # Remove https:// or http:// prefix
        url = re.sub(r"^https?://", "", url)

        return url

    def _sanitize_project_id(self, project_id: str) -> str:
        """
        Sanitize project ID to be safe for use in collection names.

        Removes or replaces characters that might cause issues in ChromaDB
        collection names or file systems.

        Args:
            project_id: Raw project ID

        Returns:
            Sanitized project ID
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure not empty
        if not sanitized:
            sanitized = "project"

        # Ensure doesn't start with a number (some systems don't like this)
        if sanitized[0].isdigit():
            sanitized = f"p{sanitized}"

        return sanitized.lower()

    def _hash_string(self, text: str) -> str:
        """
        Generate SHA256 hash of a string.

        Args:
            text: String to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get_current_git_commit(self, git_root: Path) -> Optional[str]:
        """
        Get the current git commit hash.

        Args:
            git_root: Path to git repository root

        Returns:
            Commit hash if available, None otherwise
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(git_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return result.stdout.strip()

            return None

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return None

    def has_uncommitted_changes(self, git_root: Path) -> bool:
        """
        Check if git repository has uncommitted changes.

        Args:
            git_root: Path to git repository root

        Returns:
            True if there are uncommitted changes, False otherwise
        """
        try:
            # Check for staged and unstaged changes
            result = subprocess.run(
                ["git", "-C", str(git_root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # If output is not empty, there are changes
                return bool(result.stdout.strip())

            return True  # Assume changes if command failed

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return True  # Assume changes if git not available

    def get_git_changed_files(
        self, git_root: Path, from_commit: Optional[str] = None
    ) -> list[Path]:
        """
        Get list of files changed in git since a specific commit.

        Args:
            git_root: Path to git repository root
            from_commit: Commit to compare from (None = all uncommitted changes)

        Returns:
            List of changed file paths (relative to git root)
        """
        try:
            if from_commit:
                # Get files changed between commit and HEAD
                cmd = ["git", "-C", str(git_root), "diff", "--name-only", from_commit, "HEAD"]
            else:
                # Get uncommitted changes
                cmd = ["git", "-C", str(git_root), "diff", "--name-only", "HEAD"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                return [git_root / f for f in files if f]

            return []

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return []

    def get_git_untracked_files(self, git_root: Path) -> list[Path]:
        """
        Get list of untracked files in git repository.

        Args:
            git_root: Path to git repository root

        Returns:
            List of untracked file paths (relative to git root)
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(git_root), "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                return [git_root / f for f in files if f]

            return []

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return []
