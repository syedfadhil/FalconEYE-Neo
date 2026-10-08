"""
Pure-Python regex discovery for skill Step 1 patterns.

Walks a workspace and runs a :class:`DiscoveryPattern`'s regex against
every file matching the pattern's extensions. No subprocess, no shell,
no dependency on ``grep`` being installed -- so this works on Windows
without WSL and doesn't pull in ripgrep or POSIX grep.

Deliberately bounded: caps hits per pattern, total hits, and per-file
scan cost. Skills' regexes are broad by design ("dangerous-looking
things"), so an unbounded run can produce thousands of hits and starve
the LLM stage that follows. Callers can override the caps.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ...domain.models.skill import DiscoveryHit, DiscoveryPattern


_logger = logging.getLogger(__name__)


# Directories always skipped. Kept short and non-controversial;
# operators can extend via FileDiscoveryConfig.default_exclusions but
# that plumbing is deferred until Phase 1b.
_DEFAULT_SKIP_FRAGMENTS = (
    "/.git/",
    "/.hg/",
    "/.svn/",
    "/node_modules/",
    "/bower_components/",
    "/vendor/",
    "/.venv/",
    "/venv/",
    "/env/",
    "/__pycache__/",
    "/.mypy_cache/",
    "/.pytest_cache/",
    "/bin/",
    "/obj/",
    "/dist/",
    "/build/",
    "/target/",
    "/out/",
)


class GrepRunner:
    """Runs discovery patterns against a workspace.

    Immutable across calls; a single instance can be reused for the
    same workspace across many patterns.
    """

    # Sensible default caps. Skills can be broader than one screen of
    # findings; these limits keep the runner responsive without dropping
    # so much that real signals get lost.
    DEFAULT_MAX_HITS_PER_PATTERN = 40
    DEFAULT_MAX_TOTAL_HITS = 200
    DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

    def __init__(
        self,
        *,
        workspace_root: Path,
        max_hits_per_pattern: int = DEFAULT_MAX_HITS_PER_PATTERN,
        max_total_hits: int = DEFAULT_MAX_TOTAL_HITS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        extra_skip_fragments: Iterable[str] = (),
    ) -> None:
        self._root = Path(workspace_root)
        self._max_hits_per_pattern = max_hits_per_pattern
        self._max_total_hits = max_total_hits
        self._max_file_bytes = max_file_bytes
        self._skip_fragments = tuple(_DEFAULT_SKIP_FRAGMENTS) + tuple(
            extra_skip_fragments
        )

    def discover(
        self,
        patterns: Iterable[DiscoveryPattern],
    ) -> List[DiscoveryHit]:
        """Run every pattern against the workspace, returning all hits.

        Hits are returned in the order they were found (pattern order,
        then file walk order, then line order within a file). When
        ``max_total_hits`` is exceeded, discovery stops immediately --
        remaining patterns are not run. A warning is logged so the
        operator knows the result is truncated.
        """
        all_hits: List[DiscoveryHit] = []
        for pattern in patterns:
            if len(all_hits) >= self._max_total_hits:
                _logger.warning(
                    "Discovery truncated at %d hits (max_total_hits reached); "
                    "%d patterns not run.",
                    self._max_total_hits,
                    _remaining_count(patterns, pattern),
                )
                break
            pattern_hits = self._discover_one(pattern)
            remaining_budget = self._max_total_hits - len(all_hits)
            if len(pattern_hits) > remaining_budget:
                pattern_hits = pattern_hits[:remaining_budget]
            all_hits.extend(pattern_hits)
        return all_hits

    def _discover_one(self, pattern: DiscoveryPattern) -> List[DiscoveryHit]:
        """Discover hits for a single pattern.

        Compiles the regex once, walks files with matching extensions,
        stops early when ``max_hits_per_pattern`` is hit.
        """
        try:
            compiled = re.compile(pattern.regex)
        except re.error as e:
            # Skill authors sometimes write shell-safe regexes that are
            # not valid Python. Skip with a debug log so we don't
            # explode the whole discovery run.
            _logger.debug(
                "Skipping invalid regex from skill: %s (error: %s)",
                pattern.regex[:80],
                e,
            )
            return []

        extensions = _normalize_extensions(pattern.extensions)
        if not extensions:
            _logger.debug(
                "Skipping pattern with no extensions: %s",
                pattern.regex[:80],
            )
            return []

        hits: List[DiscoveryHit] = []
        for file_path in self._iter_files(extensions):
            if len(hits) >= self._max_hits_per_pattern:
                break
            file_hits = self._scan_file(file_path, compiled, pattern)
            if not file_hits:
                continue
            budget = self._max_hits_per_pattern - len(hits)
            hits.extend(file_hits[:budget])
        return hits

    def _iter_files(self, extensions: Set[str]) -> Iterable[Path]:
        """Yield every file under the workspace matching ``extensions``.

        Skips files inside default_skip_fragments and any file whose
        size exceeds ``max_file_bytes``.
        """
        if not self._root.exists():
            return
        for candidate in self._root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in extensions:
                continue
            posix = candidate.as_posix()
            if any(frag in posix for frag in self._skip_fragments):
                continue
            try:
                if candidate.stat().st_size > self._max_file_bytes:
                    continue
            except OSError:
                continue
            yield candidate

    def _scan_file(
        self,
        file_path: Path,
        compiled: re.Pattern,
        pattern: DiscoveryPattern,
    ) -> List[DiscoveryHit]:
        """Scan a single file for regex matches.

        Reads once and iterates line-by-line so we know accurate line
        numbers. Errors reading the file (binary content, permission
        denied) are logged at debug and treated as zero matches.
        """
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            _logger.debug("Cannot read %s: %s", file_path, e)
            return []

        hits: List[DiscoveryHit] = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            m = compiled.search(line)
            if m is None:
                continue
            hits.append(
                DiscoveryHit(
                    file_path=str(file_path),
                    line_number=line_no,
                    line_text=line,
                    matched_span=(m.start(), m.end()),
                    pattern_description=pattern.description,
                )
            )
        return hits


def _normalize_extensions(extensions: Iterable[str]) -> Set[str]:
    """Ensure extensions have leading dots and lowercase."""
    out: Set[str] = set()
    for e in extensions:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.add(e)
    return out


def _remaining_count(
    patterns: Iterable[DiscoveryPattern],
    stopped_at: DiscoveryPattern,
) -> int:
    """Count how many patterns are left after ``stopped_at``.

    Best-effort -- ``patterns`` may be a generator that's already been
    consumed by the caller, in which case this returns 0.
    """
    seen = False
    count = 0
    try:
        for p in patterns:
            if seen:
                count += 1
            elif p is stopped_at:
                seen = True
    except Exception:
        return 0
    return count
