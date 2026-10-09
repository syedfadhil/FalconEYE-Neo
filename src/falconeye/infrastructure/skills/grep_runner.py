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

The workspace is walked ONCE per discovery run and each file is read
once, with every applicable pattern evaluated against it. When the
total cap binds, it is shared fairly across patterns rather than
first-come-first-served, so a skill's later patterns (injection-
prevention's LDAP / log / header checks come after four SQL and
command patterns) keep their share of the budget.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from ...domain.models.skill import DiscoveryHit, DiscoveryPattern
from ..workspace_files import iter_workspace_files


_logger = logging.getLogger(__name__)


# Directories always skipped. Kept short and non-controversial;
# operators can extend via FileDiscoveryConfig.default_exclusions but
# that plumbing is deferred until Phase 1b. Matched against paths
# relative to the workspace root (see workspace_files).
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
    # Exposed so the LLM-only sweep skips exactly what discovery skips.
    DEFAULT_SKIP_FRAGMENTS = _DEFAULT_SKIP_FRAGMENTS

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

        Hits are returned in pattern order, then file walk order
        (sorted, so stable across runs), then line order within a file.
        Each pattern keeps at most ``max_hits_per_pattern`` hits. When
        the patterns together exceed ``max_total_hits``, the budget is
        divided fairly (see :func:`_fair_share`) and a warning names
        the patterns that were trimmed.
        """
        compiled = self._compile(list(patterns))
        if not compiled:
            return []

        per_pattern: List[List[DiscoveryHit]] = [[] for _ in compiled]
        all_exts: Set[str] = set().union(*(exts for _, _, exts in compiled))

        for file_path in iter_workspace_files(
            self._root,
            extensions=all_exts,
            skip_fragments=self._skip_fragments,
            max_file_bytes=self._max_file_bytes,
        ):
            suffix = file_path.suffix.lower()
            active = [
                i for i, (_, _, exts) in enumerate(compiled)
                if suffix in exts
                and len(per_pattern[i]) < self._max_hits_per_pattern
            ]
            if not active:
                if all(
                    len(h) >= self._max_hits_per_pattern for h in per_pattern
                ):
                    break  # every pattern is full; stop walking
                continue
            self._scan_file(file_path, compiled, active, per_pattern)

        return self._apply_total_cap(compiled, per_pattern)

    # ---- internals ----

    def _compile(self, patterns: List[DiscoveryPattern]):
        """Return ``[(pattern, compiled_regex, extensions), ...]``.

        Invalid regexes and patterns without extensions are dropped
        with a debug log so one bad pattern doesn't stop discovery.
        """
        out = []
        for pattern in patterns:
            try:
                regex = re.compile(pattern.regex)
            except re.error as e:
                # Skill authors sometimes write shell-safe regexes that
                # are not valid Python.
                _logger.debug(
                    "Skipping invalid regex from skill: %s (error: %s)",
                    pattern.regex[:80],
                    e,
                )
                continue
            extensions = _normalize_extensions(pattern.extensions)
            if not extensions:
                _logger.debug(
                    "Skipping pattern with no extensions: %s",
                    pattern.regex[:80],
                )
                continue
            out.append((pattern, regex, extensions))
        return out

    def _scan_file(
        self,
        file_path: Path,
        compiled,
        active: List[int],
        per_pattern: List[List[DiscoveryHit]],
    ) -> None:
        """Run the ``active`` patterns over one file, appending hits.

        Reads once and iterates line-by-line so we know accurate line
        numbers. Errors reading the file (binary content, permission
        denied) are logged at debug and treated as zero matches.
        """
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            _logger.debug("Cannot read %s: %s", file_path, e)
            return

        cap = self._max_hits_per_pattern
        for line_no, line in enumerate(content.splitlines(), start=1):
            for i in active:
                if len(per_pattern[i]) >= cap:
                    continue
                pattern, regex, _ = compiled[i]
                m = regex.search(line)
                if m is None:
                    continue
                per_pattern[i].append(
                    DiscoveryHit(
                        file_path=str(file_path),
                        line_number=line_no,
                        line_text=line,
                        matched_span=(m.start(), m.end()),
                        pattern_description=pattern.description,
                    )
                )

    def _apply_total_cap(self, compiled, per_pattern) -> List[DiscoveryHit]:
        counts = [len(h) for h in per_pattern]
        allocation = _fair_share(counts, self._max_total_hits)
        trimmed = [
            (compiled[i][0], counts[i], allocation[i])
            for i in range(len(counts))
            if allocation[i] < counts[i]
        ]
        if trimmed:
            _logger.warning(
                "Discovery capped at %d hits (max_total_hits); trimmed %s",
                self._max_total_hits,
                ", ".join(
                    f"{p.description or p.regex[:40]!r} {have}->{kept}"
                    for p, have, kept in trimmed
                ),
            )
        hits: List[DiscoveryHit] = []
        for i, pattern_hits in enumerate(per_pattern):
            hits.extend(pattern_hits[: allocation[i]])
        return hits


def _fair_share(counts: List[int], budget: int) -> List[int]:
    """Split ``budget`` across patterns that found ``counts`` hits.

    Water-filling: every pattern with hits gets an equal share; any
    share a pattern can't use (it found fewer hits) is redistributed
    to the patterns that still have hits left. When the budget doesn't
    divide evenly, earlier patterns get the extra hit. If the counts
    fit in the budget, every hit is kept.
    """
    allocation = [0] * len(counts)
    remaining = max(budget, 0)
    active = [i for i, c in enumerate(counts) if c > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        still_active = []
        for i in active:
            take = min(share, counts[i] - allocation[i], remaining)
            allocation[i] += take
            remaining -= take
            if allocation[i] < counts[i]:
                still_active.append(i)
            if remaining == 0:
                break
        active = still_active
    return allocation


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
