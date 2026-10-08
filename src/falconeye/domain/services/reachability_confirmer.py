"""
Reachability confirmer: check whether a finding's enclosing method is
reachable from any known entry point through the call graph.

Given:
  * A built :class:`CallGraph`
  * A list of :class:`EntryPoint` s (roots)
  * The finding's ``(file_path, line_number)``

The confirmer:
  1. Locates the FunctionDef whose Location contains the finding's
     line number. This becomes the reachability *target*.
  2. Runs BFS from each entry point through the call graph. The first
     entry to reach the target contributes the shortest chain.
  3. Returns a :class:`TaintConfirmation` describing the outcome.

BFS is bounded by ``max_hops`` per traversal to avoid pathological
call graphs (e.g. a class-hierarchy with many virtual dispatches
producing a densely-connected graph) from making confirmation slow.
Default is generous (100 hops); real ASP.NET Core entry-to-sink chains
are typically 2-6 hops.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

from ..models.confirmation import (
    CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
    CONFIRMATION_STATUS_NO_ENTRY_POINTS,
    CONFIRMATION_STATUS_NOT_CONFIRMED,
    CONFIRMATION_STATUS_REACHABLE,
    EntryPoint,
    TaintConfirmation,
)
from ..models.taint import FunctionDef
from .call_graph import CallGraph


_logger = logging.getLogger(__name__)


class ReachabilityConfirmer:
    """Confirms reachability of source locations from entry points.

    Instances are workspace-scoped and immutable after construction.
    Construct once per workspace; call :meth:`confirm` per finding.
    """

    #: BFS traversal hop budget. Real callchains are short (2-6 hops).
    #: The cap exists to guarantee termination on dense graphs; not to
    #: model a semantic constraint.
    DEFAULT_MAX_HOPS = 100

    def __init__(
        self,
        *,
        call_graph: CallGraph,
        function_defs: Sequence[FunctionDef],
        entry_points: Sequence[EntryPoint],
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> None:
        self._graph = call_graph
        self._entry_points = list(entry_points)
        self._max_hops = max_hops

        # File -> list of (start_line, end_line, canonical_name), sorted
        # by start_line. Used to answer "which function contains
        # line N in file F?" for finding locations.
        self._file_index: Dict[str, List[Tuple[int, int, str]]] = {}
        for f in function_defs:
            entries = self._file_index.setdefault(f.location.file_path, [])
            entries.append(
                (f.location.line_start, f.location.line_end, f.canonical_name)
            )
        for lst in self._file_index.values():
            lst.sort()

        # Also index by normalized (basename + posix-form) path so
        # findings coming from an older report with different path
        # separators / drive-letter casing can still resolve.
        self._file_index_normalized: Dict[str, List[Tuple[int, int, str]]] = {
            _normalize_path(k): v for k, v in self._file_index.items()
        }

    # -- Public API ---------------------------------------------------

    def find_enclosing_function(
        self,
        file_path: str,
        line_number: int,
    ) -> Optional[str]:
        """Return the canonical name of the FunctionDef containing the given line.

        Tries an exact path match first, then a normalized-path match
        (basename + POSIX separators) to accommodate findings from
        reports built on a different OS. Returns None if no method
        contains the line -- usually meaning the finding is on a
        top-level statement or a file that didn't parse.
        """
        candidates = self._file_index.get(file_path)
        if candidates is None:
            candidates = self._file_index_normalized.get(_normalize_path(file_path))
        if candidates is None:
            return None
        # Smallest containing method (in case of nested lambdas etc.)
        best: Optional[Tuple[int, int, str]] = None
        for start, end, name in candidates:
            if start <= line_number <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, name)
        return best[2] if best is not None else None

    def confirm(
        self,
        file_path: str,
        line_number: int,
    ) -> TaintConfirmation:
        """Confirm reachability for a finding at ``file_path:line_number``.

        Handles the four outcomes:
          * No entry points detected in the workspace -> no_entry_points
          * Finding location has no enclosing method -> no_enclosing_method
          * Enclosing method is reachable from >=1 entry -> confirmed_reachable
          * Enclosing method exists but no entry reaches it -> not_confirmed

        Never raises; malformed inputs degrade to ``not_confirmed`` with
        a diagnostic ``reason``.
        """
        if not self._entry_points:
            return TaintConfirmation(
                status=CONFIRMATION_STATUS_NO_ENTRY_POINTS,
                reason=(
                    "No entry points detected in this workspace. "
                    "Reachability confirmation was skipped."
                ),
            )

        target = self.find_enclosing_function(file_path, line_number)
        if target is None:
            return TaintConfirmation(
                status=CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
                reason=(
                    f"No enclosing method contains {file_path}:{line_number}. "
                    "The finding may be on a top-level statement or in a file "
                    "that could not be parsed."
                ),
            )

        reachable_entries: List[str] = []
        shortest_chain: Optional[List[str]] = None
        for entry in self._entry_points:
            chain = self._find_path(entry.canonical_name, target)
            if chain is None:
                continue
            reachable_entries.append(entry.canonical_name)
            if shortest_chain is None or len(chain) < len(shortest_chain):
                shortest_chain = chain

        if reachable_entries:
            hops = (len(shortest_chain) - 1) if shortest_chain else 0
            entries_summary = (
                f"{len(reachable_entries)} entry point"
                f"{'s' if len(reachable_entries) != 1 else ''}"
            )
            return TaintConfirmation(
                status=CONFIRMATION_STATUS_REACHABLE,
                enclosing_function=target,
                reachable_entry_points=tuple(reachable_entries),
                shortest_chain=tuple(shortest_chain or ()),
                hop_count=hops,
                reason=(
                    f"Enclosing method '{target}' is reachable from "
                    f"{entries_summary} ({hops} hop{'s' if hops != 1 else ''} "
                    "via the shortest chain)."
                ),
            )

        return TaintConfirmation(
            status=CONFIRMATION_STATUS_NOT_CONFIRMED,
            enclosing_function=target,
            reason=(
                f"Enclosing method '{target}' is not reachable from any of the "
                f"{len(self._entry_points)} detected entry points. This may "
                "indicate dead code, entry-point-detection gaps, or dispatch "
                "that is not visible to static call-graph analysis (reflection, "
                "dependency injection, event handlers)."
            ),
        )

    # -- Internals ----------------------------------------------------

    def _find_path(self, source: str, target: str) -> Optional[List[str]]:
        """BFS from source to target through call-graph successors.

        Returns the shortest path (list of canonical names, source
        first, target last) or None if no path exists within
        ``max_hops``. Trivially handles source == target.
        """
        if source == target:
            return [source]
        if source not in self._graph:
            return None

        visited = {source}
        # Queue holds (node, path_to_node); path_to_node is inclusive.
        queue: deque = deque()
        queue.append((source, [source]))

        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= self._max_hops:
                continue
            for neighbor in self._graph.successors(current):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == target:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        return None


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _normalize_path(p: str) -> str:
    """Normalize a file path for cross-platform comparison.

    Replaces backslashes with forward slashes, lowercases (Windows
    paths are case-insensitive), and drops any drive-letter prefix.
    Not perfect but handles the common cases: finding produced on
    Windows compared against workspace analyzed on Linux, or vice
    versa.
    """
    p = p.replace("\\", "/")
    # Drop drive letter like "C:" if present.
    if len(p) >= 2 and p[1] == ":":
        p = p[2:]
    p = p.lower()
    return os.path.normpath(p)
