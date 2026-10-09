"""
Orchestrator: run reachability confirmation over a batch of findings.

Takes a list of :class:`SecurityFinding` s produced by ``falconeye
review`` (or loaded from a saved findings JSON) plus a repo path, and
returns the same findings enriched with :class:`TaintConfirmation`
metadata.

The service builds the call graph itself using the same
:class:`CallGraphBuilder` and adapter registry that the taint pipeline
uses. This means adding a new language for confirmation requires only
a :class:`TaintAdapter` and an :class:`EntryPointDetector` -- no
per-service plumbing.

Findings for files whose language has no registered detector are
returned with ``taint_confirmation.status = 'skipped'`` so downstream
consumers can distinguish "we tried and couldn't confirm" from "we
didn't try".
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from ...domain.models.confirmation import (
    CONFIRMATION_STATUS_SKIPPED,
    TaintConfirmation,
)
from ...domain.models.security import SecurityFinding
from ...domain.models.taint import FunctionDef
from ...domain.services.call_graph import CallGraphBuilder
from ...domain.services.entry_point_detector import EntryPointDetectorRegistry
from ...domain.services.reachability_confirmer import ReachabilityConfirmer
from ...infrastructure.taint_adapters.adapter_registry import AdapterRegistry
from ...infrastructure.taint_index import TaintIndex
from ...infrastructure.workspace_files import iter_workspace_files


_logger = logging.getLogger(__name__)


# Extension -> language mapping. Kept locally for now; the taint scan
# service has its own map and we can consolidate later. Only languages
# with both a taint adapter and an entry-point detector are useful for
# confirmation.
_LANGUAGE_EXTENSIONS: Dict[str, Set[str]] = {
    "csharp": {".cs"},
}


@dataclasses.dataclass(frozen=True)
class ConfirmationSummary:
    """Aggregate counts across a confirmation run.

    Exposed so the CLI can print a one-line diagnostic like:
        Confirmation: 12 confirmed, 3 not confirmed, 1 no method, 4 skipped
    """
    total_findings: int = 0
    reachable: int = 0
    not_confirmed: int = 0
    no_enclosing_method: int = 0
    no_entry_points: int = 0
    skipped: int = 0
    entry_points_detected: int = 0
    call_graph_nodes: int = 0
    call_graph_edges: int = 0


class ReviewConfirmationService:
    """Confirms reachability of review findings from workspace entry points.

    Instances are stateless across calls; a single instance can be
    reused for different workspaces (each call takes ``repo_path``).
    """

    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        entry_point_registry: EntryPointDetectorRegistry,
        taint_index: TaintIndex,
    ) -> None:
        self._adapter_registry = adapter_registry
        self._entry_point_registry = entry_point_registry
        # A taint index is required by CallGraphBuilder even though
        # we don't do path-finding here. Empty specs are fine -- the
        # builder uses the index for sink resolution, which we ignore.
        self._taint_index = taint_index

    def confirm(
        self,
        *,
        findings: List[SecurityFinding],
        repo_path: Path,
        languages: Iterable[str],
    ) -> Tuple[List[SecurityFinding], ConfirmationSummary]:
        """Confirm reachability for every finding.

        Returns a new list of findings with ``taint_confirmation``
        populated on each, plus a summary of counts.
        """
        languages_list = list(languages)
        supported_languages = [
            lang for lang in languages_list
            if self._entry_point_registry.get(lang) is not None
            and self._adapter_registry.get(lang) is not None
        ]

        if not supported_languages:
            _logger.warning(
                "No languages both have taint adapters and entry-point "
                "detectors registered. All findings will be marked skipped."
            )
            return self._all_skipped(findings, reason=(
                "No supported languages for confirmation. "
                f"Requested: {languages_list}. Configure a taint adapter "
                "and an entry-point detector for at least one language."
            ))

        # ---------- Build the call graph ----------
        builder = CallGraphBuilder(
            adapter_registry=self._adapter_registry,
            taint_index=self._taint_index,
        )
        extensions: Set[str] = set()
        for lang in supported_languages:
            extensions |= _LANGUAGE_EXTENSIONS.get(lang, set())

        source_by_file: Dict[str, str] = {}
        files_added = 0
        for path in self._walk_files(repo_path, extensions):
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                _logger.warning("Could not read %s: %s", path, e)
                continue
            lang = _language_for_extension(path.suffix)
            if lang is None or lang not in supported_languages:
                continue
            builder.add_file(source=source, file_path=str(path), language=lang)
            source_by_file[str(path)] = source
            files_added += 1

        graph = builder.build()
        _logger.info(
            "Confirmation call graph: %d nodes, %d edges from %d files",
            len(graph), graph.edge_count(), files_added,
        )

        # ---------- Detect entry points ----------
        function_defs: List[FunctionDef] = []
        for node in graph.user_defined_nodes():
            fd = graph.function(node)
            if fd is not None:
                function_defs.append(fd)

        entry_points = self._entry_point_registry.detect_all(
            function_defs=function_defs,
            source_by_file=source_by_file,
        )
        _logger.info("Entry points detected: %d", len(entry_points))

        # ---------- Confirm each finding ----------
        confirmer = ReachabilityConfirmer(
            call_graph=graph,
            function_defs=function_defs,
            entry_points=entry_points,
        )

        confirmed_findings: List[SecurityFinding] = []
        counts = {
            "reachable": 0,
            "not_confirmed": 0,
            "no_enclosing_method": 0,
            "no_entry_points": 0,
            "skipped": 0,
        }

        for finding in findings:
            # Skip findings whose file extension is not in the
            # supported set -- we couldn't have parsed them anyway.
            ext = _extension_of(finding.file_path)
            file_lang = _language_for_extension(ext) if ext else None
            if file_lang is None or file_lang not in supported_languages:
                confirmed_findings.append(
                    self._with_confirmation(
                        finding,
                        TaintConfirmation(
                            status=CONFIRMATION_STATUS_SKIPPED,
                            reason=(
                                f"Finding language ({file_lang or 'unknown'}) "
                                "has no entry-point detector registered."
                            ),
                        ),
                    )
                )
                counts["skipped"] += 1
                continue

            line = finding.line_start or 1
            confirmation = confirmer.confirm(
                file_path=finding.file_path,
                line_number=line,
            )
            confirmed_findings.append(
                self._with_confirmation(finding, confirmation)
            )
            key = confirmation.status
            if key in counts:
                counts[key] += 1

        summary = ConfirmationSummary(
            total_findings=len(findings),
            reachable=counts["reachable"] if "reachable" in counts else 0,
            not_confirmed=counts["not_confirmed"],
            no_enclosing_method=counts["no_enclosing_method"],
            no_entry_points=counts["no_entry_points"],
            skipped=counts["skipped"],
            entry_points_detected=len(entry_points),
            call_graph_nodes=len(graph),
            call_graph_edges=graph.edge_count(),
        )
        # The counts dict uses status-string keys; map "confirmed_reachable"
        # -> summary.reachable field. The direct match on "reachable" above
        # would miss because status is "confirmed_reachable", not
        # "reachable". Fix that here (small; keeps the counts loop simple).
        reachable_ct = sum(
            1 for f in confirmed_findings
            if f.taint_confirmation is not None
            and f.taint_confirmation.is_reachable
        )
        summary = dataclasses.replace(summary, reachable=reachable_ct)
        return confirmed_findings, summary

    # ---- helpers ----

    def _all_skipped(
        self,
        findings: List[SecurityFinding],
        *,
        reason: str,
    ) -> Tuple[List[SecurityFinding], ConfirmationSummary]:
        conf = TaintConfirmation(
            status=CONFIRMATION_STATUS_SKIPPED,
            reason=reason,
        )
        out = [self._with_confirmation(f, conf) for f in findings]
        summary = ConfirmationSummary(
            total_findings=len(findings),
            skipped=len(findings),
        )
        return out, summary

    def _with_confirmation(
        self,
        finding: SecurityFinding,
        confirmation: TaintConfirmation,
    ) -> SecurityFinding:
        """Return a new SecurityFinding with taint_confirmation set.

        SecurityFinding is a frozen dataclass so we use dataclasses.replace.
        """
        return dataclasses.replace(finding, taint_confirmation=confirmation)

    def _walk_files(self, repo_path: Path, extensions: Set[str]):
        """Walk the repo yielding files whose suffix is in ``extensions``.

        Uses the same walker as GrepRunner and the taint scan --
        keeping the behavior consistent between the skill discovery
        pass and confirmation. Skip fragments are matched relative to
        ``repo_path``: matched against the absolute path, a repo under
        e.g. ``/build/`` produced an empty call graph and every
        finding came back no_entry_points.
        """
        # Reuse the taint scan service's skip list; kept local to
        # avoid a cross-service import for one constant.
        skip_fragments = (
            "/.git/", "/node_modules/", "/bin/", "/obj/",
            "/dist/", "/build/", "/target/", "/out/",
            "/__pycache__/", "/.venv/", "/venv/",
        )
        return iter_workspace_files(
            repo_path,
            extensions=extensions,
            skip_fragments=skip_fragments,
        )


def _extension_of(file_path: str) -> str:
    """Return the lowercased suffix (with dot) or empty string."""
    idx = file_path.rfind(".")
    if idx == -1:
        return ""
    ext = file_path[idx:].lower()
    # Strip trailing punctuation that a report might have appended.
    return ext


def _language_for_extension(ext: str) -> "str | None":
    ext = ext.lower()
    for lang, exts in _LANGUAGE_EXTENSIONS.items():
        if ext in exts:
            return lang
    return None
