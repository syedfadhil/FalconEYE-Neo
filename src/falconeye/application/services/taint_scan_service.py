"""
Taint scan orchestrator.

Ties the taint analysis pipeline together end-to-end: walks a repo,
builds a call graph, finds candidate source-to-sink paths, verifies
each with the LLM, and emits confirmed paths as
:class:`SecurityFinding` for downstream reporter consumption.

Runs entirely in the application layer, above the domain services
that do the actual work. This layering means the CLI and any other
entry points (SDKs, HTTP APIs, ...) share the same orchestration.

The service is deliberately narrow in scope for this first cut:
  * C# only (only shipped adapter and specs; other languages plug in
    trivially via the adapter registry).
  * Per-file precision for finding locations (points at the sink
    caller method; per-call-site precision is a follow-up refinement).
  * Sequential LLM verification (each candidate is verified one at a
    time; batching / parallel verification is a follow-up).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Set

from ...domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    SecurityReview,
    Severity,
)
from ...domain.models.taint import TaintPath
from ...domain.services.call_graph import CallGraph, CallGraphBuilder
from ...domain.services.taint_path_finder import TaintPathFinder
from ...domain.services.taint_verifier import (
    TaintVerifier,
    VERIFIER_VERDICT_CONFIRMED,
    VERIFIER_VERDICT_FALSE_POSITIVE,
    VERIFIER_VERDICT_UNCERTAIN,
    VerifierResult,
)

if TYPE_CHECKING:
    from ...infrastructure.taint_adapters.adapter_registry import AdapterRegistry
    from ...infrastructure.taint_index import TaintIndex


# Extensions per language for file walking. Kept in step with
# EnhancedASTAnalyzer.LANGUAGE_MAP but narrowed to languages that
# have taint adapters shipped. As new adapters land, add extensions
# here.
_LANGUAGE_EXTENSIONS: Mapping[str, Set[str]] = {
    "csharp": {".cs", ".csx"},
}


# Files or path fragments to skip regardless of language. Kept short
# to avoid surprising the operator; the FileDiscoveryConfig covers
# the broader set for other pipelines.
_DEFAULT_SKIP_PATTERNS = (
    "/.git/",
    "/node_modules/",
    "/bin/",
    "/obj/",
    "/venv/",
    "/.venv/",
    "/dist/",
    "/build/",
    "/target/",
)


@dataclass(frozen=True)
class TaintScanResult:
    """Bundle of everything the scan produced.

    ``review`` carries confirmed and uncertain findings for downstream
    reporters (which already know how to render SecurityReview).
    ``graph``, ``candidate_paths``, and ``verdicts`` are exposed for
    diagnostic use -- CLI --verbose mode can show how many candidates
    were generated and how many the LLM confirmed.
    """
    review: SecurityReview
    graph: CallGraph
    candidate_paths: List[TaintPath]
    verdicts: Mapping[TaintPath, VerifierResult]


class TaintScanService:
    """End-to-end orchestrator for taint analysis.

    Instantiated once at DI setup. Callers invoke :meth:`scan` with
    a repo path and per-language taint config; the service does the
    rest.
    """

    def __init__(
        self,
        *,
        adapter_registry: "AdapterRegistry",
        taint_index: "TaintIndex",
        verifier: TaintVerifier,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._adapter_registry = adapter_registry
        self._taint_index = taint_index
        self._verifier = verifier
        self._logger = logger or logging.getLogger(__name__)

    async def scan(
        self,
        *,
        repo_path: Path,
        languages: Iterable[str],
        max_hops: int = 5,
        verify: bool = True,
    ) -> TaintScanResult:
        """Run the pipeline against ``repo_path``.

        Args:
          repo_path: Root of the repository to analyze.
          languages: Languages to include. Files whose extension
            doesn't match any language's extension set are skipped.
          max_hops: Path finder budget (caller-to-caller edges).
          verify: When False, skip LLM verification and emit every
            candidate path as a finding tagged ``taint:unverified``.
            Useful for debugging / seeing raw candidate output.
        """
        review = SecurityReview.create(
            codebase_path=str(repo_path),
            language=",".join(sorted(languages)),
        )

        # ---------- Build the call graph ----------
        builder = CallGraphBuilder(
            adapter_registry=self._adapter_registry,
            taint_index=self._taint_index,
        )

        extensions: Set[str] = set()
        for lang in languages:
            extensions |= _LANGUAGE_EXTENSIONS.get(lang, set())

        files_seen = 0
        for path in self._walk_files(repo_path, extensions):
            files_seen += 1
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                self._logger.warning(
                    "Could not read %s: %s", path, e,
                )
                continue
            language = self._language_for_extension(path.suffix)
            if language is None:
                continue
            builder.add_file(
                source=source,
                file_path=str(path),
                language=language,
            )

        review.files_analyzed = files_seen
        graph = builder.build()
        self._logger.info(
            "Call graph built: %d nodes, %d edges from %d files",
            len(graph), graph.edge_count(), files_seen,
        )

        # ---------- Find candidate paths ----------
        finder = TaintPathFinder(
            graph=graph,
            taint_index=self._taint_index,
            max_hops=max_hops,
        )
        candidates = finder.find_paths()
        self._logger.info("Candidate paths found: %d", len(candidates))

        # ---------- Verify each candidate ----------
        verdicts: dict = {}
        if not verify:
            # Unverified mode: convert every candidate to a finding
            # with a tag so the reporter shows it as unconfirmed.
            for path in candidates:
                finding = self._path_to_finding(
                    path,
                    graph,
                    verdict=None,
                    extra_tags=("taint:unverified",),
                )
                if finding is not None:
                    review.add_finding(finding)
                    verdicts[path] = VerifierResult(
                        verdict=VERIFIER_VERDICT_UNCERTAIN,
                        confidence=FindingConfidence.LOW,
                        reasoning="Verification skipped by --no-verify.",
                    )
        else:
            for path in candidates:
                verdict = await self._verifier.verify(
                    path=path,
                    graph=graph,
                    taint_index=self._taint_index,
                )
                verdicts[path] = verdict
                if verdict.verdict == VERIFIER_VERDICT_FALSE_POSITIVE:
                    # Drop -- LLM was confident it's not a real flow.
                    continue
                finding = self._path_to_finding(
                    path,
                    graph,
                    verdict=verdict,
                )
                if finding is not None:
                    review.add_finding(finding)

        review.complete()
        return TaintScanResult(
            review=review,
            graph=graph,
            candidate_paths=candidates,
            verdicts=verdicts,
        )

    # ------------------------------------------------------------------
    # File walking
    # ------------------------------------------------------------------

    def _walk_files(self, root: Path, extensions: Set[str]) -> Iterable[Path]:
        """Yield files under ``root`` with an extension in ``extensions``.

        Applies simple default exclusions (VCS, package managers,
        build outputs). Doesn't follow symlinks to avoid infinite
        loops on pathological repos.
        """
        if not root.exists():
            return
        if not extensions:
            return
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in extensions:
                continue
            posix = candidate.as_posix()
            skip = False
            for pat in _DEFAULT_SKIP_PATTERNS:
                if pat in posix:
                    skip = True
                    break
            if skip:
                continue
            yield candidate

    def _language_for_extension(self, ext: str) -> Optional[str]:
        e = ext.lower()
        for lang, exts in _LANGUAGE_EXTENSIONS.items():
            if e in exts:
                return lang
        return None

    # ------------------------------------------------------------------
    # Finding emission
    # ------------------------------------------------------------------

    def _path_to_finding(
        self,
        path: TaintPath,
        graph: CallGraph,
        *,
        verdict: Optional[VerifierResult],
        extra_tags: tuple = (),
    ) -> Optional[SecurityFinding]:
        """Convert a verified TaintPath into a SecurityFinding.

        The finding is located at the sink CALLER method: file and
        line range come from that method's FunctionDef. This is
        method-level precision -- per-call-site precision (pointing
        at the exact ``new SqlCommand(...)`` line) is a follow-up.
        """
        sink_spec = self._taint_index.get_sink(path.sink)
        source_spec = self._taint_index.get_source(path.source)
        if sink_spec is None or source_spec is None:
            # Should not happen -- the path finder only produces paths
            # whose source and sink are index entries. Defensive skip.
            return None

        sink_caller_fn = graph.function(path.sink_caller)
        if sink_caller_fn is None:
            # Same defensive skip: sink_caller comes from a call in
            # user code, so its FunctionDef should exist. Skip cleanly
            # if we somehow got here.
            return None

        # Severity comes from the sink's spec; the LLM's confidence
        # (or a default when not verified) becomes the finding's
        # confidence.
        severity = self._severity_from_spec(sink_spec.severity)
        confidence = (
            verdict.confidence
            if verdict is not None
            else FindingConfidence.LOW
        )

        code_snippet = self._read_method_snippet(sink_caller_fn)

        tags = [
            "taint",
            f"taint:kind:{sink_spec.kind}",
            f"taint:source:{source_spec.kind}",
            f"taint:hops:{path.hop_count}",
        ]
        if verdict is not None and verdict.call_failed:
            tags.append("verifier:call_failed")
        if verdict is not None and verdict.verdict == VERIFIER_VERDICT_UNCERTAIN:
            tags.append("verifier:uncertain")
        tags.extend(extra_tags)

        issue = (
            f"Tainted data from {source_spec.kind} may reach "
            f"{sink_spec.kind} sink in {self._short_name(path.sink_caller)}"
        )
        reasoning_parts = [
            f"Candidate taint flow:",
            f"  Source: {path.source} ({source_spec.kind})",
            f"    called by {path.source_caller}",
            f"  Sink:   {path.sink} ({sink_spec.kind}, {sink_spec.cwe})",
            f"    called by {path.sink_caller}",
            f"  Chain:  {' -> '.join(path.via)}  (hops={path.hop_count})",
        ]
        if verdict is not None:
            reasoning_parts.extend([
                "",
                f"LLM verifier verdict: {verdict.verdict}",
                f"Confidence: {verdict.confidence.value}",
                "",
                "Verifier reasoning:",
                verdict.reasoning,
            ])
        reasoning = "\n".join(reasoning_parts)

        mitigation = self._mitigation_for_kind(sink_spec.kind)

        return SecurityFinding.create(
            issue=issue,
            reasoning=reasoning,
            mitigation=mitigation,
            severity=severity,
            confidence=confidence,
            file_path=sink_caller_fn.location.file_path,
            code_snippet=code_snippet,
            line_start=sink_caller_fn.location.line_start,
            line_end=sink_caller_fn.location.line_end,
            cwe_id=sink_spec.cwe,
            tags=tags,
            evidence_lines=(
                list(verdict.evidence_lines) if verdict is not None else []
            ),
            verifier_verdict=(verdict.verdict if verdict is not None else None),
            verifier_reasoning=(verdict.reasoning if verdict is not None else ""),
        )

    def _read_method_snippet(self, fn) -> str:
        """Extract the method body as a numbered code snippet.

        Used for the SecurityFinding.code_snippet field. Matches the
        line-numbering format used by the existing single-file
        analyzer so reporters render it consistently.
        """
        try:
            content = Path(fn.location.file_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return f"(method body not readable at {fn.location.file_path})"
        lines = content.splitlines()
        start = max(fn.location.line_start - 1, 0)
        end = min(fn.location.line_end, len(lines))
        if start >= end:
            return "(empty method body)"
        width = len(str(end))
        numbered = []
        for i, line in enumerate(lines[start:end], start=fn.location.line_start):
            numbered.append(f"{i:>{width}} | {line}")
        return "\n".join(numbered)

    @staticmethod
    def _severity_from_spec(spec_severity: str) -> Severity:
        """Coerce a spec-level severity string to the Severity enum."""
        try:
            return Severity(spec_severity.lower())
        except (ValueError, AttributeError):
            return Severity.MEDIUM

    @staticmethod
    def _short_name(canonical: str) -> str:
        """Trim the language prefix for prose. csharp:X.Y.Z -> X.Y.Z."""
        if ":" in canonical:
            return canonical.split(":", 1)[1]
        return canonical

    @staticmethod
    def _mitigation_for_kind(kind: str) -> str:
        """Generic mitigation guidance by sink kind.

        Kept as a small hard-coded map because the taint specs don't
        currently carry mitigation text. A future taint_specs schema
        extension could move this into the YAML.
        """
        return _MITIGATIONS.get(kind, _MITIGATIONS["_default"])


_MITIGATIONS = {
    "sql_query": (
        "Use parameterized queries (SqlParameter, DbParameter, or Dapper's "
        "named parameters). Never concatenate user input directly into "
        "SQL strings. If dynamic SQL is truly required, use an approved "
        "escaping library, never manual string formatting."
    ),
    "cmd_exec": (
        "Never pass user input to the shell. Use argument arrays "
        "(ProcessStartInfo with UseShellExecute=false and Arguments as "
        "a list). Validate against an allow-list of known-safe commands."
    ),
    "path_traversal": (
        "Validate that the resolved path is under an expected base "
        "directory. Use Path.GetFullPath and check the result starts "
        "with the expected root. Reject inputs containing '..' or null "
        "bytes."
    ),
    "xxe": (
        "Set XmlReaderSettings.DtdProcessing = DtdProcessing.Prohibit "
        "and XmlResolver = null. Never enable external entity resolution "
        "on untrusted input."
    ),
    "deserialization": (
        "Never deserialize untrusted data with type-aware formatters "
        "(BinaryFormatter, LosFormatter, NetDataContractSerializer). "
        "Use System.Text.Json or Newtonsoft.Json with strict type "
        "handling, and validate the schema of incoming JSON."
    ),
    "ssrf": (
        "Validate URLs against an allow-list of hosts. Never fetch a "
        "URL constructed from user input without host validation. Use "
        "a dedicated HTTP client that respects proxy and DNS pinning."
    ),
    "code_injection": (
        "Never compile or evaluate user input as code. Use configuration "
        "or template engines with strict sandboxing instead of runtime "
        "code generation."
    ),
    "_default": (
        "Do not pass untrusted data to this sink without validation. "
        "Refer to the CWE description for the specific attack class."
    ),
}
