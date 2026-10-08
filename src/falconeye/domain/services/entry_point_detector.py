"""
Entry-point detection: identify the roots of external reachability.

An :class:`EntryPointDetector` scans a workspace's function definitions
plus source files and returns the methods that external input can
reach directly. Reachability from these roots is what the
:class:`ReachabilityConfirmer` checks.

Detectors are language- and framework-specific. This module ships a
detector for ASP.NET Core (C#); Python/Java/JavaScript detectors
follow in later patches.

Detectors are deliberately narrow: they favor precision over recall.
A false negative in detection (missing a real entry point) means we
mark reachable findings as "not_confirmed", which is safe -- the
finding still ships at its original confidence. A false positive
(marking a helper as an entry point) inflates confidence
inappropriately. Prefer detectors that are conservative.
"""

from __future__ import annotations

import logging
import re
from typing import List, Mapping, Protocol, Sequence

from ..models.confirmation import (
    ENTRY_POINT_KIND_HTTP,
    ENTRY_POINT_KIND_MIDDLEWARE,
    EntryPoint,
)
from ..models.taint import FunctionDef


_logger = logging.getLogger(__name__)


class EntryPointDetector(Protocol):
    """Interface for language-specific entry-point detection.

    Implementations receive the full list of function definitions in
    the workspace plus a map from file path to source text (needed for
    attribute detection since attributes live above the function's
    starting line and aren't captured in FunctionDef itself).
    """

    def language(self) -> str:
        """The language identifier this detector handles (e.g. 'csharp')."""
        ...

    def detect(
        self,
        function_defs: Sequence[FunctionDef],
        source_by_file: Mapping[str, str],
    ) -> List[EntryPoint]:
        """Return every entry point in the given functions.

        MUST return canonical_name values that match the FunctionDef
        objects passed in -- the reachability confirmer does an exact
        string lookup against the call graph.
        """
        ...


# --------------------------------------------------------------------
# ASP.NET Core detector
# --------------------------------------------------------------------


class AspNetCoreEntryPointDetector:
    """Detects ASP.NET Core entry points in C# source.

    Currently recognizes:

    * Methods annotated with ``[HttpGet]``, ``[HttpPost]``, ``[HttpPut]``,
      ``[HttpDelete]``, ``[HttpPatch]``, ``[HttpHead]``, or
      ``[HttpOptions]`` -- with or without route arguments.
    * Methods annotated with ``[Route(...)]``, which is used for
      convention-based routing without a verb-specific attribute.
    * ``Invoke`` / ``InvokeAsync`` methods on classes whose name ends
      in ``Middleware`` -- the ASP.NET Core middleware convention.

    Not yet recognized (deferred to follow-up patches):

    * Minimal API endpoints (``app.MapGet("/", handler)``). The
      handler is often an inline lambda or method reference; matching
      it back to a FunctionDef requires resolution work that isn't in
      the current C# adapter.
    * SignalR hubs (methods on classes deriving from ``Hub``).
    * gRPC service methods (methods on classes deriving from
      ``ServiceBase`` in generated gRPC code).
    * Controller inheritance convention (any public method on a class
      inheriting from ``Controller`` / ``ControllerBase`` without an
      explicit attribute). Included the attribute-based cases because
      those are unambiguous; unattributed methods would need class
      inheritance tracking that we don't have yet.

    Precision vs recall: this detector prioritizes precision. Missing
    a real entry point is safer than falsely claiming one. See module
    docstring.
    """

    _HTTP_VERB_RE = re.compile(
        r"\[\s*Http(Get|Post|Put|Delete|Patch|Head|Options)\b",
        re.IGNORECASE,
    )
    _ROUTE_RE = re.compile(
        r"\[\s*Route\s*\(",
        re.IGNORECASE,
    )
    #: Middleware method names per ASP.NET Core convention. The class
    #: heuristic (name ending in "Middleware") narrows this so a
    #: random Invoke() method somewhere else doesn't get flagged.
    _MIDDLEWARE_METHOD_NAMES = frozenset({"Invoke", "InvokeAsync"})

    #: How many source lines above a method's start we scan for
    #: attributes. Six lines is generous -- multi-line ``[Route]``
    #: with parameters plus a ``[HttpGet]`` on top would take 3-4
    #: lines; we allow blank lines between the attributes and the
    #: method signature.
    _ATTRIBUTE_LOOKBACK_LINES = 6

    def language(self) -> str:
        return "csharp"

    def detect(
        self,
        function_defs: Sequence[FunctionDef],
        source_by_file: Mapping[str, str],
    ) -> List[EntryPoint]:
        entries: List[EntryPoint] = []
        for func in function_defs:
            if func.language != "csharp":
                continue
            source = source_by_file.get(func.location.file_path)
            if source is None:
                # Source not provided -- can't check attributes.
                # Still check the middleware naming heuristic.
                entry = self._middleware_from_name(func)
                if entry is not None:
                    entries.append(entry)
                continue

            attr_text = self._attribute_lookback(source, func.location.line_start)
            entry = self._attribute_entry(func, attr_text)
            if entry is None:
                entry = self._middleware_from_name(func)
            if entry is not None:
                entries.append(entry)
        _logger.debug(
            "AspNetCore detector found %d entry points in %d functions",
            len(entries), len(function_defs),
        )
        return entries

    # ---- helpers ----

    def _attribute_lookback(self, source: str, line_start: int) -> str:
        """Return the text where a method's attributes might appear.

        Tree-sitter's C# grammar treats ``[HttpGet]`` above a method
        as part of the method's own syntactic range: FunctionDef.
        Location.line_start can point at the attribute line, not the
        ``public string GetUser(...)`` signature line. So a naive
        "search N lines above line_start" strategy either misses the
        attribute (when line_start IS the attribute line) or captures
        the previous method's attribute (when methods sit close).

        Strategy handles both:

        * ``start_idx`` walks backward from line_start until either
          the lookback budget runs out OR we hit a line that's just a
          closing ``}``. The closing brace is a reliable marker for
          "previous method ended here"; anything above it can't be
          this method's attribute.

        * ``end_idx`` walks forward from line_start until we hit the
          method's own opening ``{``. Attribute lines (starting with
          ``[``) are SKIPPED during this search because a route
          template like ``[HttpGet("api/users/{id}")]`` contains a
          ``{`` inside its string that isn't the method's brace.

        Line numbers are 1-based (Location semantics). Result is a
        newline-joined slice suitable for regex search.
        """
        lines = source.splitlines()
        if not lines:
            return ""
        method_idx = max(0, min(line_start - 1, len(lines) - 1))  # 0-based, bounded
        start_idx = max(0, method_idx - self._ATTRIBUTE_LOOKBACK_LINES)

        # Bound start_idx: don't look past the previous method's
        # closing brace. Walk backward from method_idx-1 down to the
        # existing start_idx floor. Match on lines that END with `}`
        # so inline single-line methods (``void First() { }``) also
        # act as a boundary, not just standalone-brace lines.
        for i in range(method_idx - 1, start_idx - 1, -1):
            stripped = lines[i].strip()
            if stripped.endswith("}"):
                start_idx = i + 1
                break

        # Find the method's opening brace forward. Skip attribute
        # lines (leading ``[``) because their {"..."} string content
        # would falsely satisfy the check.
        end_idx = min(len(lines), method_idx + 6)
        for i in range(method_idx, end_idx):
            stripped = lines[i].lstrip()
            if stripped.startswith("["):
                continue
            if "{" in lines[i]:
                end_idx = i
                break
        return "\n".join(lines[start_idx:end_idx])

    def _attribute_entry(
        self,
        func: FunctionDef,
        attr_text: str,
    ) -> "EntryPoint | None":
        http_match = self._HTTP_VERB_RE.search(attr_text)
        if http_match is not None:
            return EntryPoint(
                canonical_name=func.canonical_name,
                reason=f"[Http{http_match.group(1)}]",
                location=func.location,
                kind=ENTRY_POINT_KIND_HTTP,
            )
        if self._ROUTE_RE.search(attr_text) is not None:
            return EntryPoint(
                canonical_name=func.canonical_name,
                reason="[Route]",
                location=func.location,
                kind=ENTRY_POINT_KIND_HTTP,
            )
        return None

    def _middleware_from_name(self, func: FunctionDef) -> "EntryPoint | None":
        """Match the ASP.NET Core middleware convention.

        The class name (segment before the method name in the
        canonical name) must end in ``Middleware`` and the method
        name must be ``Invoke`` or ``InvokeAsync``. This is loose but
        avoids false positives on ``FooBar.Invoke`` methods that
        aren't ASP.NET Core middleware.
        """
        method_name = func.canonical_name.rsplit(".", 1)[-1]
        if method_name not in self._MIDDLEWARE_METHOD_NAMES:
            return None
        # Recover the class portion: everything between the language
        # prefix and the method name. Canonical names typically look
        # like ``csharp:MyApp.Middleware.AuthMiddleware.InvokeAsync``.
        parent = func.canonical_name.rsplit(".", 1)[0]
        class_segment = parent.rsplit(".", 1)[-1] if "." in parent else parent
        if not class_segment.endswith("Middleware"):
            return None
        return EntryPoint(
            canonical_name=func.canonical_name,
            reason=f"Middleware.{method_name}",
            location=func.location,
            kind=ENTRY_POINT_KIND_MIDDLEWARE,
        )


# --------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------


class EntryPointDetectorRegistry:
    """Look up detectors by language identifier.

    Analogous to :class:`AdapterRegistry` for taint adapters. The
    orchestration service holds one of these; adding a new language
    is a matter of registering the detector.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, EntryPointDetector] = {}

    def register(self, detector: EntryPointDetector) -> None:
        self._detectors[detector.language()] = detector

    def get(self, language: str) -> "EntryPointDetector | None":
        return self._detectors.get(language)

    def languages(self) -> List[str]:
        return sorted(self._detectors)

    def detect_all(
        self,
        function_defs: Sequence[FunctionDef],
        source_by_file: Mapping[str, str],
    ) -> List[EntryPoint]:
        """Run every registered detector against the given functions.

        Detectors are language-scoped internally; each filters to its
        own language and ignores the rest.
        """
        all_entries: List[EntryPoint] = []
        for detector in self._detectors.values():
            all_entries.extend(detector.detect(function_defs, source_by_file))
        return all_entries
