"""
Language adapter interface for taint analysis.

Each supported language provides a concrete implementation of
``TaintLanguageAdapter`` that walks source in that language and
produces the domain-level primitives (``FunctionDef``,
``UnresolvedCall``) that downstream taint analysis consumes.

Design goals:

- **Small surface.** Adding a language should be a bounded amount of
  work. This interface is deliberately narrow so that even a
  half-implemented adapter (function extraction only, no calls) is a
  legitimate contributor.
- **No graph knowledge.** Adapters do not build the call graph, do
  not know about ``TaintIndex``, and do not run LLM prompts. They
  emit facts about the source; the graph builder consumes them.
- **Deterministic.** Two runs on identical source produce identical
  output. No wall-clock, no dependency on order-sensitive imports.

See ``taint_specs/SCHEMA.md`` §12 for the checklist a contributor
follows when adding a new language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from typing import List

from ..models.taint import (
    FileResolutionContext,
    FunctionDef,
    UnresolvedCall,
)


if TYPE_CHECKING:
    # Only used in type hints. Avoids an import cycle:
    # infrastructure.taint_index imports the adapter interface
    # via the domain package, so importing TaintIndex here would
    # create a runtime cycle.
    from ...infrastructure.taint_index import TaintIndex


class TaintLanguageAdapter(ABC):
    """Abstract adapter for one language's taint-relevant source facts.

    Concrete adapters own the tree-sitter (or other parser) integration
    and the language-specific naming conventions. Instances are
    stateless with respect to input source -- extraction methods are
    pure functions of their arguments.
    """

    #: Canonical language tag, matching the ``language:`` field in
    #: taint spec YAML files (``csharp``, ``python``, ``java``, ...).
    language: str = ""

    @abstractmethod
    def extract_functions(
        self,
        source: str,
        file_path: str,
    ) -> List[FunctionDef]:
        """Every function/method definition in ``source``.

        Includes: named methods, constructors, property accessors
        (as separate get_X / set_X entries), local functions, and
        any language-specific callable form (C# local functions,
        Python nested defs).

        Excludes: lambda expressions and other anonymous callables
        that cannot be given a stable canonical name.

        Order of returned functions is source order (top-to-bottom,
        outer-to-inner) so downstream deterministic ordering falls
        out for free.

        Args:
            source: Full source text of one file.
            file_path: Path used for the ``Location.file_path`` field.
                Not read from disk; passed through verbatim.
        """

    def extract_calls(
        self,
        source: str,
        file_path: str,
    ) -> List[UnresolvedCall]:
        """Every call site inside ``source``, with an unresolved callee.

        Includes three syntactic forms, each distinguished by flags
        on the returned :class:`UnresolvedCall`:

          * ordinary invocations (``foo()``, ``obj.Bar()``,
            ``Class.Static()``, ``Ns.Class.Static()``)
          * constructor calls (``new TypeName(...)``) --
            :attr:`~UnresolvedCall.is_constructor` is ``True``
          * property setter assignments (``obj.Name = value``) --
            :attr:`~UnresolvedCall.is_property_set` is ``True``

        Every emitted call has a non-empty ``in_function`` field:
        calls that appear outside any function body (top-level
        statements, module initializers) are dropped because they
        cannot participate in a call-graph edge.

        The default implementation returns ``[]``. This lets an
        early adapter (function extraction only) be a legitimate
        contributor -- downstream code that queries calls just
        finds none for that language until the adapter is extended.

        Args:
            source: Full source text of one file.
            file_path: Path passed through to ``Location.file_path``.
        """
        return []

    def build_resolution_context(
        self,
        source: str,
        file_path: str,
    ) -> FileResolutionContext:
        """Extract everything the resolver will need for this file.

        Collects:

          * the enclosing namespace (from block-scoped or file-scoped
            namespace declarations)
          * ordinary and static ``using`` statements
          * per-method local variable types (parameters, explicit
            declarations, and ``var x = new T(...)`` inference)
          * per-class field types

        The default implementation returns an empty context, which
        keeps the interface additive: a partial adapter without
        resolution support still parses correctly, its calls just
        won't resolve to anything.

        Called once per file at scan start. The resulting context
        is immutable and reusable across every ``resolve_call``
        invocation for that file.
        """
        return FileResolutionContext(file_path=file_path)

    def resolve_call(
        self,
        call: UnresolvedCall,
        context: FileResolutionContext,
        taint_index: "TaintIndex",
    ) -> Optional[str]:
        """Turn an :class:`UnresolvedCall` into a canonical name.

        Return values:

          * A canonical name matching an entry in ``taint_index``
            (source, sink, sanitizer, or propagator). Preferred:
            this is what taint findings ultimately need.
          * A best-guess canonical name for user-defined code that
            no spec covers. The graph builder can still use this
            to add edges between application methods.
          * ``None`` when the call cannot be resolved at all -- 
            typically because the receiver's type isn't derivable
            from the local context (chained calls, complex
            expressions). The call is dropped from the graph.

        The default implementation returns ``None`` so a partial
        adapter still behaves sensibly.
        """
        return None

    @abstractmethod
    def detect_frameworks(self, project_root: Path) -> List[str]:
        """Names of framework-scoped spec files that apply to this project.

        Called once at scan start. Returns a list of spec identifiers
        (without ``.yaml``) that the taint index loader will treat as
        auto-hinted. For example, a C# adapter finding an AspNetCore
        package reference returns ``['csharp_aspnet']``.

        Always returns the language-generic spec name implicitly --
        callers do not need it in the result. Returning an empty list
        is valid (no frameworks detected).

        Behaviour when the language isn't present in ``project_root``
        should be a fast no-op returning ``[]``, not an error.
        """
