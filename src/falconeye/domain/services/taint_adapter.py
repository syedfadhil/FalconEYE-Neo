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
from pathlib import Path
from typing import List

from ..models.taint import FunctionDef, UnresolvedCall


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
