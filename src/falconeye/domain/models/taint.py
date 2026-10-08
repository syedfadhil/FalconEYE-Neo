"""
Domain models for taint analysis.

These types represent the *contract* between language adapters (which
walk source code) and downstream consumers (call-graph builder, path
finder, LLM verifier). They deliberately have no knowledge of tree-
sitter, YAML specs, or any specific language -- they are pure data.

Canonical name format follows ``taint_specs/SCHEMA.md`` §2:
``<lang>:<fully_qualified_symbol>``. Examples:

    csharp:System.Data.SqlClient.SqlCommand.<init>
    csharp:MyApp.Web.UserController.GetById
    csharp:MyApp.Domain.User.get_Email

Constructors use ``<init>``. Property accessors use ``get_X`` / ``set_X``.
See SCHEMA.md §2 for the full conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Location:
    """Position of a syntactic element in a source file.

    Line numbers are 1-based (matching editors and existing
    ``SecurityFinding.line_start`` semantics). Columns are 0-based
    (matching tree-sitter's native indexing).
    """
    file_path: str
    line_start: int
    line_end: int
    col_start: int = 0
    col_end: int = 0


@dataclass(frozen=True)
class FunctionDef:
    """
    A function or method definition extracted from source code.

    ``canonical_name`` is the graph key -- other functions' calls
    resolve into this name space. The adapter is responsible for
    producing a canonical name; the call-graph builder does not
    re-derive it.

    ``parameters`` is a tuple of parameter names in declaration order.
    Type information is best-effort -- some languages provide types
    trivially (Java, C#), others require inference (Python). Callers
    that need type info should use ``return_type`` and add types to
    ``parameters`` only when they can be recovered reliably.
    """
    canonical_name: str
    language: str
    location: Location
    parameters: Tuple[str, ...] = ()
    return_type: Optional[str] = None
    is_constructor: bool = False
    is_static: bool = False
    accessor: Optional[str] = None  # 'get' or 'set' for property accessors


@dataclass(frozen=True)
class UnresolvedCall:
    """
    A call site whose callee has not yet been resolved to a canonical
    name. Produced by an adapter's raw call extraction; consumed by
    the adapter's ``resolve_call`` step in a later patch.

    ``callee_text`` is the raw expression as it appears in source:
    ``foo()``, ``obj.Bar()``, ``this.Baz()``, ``System.IO.File.ReadAllText(...)``.
    The resolver's job is to combine this with import / using / class
    context to produce a canonical name.

    ``in_function`` is the canonical name of the enclosing function,
    so call-graph edges can be constructed once resolution completes.
    """
    callee_text: str
    location: Location
    in_function: str
    positional_arg_count: int = 0

    #: True when the syntactic form is ``new TypeName(...)``. The
    #: resolver treats callee_text as a type name and emits an
    #: edge to ``<lang>:<qualified_type>.<init>``. Distinct from
    #: an ordinary call because there is no receiver to resolve
    #: against.
    is_constructor: bool = False

    #: True when the syntactic form is ``lhs = rhs`` and ``lhs``
    #: is a member access (``obj.Name`` or ``Class.Name``). The
    #: resolver treats this as a property setter call: the target
    #: canonical name is ``<qualified_owner>.set_<Name>`` and the
    #: RHS is the sole positional argument. We over-emit here --
    #: at extraction time we cannot always distinguish a property
    #: assignment from a field assignment. The resolver drops
    #: entries that don't match any known property; taint specs
    #: filter the rest.
    is_property_set: bool = False

@dataclass(frozen=True)
class MethodResolutionContext:
    """Resolution context for a single method or property accessor.

    The variable_types map covers three sources:

      * method parameters, keyed by parameter name
      * local variables with explicit type annotation
        (``SqlCommand cmd = ...``)
      * local variables with inferred type from ``var`` when the
        RHS is a ``new TypeName(...)`` expression

    Class fields and inherited fields are handled at the file
    level (see :class:`FileResolutionContext`), so a method's
    map does not need to include them -- the resolver falls back
    to the file context when a receiver is not a local variable.

    Type names are stored RAW (as written in source): they may or
    may not be namespace-qualified. The resolver expands them via
    the file's using statements when producing a candidate name.
    """
    canonical_name: str
    variable_types: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FileResolutionContext:
    """Everything the resolver needs to know about one source file.

    Produced once per file by ``TaintLanguageAdapter.build_resolution_context``
    and passed to every ``resolve_call`` invocation for calls in
    that file. Cheap to build; the payoff is that name resolution
    itself becomes a series of dictionary lookups.

    Fields:
      file_path       -- passed through for diagnostic purposes
      file_namespace  -- the enclosing namespace as a dotted string,
                         or None for global-scope files. Covers
                         both block-scoped and file-scoped forms.
      usings          -- ordinary ``using Ns.Sub;`` targets
      static_usings   -- ``using static Ns.Type;`` targets, which
                         bring member methods into bare-name scope
      class_fields    -- class-level field name -> raw type,
                         keyed by the enclosing type's canonical
                         name (matches FunctionDef.canonical_name
                         without the trailing member)
      method_contexts -- canonical method name ->
                         MethodResolutionContext for that method
    """
    file_path: str
    file_namespace: Optional[str] = None
    usings: Tuple[str, ...] = ()
    static_usings: Tuple[str, ...] = ()
    class_fields: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    method_contexts: Mapping[str, MethodResolutionContext] = field(default_factory=dict)
