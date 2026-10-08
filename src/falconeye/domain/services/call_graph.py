"""
Call graph domain service.

An in-memory directed graph whose nodes are function canonical names
(both user-defined via ``FunctionDef`` and spec-referenced via
:class:`TaintIndex` entries) and whose edges represent call
relationships: ``M -> N`` means M's body calls N.

The graph is deliberately CALL-SHAPED, not data-flow shaped. It says
nothing about which parameters are tainted or how return values
propagate; that's the LLM verifier's job to determine from the actual
source. The graph's role is candidate generation: identifying method
pairs where a taint flow COULD exist based on the call structure.

Two design invariants:

1. Nodes are strings (canonical names). No metadata attached; downstream
   consumers combine the graph with the taint index / function map to
   classify nodes.

2. The graph is append-only during construction and read-only afterward.
   :class:`CallGraphBuilder` gathers edges as files are processed;
   :meth:`CallGraphBuilder.build` returns an immutable :class:`CallGraph`.

Persistence (SQLite / on-disk indexing) is not implemented here; the
graph lives entirely in memory. For typical C# codebases this is a
non-issue -- a large repo's call graph fits comfortably in tens of
megabytes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from ..models.taint import FunctionDef, UnresolvedCall

if TYPE_CHECKING:
    from ...infrastructure.taint_index import TaintIndex
    from .taint_adapter import TaintLanguageAdapter


class CallGraph:
    """Immutable directed graph of caller -> callee canonical names.

    Constructed by :class:`CallGraphBuilder`; not intended to be
    instantiated directly by downstream code.
    """

    def __init__(
        self,
        *,
        successors: Mapping[str, FrozenSet[str]],
        predecessors: Mapping[str, FrozenSet[str]],
        functions: Mapping[str, FunctionDef],
    ) -> None:
        # Defensive copy plus freezing so the returned graph really is
        # immutable from the outside.
        self._successors: Dict[str, FrozenSet[str]] = dict(successors)
        self._predecessors: Dict[str, FrozenSet[str]] = dict(predecessors)
        self._functions: Dict[str, FunctionDef] = dict(functions)

    # -- Structural queries -------------------------------------------

    def nodes(self) -> Iterator[str]:
        """Every node in the graph (union of edge endpoints + function
        definitions), yielded in canonical-name lexicographic order for
        deterministic iteration."""
        seen: Set[str] = set()
        seen.update(self._successors.keys())
        seen.update(self._predecessors.keys())
        seen.update(self._functions.keys())
        for n in sorted(seen):
            yield n

    def successors(self, node: str) -> FrozenSet[str]:
        """Nodes ``node`` calls. Empty frozenset for unknown / leaf nodes."""
        return self._successors.get(node, frozenset())

    def predecessors(self, node: str) -> FrozenSet[str]:
        """Nodes that call ``node``. Empty frozenset for unknown / root nodes."""
        return self._predecessors.get(node, frozenset())

    def __contains__(self, node: str) -> bool:
        return (
            node in self._successors
            or node in self._predecessors
            or node in self._functions
        )

    # -- Function-definition lookup -----------------------------------

    def function(self, canonical_name: str) -> Optional[FunctionDef]:
        """The :class:`FunctionDef` for a user-defined node, or None.

        Nodes representing spec-only entries (sources/sinks whose
        implementations live outside the analyzed code) return None
        because there's no source-level definition to point at.
        """
        return self._functions.get(canonical_name)

    def user_defined_nodes(self) -> Iterator[str]:
        """Only nodes with a :class:`FunctionDef` -- i.e. functions
        actually present in the analyzed source. Excludes spec-only
        endpoints like ``csharp:System.Console.ReadLine``."""
        for n in sorted(self._functions):
            yield n

    # -- Diagnostics --------------------------------------------------

    def __len__(self) -> int:
        """Total number of distinct nodes in the graph."""
        return sum(1 for _ in self.nodes())

    def edge_count(self) -> int:
        return sum(len(s) for s in self._successors.values())


class CallGraphBuilder:
    """Accumulate function definitions and calls; produce a CallGraph.

    Usage::

        builder = CallGraphBuilder(adapter_registry=reg, taint_index=idx)
        for path, source in files_to_scan:
            language = detect_language(path)
            builder.add_file(source=source, file_path=path, language=language)
        graph = builder.build()

    The builder holds an ``AdapterRegistry`` reference so it can
    dispatch per-language parsing. The taint index is threaded through
    to :meth:`TaintLanguageAdapter.resolve_call` so the resolver can
    prefer candidate names matching known spec entries.

    Files whose language has no registered adapter are silently
    skipped -- the builder is meant to work with a subset of the
    codebase (only opted-in languages).
    """

    def __init__(
        self,
        *,
        adapter_registry,
        taint_index: "TaintIndex",
    ) -> None:
        self._adapter_registry = adapter_registry
        self._taint_index = taint_index
        self._successors: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._functions: Dict[str, FunctionDef] = {}

    def add_file(
        self,
        *,
        source: str,
        file_path: str,
        language: str,
    ) -> None:
        """Process one source file into the accumulating graph.

        Steps:
          1. Look up the adapter for ``language``. Skip if not present.
          2. Extract functions -> register nodes with FunctionDef metadata.
          3. Build the resolution context for this file.
          4. Extract calls, resolve each, and add caller -> resolved edges.

        Unresolvable calls are dropped (no edge added). Calls that
        resolve to a canonical name are added as edges even if the
        target doesn't have a :class:`FunctionDef` -- these represent
        calls into spec-referenced code (stdlib, framework, etc.).
        """
        adapter = self._adapter_registry.get(language)
        if adapter is None:
            return

        for fn in adapter.extract_functions(source, file_path):
            self._functions[fn.canonical_name] = fn

        context = adapter.build_resolution_context(source, file_path)
        for call in adapter.extract_calls(source, file_path):
            resolved = adapter.resolve_call(call, context, self._taint_index)
            if resolved is None:
                continue
            self._add_edge(call.in_function, resolved)

    def _add_edge(self, from_node: str, to_node: str) -> None:
        # Self-loops are legitimate (recursive functions) but they don't
        # help path finding and can produce degenerate BFS behaviour.
        # Drop them at construction so downstream code doesn't need to
        # special-case them.
        if from_node == to_node:
            return
        self._successors[from_node].add(to_node)
        self._predecessors[to_node].add(from_node)

    def build(self) -> CallGraph:
        """Freeze the accumulated state into an immutable CallGraph."""
        successors: Dict[str, FrozenSet[str]] = {
            k: frozenset(v) for k, v in self._successors.items()
        }
        predecessors: Dict[str, FrozenSet[str]] = {
            k: frozenset(v) for k, v in self._predecessors.items()
        }
        return CallGraph(
            successors=successors,
            predecessors=predecessors,
            functions=self._functions,
        )
