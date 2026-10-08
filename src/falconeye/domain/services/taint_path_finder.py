"""
Taint path finder.

Given a :class:`CallGraph` and a :class:`TaintIndex`, enumerate
candidate source-to-sink paths for downstream LLM verification. The
path finder is a candidate generator, not a decision maker:

  * A "path" here means the call structure MIGHT admit taint flow,
    not that taint actually flows. Confirmation is the LLM verifier's
    job in a later patch.
  * Sanitizer NODES are excluded from path traversal -- BFS doesn't
    walk through them as intermediates. Deeper filtering ("this
    method's callee is a sanitizer, so the method is safe") requires
    parameter-level data flow and belongs to the LLM verifier. The
    path finder over-emits by design; the LLM prunes.
  * Paths are bounded by ``max_hops``, the number of caller-to-caller
    edges between the source's caller and the sink's caller. A hop
    count of 0 means the source and sink are called by the same user
    method (intra-method flow).

Two flow patterns are captured:

  1. **Intra-method flow**: a user method M calls both a source and
     a sink. The path is M itself, hop_count=0.
  2. **Inter-method flow**: user method M1 calls a source, M1 (transitively)
     calls M_k which calls a sink. The path is M1 -> ... -> M_k,
     hop_count = k-1.

Return-value taint (source's output flows back to M1 which then passes
tainted return to M2) is not modelled: this needs data-flow analysis
that lives in a follow-up patch. The current model treats CALL
RELATIONSHIPS as the sole path constraint.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Tuple

from ..models.taint import TaintPath

if TYPE_CHECKING:
    from ...infrastructure.taint_index import TaintIndex
    from .call_graph import CallGraph


class TaintPathFinder:
    """Enumerate candidate source-to-sink paths in a :class:`CallGraph`.

    The finder is stateless with respect to input: :meth:`find_paths`
    is a pure function of (graph, taint_index, max_hops). Instances
    are cheap to construct and reusable across analyses.
    """

    def __init__(
        self,
        *,
        graph: "CallGraph",
        taint_index: "TaintIndex",
        max_hops: int = 5,
    ) -> None:
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")
        self._graph = graph
        self._taint_index = taint_index
        self._max_hops = max_hops

    def find_paths(self) -> List[TaintPath]:
        """Every candidate source-to-sink path in the graph.

        Deduplicated on (source, sink, source_caller, sink_caller):
        different intermediate chains between the same endpoints
        collapse to the shortest one. This keeps LLM verification
        cost bounded when many parallel chains exist.

        Results are returned in a stable order (source canonical name,
        sink canonical name, then chain) so repeated runs on identical
        input produce identical output.
        """
        source_names = set(self._taint_index.sources.keys())
        sink_names = set(self._taint_index.sinks.keys())
        sanitizer_names = set(self._taint_index.sanitizers.keys())

        # For each user method: which sources / sinks does it directly call?
        # A "user method" here is any predecessor of a source or sink -- it
        # need not have a FunctionDef (spec-only nodes can chain through).
        # We look at direct predecessors only because that's what "calls
        # the source" means in graph terms.
        source_callers: Dict[str, Set[str]] = {}   # caller -> {source names}
        sink_callers: Dict[str, Set[str]] = {}     # caller -> {sink names}

        for s in source_names:
            if s not in self._graph:
                continue
            for pred in self._graph.predecessors(s):
                source_callers.setdefault(pred, set()).add(s)
        for k in sink_names:
            if k not in self._graph:
                continue
            for pred in self._graph.predecessors(k):
                sink_callers.setdefault(pred, set()).add(k)

        # Fast path: no source/sink adjacency in the graph -> no paths.
        if not source_callers or not sink_callers:
            return []

        # For deduplication: (source, sink, source_caller, sink_caller)
        # -> shortest via chain seen so far.
        best: Dict[Tuple[str, str, str, str], Tuple[str, ...]] = {}

        for src_caller, srcs in source_callers.items():
            # BFS forward from the source caller through the call graph.
            # `via` accumulates the caller chain (inclusive on both ends).
            # We stop expanding at max_hops or when hitting a sanitizer.
            #
            # deque of (current_node, via_so_far)
            queue = deque([(src_caller, (src_caller,))])
            seen_on_this_bfs: Set[str] = {src_caller}

            while queue:
                current, via = queue.popleft()

                # Any sinks called directly by `current` produce candidates.
                for sink in sink_callers.get(current, ()):
                    for source in srcs:
                        key = (source, sink, src_caller, current)
                        existing = best.get(key)
                        if existing is None or len(via) < len(existing):
                            best[key] = via

                # Extend the BFS: don't exceed max_hops, don't revisit,
                # don't traverse through sanitizers.
                if len(via) - 1 >= self._max_hops:
                    continue
                for successor in self._graph.successors(current):
                    # Sources/sinks are terminal in path enumeration --
                    # we already handle them as adjacent to callers, and
                    # traversing through them would double-count.
                    if successor in source_names or successor in sink_names:
                        continue
                    if successor in sanitizer_names:
                        continue
                    if successor in seen_on_this_bfs:
                        continue
                    seen_on_this_bfs.add(successor)
                    queue.append((successor, via + (successor,)))

        # Materialize TaintPath objects in stable order.
        results: List[TaintPath] = []
        for (source, sink, src_caller, sink_caller), via in sorted(best.items()):
            results.append(TaintPath(
                source=source,
                sink=sink,
                source_caller=src_caller,
                sink_caller=sink_caller,
                via=via,
            ))
        return results
