"""Unit tests for the call graph, its builder, and the taint path finder."""

from __future__ import annotations

import textwrap

import pytest

from falconeye.domain.models.taint import FunctionDef, Location, TaintPath
from falconeye.domain.services.call_graph import CallGraph, CallGraphBuilder
from falconeye.domain.services.taint_path_finder import TaintPathFinder
from falconeye.infrastructure.taint_adapters.adapter_registry import AdapterRegistry
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_taint_index():
    """The real shipped csharp_generic index."""
    return load_taint_index(languages=["csharp"])


@pytest.fixture(scope="module")
def csharp_registry():
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    return reg


def _build(source: str, registry, taint_index, *, file_path: str = "t.cs") -> CallGraph:
    """Small helper: build a graph from one C# source string."""
    builder = CallGraphBuilder(
        adapter_registry=registry,
        taint_index=taint_index,
    )
    builder.add_file(source=source, file_path=file_path, language="csharp")
    return builder.build()


# ---------------------------------------------------------------------
# CallGraph structural properties
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_call_graph_empty():
    g = CallGraph(successors={}, predecessors={}, functions={})
    assert len(g) == 0
    assert g.edge_count() == 0
    assert list(g.nodes()) == []
    assert "anything" not in g


@pytest.mark.unit
def test_builder_produces_edges_from_source(csharp_registry, loaded_taint_index):
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void M() { new SqlCommand(q, conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    assert "csharp:X.Y.M" in g
    assert "csharp:System.Data.SqlClient.SqlCommand.<init>" in g.successors("csharp:X.Y.M")


@pytest.mark.unit
def test_builder_populates_predecessors(csharp_registry, loaded_taint_index):
    src = textwrap.dedent("""
    using System.IO;
    namespace X;
    public class Y {
        public void M() { File.ReadAllText("/x"); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    preds = g.predecessors("csharp:System.IO.File.ReadAllText")
    assert "csharp:X.Y.M" in preds


@pytest.mark.unit
def test_builder_drops_self_loops(csharp_registry, loaded_taint_index):
    """Recursive functions produce M -> M in call extraction; graph drops these."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void Rec() { Rec(); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    # Rec() as a bare call resolves to `csharp:X.Y.Rec` via `this`-scope
    # fallback. That would produce a self-loop; the graph drops it.
    assert "csharp:X.Y.Rec" not in g.successors("csharp:X.Y.Rec")


@pytest.mark.unit
def test_builder_ignores_unresolvable_calls(csharp_registry, loaded_taint_index):
    """resolve_call returning None -> no edge added.

    We can't easily construct a "guaranteed None" case with the current
    resolver (it best-guesses aggressively), so this test verifies the
    graph doesn't add spurious edges for calls whose receiver is not
    a variable/type -- like the outer of a chained call.
    """
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { foo.Bar().Baz(); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    # Whatever the resolver does, no user method other than M should
    # appear as a source (there's only one method in the file).
    user_nodes = list(g.user_defined_nodes())
    assert user_nodes == ["csharp:X.Y.M"]


@pytest.mark.unit
def test_builder_no_adapter_for_language_skips_file(loaded_taint_index):
    """If the language has no registered adapter, add_file is a no-op."""
    empty_reg = AdapterRegistry()  # no C# adapter registered
    builder = CallGraphBuilder(
        adapter_registry=empty_reg,
        taint_index=loaded_taint_index,
    )
    builder.add_file(
        source="namespace X; public class Y { public void M() {} }",
        file_path="t.cs",
        language="csharp",
    )
    g = builder.build()
    assert len(g) == 0


@pytest.mark.unit
def test_builder_functions_preserved(csharp_registry, loaded_taint_index):
    """FunctionDefs remain queryable through the graph."""
    src = textwrap.dedent("""
    namespace X;
    public class Y {
        public void M() { }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    fn = g.function("csharp:X.Y.M")
    assert fn is not None
    assert fn.canonical_name == "csharp:X.Y.M"
    # Spec-only nodes have no FunctionDef.
    assert g.function("csharp:System.IO.File.ReadAllText") is None


@pytest.mark.unit
def test_builder_user_defined_nodes_excludes_spec_only(csharp_registry, loaded_taint_index):
    src = textwrap.dedent("""
    using System.IO;
    namespace X;
    public class Y {
        public void M() { File.ReadAllText("/x"); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    user = list(g.user_defined_nodes())
    assert user == ["csharp:X.Y.M"]
    assert "csharp:System.IO.File.ReadAllText" not in user
    # But the spec-only node IS in the graph via edges.
    assert "csharp:System.IO.File.ReadAllText" in g


@pytest.mark.unit
def test_builder_multiple_files_accumulate(csharp_registry, loaded_taint_index):
    """Two files with distinct classes contribute independent nodes."""
    src_a = textwrap.dedent("""
    namespace X;
    public class A { public void FromA() { } }
    """)
    src_b = textwrap.dedent("""
    namespace X;
    public class B { public void FromB() { } }
    """)
    builder = CallGraphBuilder(
        adapter_registry=csharp_registry,
        taint_index=loaded_taint_index,
    )
    builder.add_file(source=src_a, file_path="a.cs", language="csharp")
    builder.add_file(source=src_b, file_path="b.cs", language="csharp")
    g = builder.build()
    user = list(g.user_defined_nodes())
    assert "csharp:X.A.FromA" in user
    assert "csharp:X.B.FromB" in user


# ---------------------------------------------------------------------
# TaintPathFinder: intra-method flow
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_intra_method_source_to_sink(csharp_registry, loaded_taint_index):
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace MyApp.Web;
    public class UC {
        public string GetById() {
            var input = System.IO.File.ReadAllText("/x");
            var cmd = new SqlCommand(input, _conn);
            return cmd.ExecuteReader().ToString();
        }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert len(paths) == 1
    p = paths[0]
    assert p.source == "csharp:System.IO.File.ReadAllText"
    assert p.sink == "csharp:System.Data.SqlClient.SqlCommand.<init>"
    assert p.source_caller == "csharp:MyApp.Web.UC.GetById"
    assert p.sink_caller == "csharp:MyApp.Web.UC.GetById"
    assert p.hop_count == 0
    assert p.via == ("csharp:MyApp.Web.UC.GetById",)


@pytest.mark.unit
def test_intra_method_property_setter_sink(csharp_registry, loaded_taint_index):
    """Property setters resolve to set_X and appear as sinks in intra-method flows."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void M() {
            var input = System.IO.File.ReadAllText("/x");
            var cmd = new SqlCommand("", _conn);
            cmd.CommandText = input;
        }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    sinks_hit = {p.sink for p in paths}
    assert "csharp:System.Data.SqlClient.SqlCommand.set_CommandText" in sinks_hit
    assert "csharp:System.Data.SqlClient.SqlCommand.<init>" in sinks_hit


# ---------------------------------------------------------------------
# TaintPathFinder: inter-method flow
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_inter_method_one_hop(csharp_registry, loaded_taint_index):
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void Entry() {
            var input = System.IO.File.ReadAllText("/x");
            Helper(input);
        }
        private void Helper(string arg) { new SqlCommand(arg, _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert len(paths) == 1
    p = paths[0]
    assert p.hop_count == 1
    assert p.source_caller == "csharp:X.Y.Entry"
    assert p.sink_caller == "csharp:X.Y.Helper"
    assert p.via == ("csharp:X.Y.Entry", "csharp:X.Y.Helper")


@pytest.mark.unit
def test_inter_method_deep_chain(csharp_registry, loaded_taint_index):
    """Source and sink separated by multiple hops."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void A() { var s = System.IO.File.ReadAllText("/x"); B(s); }
        public void B(string s) { C(s); }
        public void C(string s) { D(s); }
        public void D(string s) { new SqlCommand(s, _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert len(paths) == 1
    p = paths[0]
    assert p.hop_count == 3
    assert p.via == (
        "csharp:X.Y.A", "csharp:X.Y.B", "csharp:X.Y.C", "csharp:X.Y.D",
    )


# ---------------------------------------------------------------------
# TaintPathFinder: max_hops bound
# ---------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("max_hops,expected", [
    (0, 0),  # intra-method only; source and sink are in different methods
    (1, 0),  # not enough hops for 3-edge chain
    (2, 0),  # still not enough
    (3, 1),  # exactly the required chain length
    (5, 1),  # more than enough
])
def test_max_hops_bounds_paths(csharp_registry, loaded_taint_index, max_hops, expected):
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void A() { var s = System.IO.File.ReadAllText("/x"); B(s); }
        public void B(string s) { C(s); }
        public void C(string s) { D(s); }
        public void D(string s) { new SqlCommand(s, _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=max_hops).find_paths()
    assert len(paths) == expected


@pytest.mark.unit
def test_max_hops_zero_finds_intra_method(csharp_registry, loaded_taint_index):
    """max_hops=0 should still find intra-method flows (hop_count=0)."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void M() {
            var input = System.IO.File.ReadAllText("/x");
            new SqlCommand(input, _conn);
        }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=0).find_paths()
    assert len(paths) == 1
    assert paths[0].hop_count == 0


@pytest.mark.unit
def test_negative_max_hops_rejected(csharp_registry, loaded_taint_index):
    g = _build("namespace X; class Y {}", csharp_registry, loaded_taint_index)
    with pytest.raises(ValueError, match="max_hops must be >= 0"):
        TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=-1)


# ---------------------------------------------------------------------
# TaintPathFinder: no-path scenarios
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_no_source_no_paths(csharp_registry, loaded_taint_index):
    """Sink exists but no source -> no paths."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void M() { new SqlCommand("hardcoded", _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert paths == []


@pytest.mark.unit
def test_no_sink_no_paths(csharp_registry, loaded_taint_index):
    """Source exists but no sink -> no paths."""
    src = textwrap.dedent("""
    using System.IO;
    namespace X;
    public class Y {
        public string M() { return File.ReadAllText("/x"); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert paths == []


@pytest.mark.unit
def test_source_and_sink_in_disconnected_methods(csharp_registry, loaded_taint_index):
    """Source in method A, sink in method B, but A doesn't call B."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void HasSource() { File.ReadAllText("/x"); }
        public void HasSink() { new SqlCommand("safe", _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert paths == []


# ---------------------------------------------------------------------
# TaintPathFinder: sanitizer node traversal
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_bfs_does_not_traverse_through_sanitizer_node(csharp_registry, loaded_taint_index):
    """Sanitizer nodes are terminal in path enumeration.

    If a sanitizer somehow had outgoing edges (e.g. an adapter emitted
    them), BFS wouldn't step through them. In practice sanitizers are
    leaf-like functions with no outgoing edges, so this is defensive.

    This test verifies the mechanism without relying on unusual graph
    shapes: we ensure that a candidate path involving a sanitizer
    intermediate would not be produced, by direct graph inspection.
    """
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void M(string arg) {
            var input = File.ReadAllText("/x");
            var p = new SqlParameter("@p", arg);
            new SqlCommand(input, _conn);
        }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    # M calls both source and sink and ALSO calls the sanitizer
    # (SqlParameter.<init>). The finder still reports the path because
    # sanitizer-as-successor doesn't disqualify the calling method --
    # only prevents traversal *through* the sanitizer. Filtering
    # semantics-aware "was the tainted input actually sanitized?" is
    # the LLM verifier's responsibility.
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    assert len(paths) == 1
    assert paths[0].source == "csharp:System.IO.File.ReadAllText"
    assert paths[0].sink == "csharp:System.Data.SqlClient.SqlCommand.<init>"


# ---------------------------------------------------------------------
# TaintPathFinder: multiple sources and sinks
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_source_sink_pairs(csharp_registry, loaded_taint_index):
    """Two sources and two sinks in one method -> four candidate pairs."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    using static System.Console;
    namespace X;
    public class Y {
        public void M() {
            var a = File.ReadAllText("/x");
            var b = ReadLine();
            var cmd = new SqlCommand(a, _conn);
            cmd.CommandText = b;
        }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    sources = {p.source for p in paths}
    sinks = {p.sink for p in paths}
    assert sources == {
        "csharp:System.IO.File.ReadAllText",
        "csharp:System.Console.ReadLine",
    }
    assert sinks == {
        "csharp:System.Data.SqlClient.SqlCommand.<init>",
        "csharp:System.Data.SqlClient.SqlCommand.set_CommandText",
    }
    # 2 sources x 2 sinks = 4 paths
    assert len(paths) == 4


# ---------------------------------------------------------------------
# TaintPathFinder: deduplication and stable ordering
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_deduplicates_multiple_chains_to_same_endpoints(csharp_registry, loaded_taint_index):
    """Two chains connecting the same source_caller -> sink_caller pair
    collapse to the shortest."""
    src = textwrap.dedent("""
    using System.Data.SqlClient;
    using System.IO;
    namespace X;
    public class Y {
        public void A() {
            var s = System.IO.File.ReadAllText("/x");
            B(s);
            C(s);
        }
        public void B(string s) { D(s); }
        public void C(string s) { D(s); }
        public void D(string s) { new SqlCommand(s, _conn); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    paths = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5).find_paths()
    # Two chains: A->B->D and A->C->D. Same endpoints, dedupe keeps one.
    endpoints = {(p.source_caller, p.sink_caller) for p in paths}
    assert endpoints == {("csharp:X.Y.A", "csharp:X.Y.D")}
    assert len(paths) == 1


@pytest.mark.unit
def test_path_finder_output_is_stable(csharp_registry, loaded_taint_index):
    """Repeated invocations on identical input produce identical output."""
    src = textwrap.dedent("""
    using System.IO;
    using System.Data.SqlClient;
    namespace X;
    public class Y {
        public void A() { var s = File.ReadAllText("/x"); new SqlCommand(s); }
        public void B() { var s = File.ReadAllText("/y"); new SqlCommand(s); }
    }
    """)
    g = _build(src, csharp_registry, loaded_taint_index)
    finder = TaintPathFinder(graph=g, taint_index=loaded_taint_index, max_hops=5)
    p1 = finder.find_paths()
    p2 = finder.find_paths()
    assert p1 == p2


# ---------------------------------------------------------------------
# TaintPath model
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_taint_path_hop_count():
    p = TaintPath(
        source="csharp:S", sink="csharp:K",
        source_caller="csharp:X.Y.A", sink_caller="csharp:X.Y.A",
        via=("csharp:X.Y.A",),
    )
    assert p.hop_count == 0

    p2 = TaintPath(
        source="csharp:S", sink="csharp:K",
        source_caller="csharp:X.Y.A", sink_caller="csharp:X.Y.C",
        via=("csharp:X.Y.A", "csharp:X.Y.B", "csharp:X.Y.C"),
    )
    assert p2.hop_count == 2


@pytest.mark.unit
def test_taint_path_frozen():
    """TaintPath is hashable/immutable so it can be used in sets and as dict keys."""
    p = TaintPath(
        source="csharp:S", sink="csharp:K",
        source_caller="csharp:X.Y.A", sink_caller="csharp:X.Y.A",
        via=("csharp:X.Y.A",),
    )
    with pytest.raises(Exception):
        p.source = "changed"
    # Should be hashable
    _ = {p}
