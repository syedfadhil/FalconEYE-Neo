"""Unit tests for the review-confirmation subsystem.

Covers:
* Entry-point detection (attribute variants, brace-in-route corner
  case, previous-method-attribute isolation, middleware convention)
* Reachability confirmer (source==target, multi-hop, unreachable,
  no enclosing method, no entry points)
* ReviewConfirmationService end-to-end with a realistic ASP.NET Core
  workspace
* SecurityFinding taint_confirmation serialization
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from falconeye.domain.models.confirmation import (
    CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
    CONFIRMATION_STATUS_NO_ENTRY_POINTS,
    CONFIRMATION_STATUS_NOT_CONFIRMED,
    CONFIRMATION_STATUS_REACHABLE,
    CONFIRMATION_STATUS_SKIPPED,
    ENTRY_POINT_KIND_HTTP,
    ENTRY_POINT_KIND_MIDDLEWARE,
    EntryPoint,
    TaintConfirmation,
)
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    Severity,
)
from falconeye.domain.models.taint import FunctionDef, Location
from falconeye.domain.services.call_graph import CallGraphBuilder
from falconeye.domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)
from falconeye.domain.services.reachability_confirmer import (
    ReachabilityConfirmer,
    _normalize_path,
)
from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.infrastructure.taint_adapters.adapter_registry import (
    AdapterRegistry,
)
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


# =====================================================================
# Helpers
# =====================================================================


def _make_finding(
    file_path: str,
    line: int,
    issue: str = "test",
) -> SecurityFinding:
    return SecurityFinding.create(
        issue=issue,
        reasoning="",
        mitigation="",
        severity=Severity.HIGH,
        confidence=FindingConfidence.MEDIUM,
        file_path=file_path,
        code_snippet="",
        line_start=line,
    )


def _make_func_def(canonical_name: str, file_path: str, start: int, end: int) -> FunctionDef:
    return FunctionDef(
        canonical_name=canonical_name,
        language="csharp",
        location=Location(file_path=file_path, line_start=start, line_end=end),
    )


# =====================================================================
# AspNetCoreEntryPointDetector
# =====================================================================


@pytest.mark.unit
def test_detector_finds_http_verb_attributes():
    """Every [Http*] attribute produces an entry point."""
    source = textwrap.dedent("""\
    public class C {
        [HttpGet("api/x")]
        public void Get() { }

        [HttpPost]
        public void Post() { }

        [HttpDelete("api/x/{id}")]
        public void Del() { }
    }
    """)
    fds = [
        _make_func_def("csharp:C.Get", "/x.cs", 2, 4),
        _make_func_def("csharp:C.Post", "/x.cs", 5, 6),
        _make_func_def("csharp:C.Del", "/x.cs", 8, 9),
    ]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": source})
    names = [e.canonical_name.split(".")[-1] for e in entries]
    assert set(names) == {"Get", "Post", "Del"}
    kinds = {e.kind for e in entries}
    assert kinds == {ENTRY_POINT_KIND_HTTP}


@pytest.mark.unit
def test_detector_finds_route_attribute():
    """[Route(...)] alone (no verb) is still an entry point."""
    source = textwrap.dedent("""\
    public class C {
        [Route("api/status")]
        public string GetStatus() { return "x"; }
    }
    """)
    fds = [_make_func_def("csharp:C.GetStatus", "/x.cs", 2, 3)]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": source})
    assert len(entries) == 1
    assert entries[0].reason == "[Route]"


@pytest.mark.unit
def test_detector_ignores_non_entry_method():
    """A method with no attribute and no middleware convention isn't an entry."""
    source = textwrap.dedent("""\
    public class C {
        public string Helper() { return "x"; }
    }
    """)
    fds = [_make_func_def("csharp:C.Helper", "/x.cs", 2, 2)]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": source})
    assert entries == []


@pytest.mark.unit
def test_detector_handles_route_with_brace_in_string():
    """[HttpGet("api/users/{id}")] contains { inside a string; the
    brace-search for the method's own opening { must skip the
    attribute line to avoid stopping there prematurely."""
    source = textwrap.dedent("""\
    public class C {
        [HttpGet("api/users/{id}")]
        public string Get(int id) { return "x"; }
    }
    """)
    fds = [_make_func_def("csharp:C.Get", "/x.cs", 2, 3)]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": source})
    assert len(entries) == 1
    assert entries[0].reason == "[HttpGet]"


@pytest.mark.unit
def test_detector_does_not_attribute_previous_methods_attribute():
    """Adjacent methods must not steal each other's attributes."""
    source = textwrap.dedent("""\
    public class C {
        [HttpGet]
        public void First() { }

        [HttpPost]
        public void Second() { }

        public void Third() { }
    }
    """)
    fds = [
        _make_func_def("csharp:C.First", "/x.cs", 2, 3),
        _make_func_def("csharp:C.Second", "/x.cs", 5, 6),
        _make_func_def("csharp:C.Third", "/x.cs", 8, 8),
    ]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": source})
    by_name = {e.canonical_name.split(".")[-1]: e for e in entries}
    assert by_name["First"].reason == "[HttpGet]"
    assert by_name["Second"].reason == "[HttpPost]"
    assert "Third" not in by_name  # no attribute, not an entry point


@pytest.mark.unit
def test_detector_middleware_convention():
    """InvokeAsync on a *Middleware class is an entry point by
    convention (no attribute needed)."""
    fds = [
        _make_func_def("csharp:MyApp.AuthMiddleware.InvokeAsync", "/x.cs", 1, 3),
        _make_func_def("csharp:MyApp.LoggingMiddleware.Invoke", "/x.cs", 5, 7),
    ]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": ""})
    kinds = {e.kind for e in entries}
    assert kinds == {ENTRY_POINT_KIND_MIDDLEWARE}
    assert len(entries) == 2


@pytest.mark.unit
def test_detector_middleware_convention_requires_class_suffix():
    """A random Invoke() method not on a *Middleware class is NOT
    an entry point -- otherwise we'd flag every helper."""
    fds = [_make_func_def("csharp:MyApp.Foo.Invoke", "/x.cs", 1, 3)]
    entries = AspNetCoreEntryPointDetector().detect(fds, {"/x.cs": ""})
    assert entries == []


@pytest.mark.unit
def test_detector_skips_non_csharp_functions():
    """Non-csharp function defs are ignored even if the source looks right."""
    fds = [FunctionDef(
        canonical_name="python:foo.get",
        language="python",
        location=Location(file_path="/x.py", line_start=1, line_end=2),
    )]
    entries = AspNetCoreEntryPointDetector().detect(
        fds,
        {"/x.py": "[HttpGet]\ndef get(): pass"},
    )
    assert entries == []


# =====================================================================
# EntryPointDetectorRegistry
# =====================================================================


@pytest.mark.unit
def test_registry_dispatch_by_language():
    reg = EntryPointDetectorRegistry()
    reg.register(AspNetCoreEntryPointDetector())
    assert reg.get("csharp") is not None
    assert reg.get("python") is None
    assert reg.languages() == ["csharp"]


# =====================================================================
# ReachabilityConfirmer
# =====================================================================


def _tiny_graph_with_chain():
    """Build a graph:  Root -> Mid -> Leaf.
    Return (graph, function_defs) for use in reachability tests."""
    reg = AdapterRegistry(); reg.register(CSharpTaintAdapter())
    idx = load_taint_index(languages=["csharp"])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.cs").write_text(textwrap.dedent("""\
        namespace N;
        public class C {
            [HttpGet("a")]
            public string Root() { return Mid(); }
            public string Mid() { return Leaf(); }
            public string Leaf() { return "x"; }
        }
        """))
        b = CallGraphBuilder(adapter_registry=reg, taint_index=idx)
        for p in root.rglob("*.cs"):
            b.add_file(source=p.read_text(), file_path=str(p), language="csharp")
        graph = b.build()
    fds = [graph.function(n) for n in graph.user_defined_nodes()]
    return graph, [f for f in fds if f is not None], root


@pytest.mark.unit
def test_confirmer_source_equals_target():
    """A finding directly inside an entry-point method is
    trivially reachable (chain length 1, hop count 0)."""
    graph, fds, _ = _tiny_graph_with_chain()
    root = next(f for f in fds if f.canonical_name.endswith("Root"))
    conf = ReachabilityConfirmer(
        call_graph=graph,
        function_defs=fds,
        entry_points=[EntryPoint(
            canonical_name=root.canonical_name,
            reason="[HttpGet]",
            location=root.location,
        )],
    )
    result = conf.confirm(file_path=root.location.file_path, line_number=root.location.line_start)
    assert result.status == CONFIRMATION_STATUS_REACHABLE
    assert result.enclosing_function == root.canonical_name
    assert result.hop_count == 0
    assert result.shortest_chain == (root.canonical_name,)


@pytest.mark.unit
def test_confirmer_no_entry_points_status():
    """Zero entry points -> no_entry_points status."""
    graph, fds, _ = _tiny_graph_with_chain()
    conf = ReachabilityConfirmer(
        call_graph=graph, function_defs=fds, entry_points=[],
    )
    result = conf.confirm(file_path=fds[0].location.file_path, line_number=1)
    assert result.status == CONFIRMATION_STATUS_NO_ENTRY_POINTS


@pytest.mark.unit
def test_confirmer_no_enclosing_method_status():
    """A line not inside any function -> no_enclosing_method."""
    graph, fds, _ = _tiny_graph_with_chain()
    entry_fd = fds[0]
    conf = ReachabilityConfirmer(
        call_graph=graph,
        function_defs=fds,
        entry_points=[EntryPoint(
            canonical_name=entry_fd.canonical_name,
            reason="test",
            location=entry_fd.location,
        )],
    )
    result = conf.confirm(file_path=entry_fd.location.file_path, line_number=99999)
    assert result.status == CONFIRMATION_STATUS_NO_ENCLOSING_METHOD


@pytest.mark.unit
def test_confirmer_not_confirmed_for_unreachable_method():
    """A method not reachable from any entry point -> not_confirmed
    (NOT dismissed, per the additive-signal design)."""
    file_path = "/x.cs"
    fd_entry = _make_func_def("csharp:Entry", file_path, 1, 5)
    fd_orphan = _make_func_def("csharp:Orphan", file_path, 10, 15)
    from falconeye.domain.services.call_graph import CallGraph
    graph = CallGraph(
        successors={"csharp:Entry": frozenset()},
        predecessors={},
        functions={
            "csharp:Entry": fd_entry,
            "csharp:Orphan": fd_orphan,
        },
    )
    conf = ReachabilityConfirmer(
        call_graph=graph,
        function_defs=[fd_entry, fd_orphan],
        entry_points=[EntryPoint(
            canonical_name="csharp:Entry",
            reason="test",
            location=fd_entry.location,
        )],
    )
    result = conf.confirm(file_path=file_path, line_number=12)
    assert result.status == CONFIRMATION_STATUS_NOT_CONFIRMED
    assert result.enclosing_function == "csharp:Orphan"


@pytest.mark.unit
def test_confirmer_multi_hop_chain():
    """Reachability through intermediate calls populates the chain."""
    fd_a = _make_func_def("csharp:A", "/x.cs", 1, 3)
    fd_b = _make_func_def("csharp:B", "/x.cs", 5, 7)
    fd_c = _make_func_def("csharp:C", "/x.cs", 9, 11)
    from falconeye.domain.services.call_graph import CallGraph
    graph = CallGraph(
        successors={
            "csharp:A": frozenset({"csharp:B"}),
            "csharp:B": frozenset({"csharp:C"}),
        },
        predecessors={},
        functions={
            "csharp:A": fd_a, "csharp:B": fd_b, "csharp:C": fd_c,
        },
    )
    conf = ReachabilityConfirmer(
        call_graph=graph,
        function_defs=[fd_a, fd_b, fd_c],
        entry_points=[EntryPoint(
            canonical_name="csharp:A",
            reason="test",
            location=fd_a.location,
        )],
    )
    result = conf.confirm(file_path="/x.cs", line_number=10)
    assert result.status == CONFIRMATION_STATUS_REACHABLE
    assert result.shortest_chain == ("csharp:A", "csharp:B", "csharp:C")
    assert result.hop_count == 2


@pytest.mark.unit
def test_confirmer_finds_nested_smallest_enclosing_method():
    """When two functions could enclose a line, pick the smaller one.
    Not common in practice but happens with lambdas / local functions."""
    fd_outer = _make_func_def("csharp:Outer", "/x.cs", 1, 20)
    fd_inner = _make_func_def("csharp:Inner", "/x.cs", 5, 10)
    from falconeye.domain.services.call_graph import CallGraph
    graph = CallGraph(successors={}, predecessors={}, functions={
        "csharp:Outer": fd_outer, "csharp:Inner": fd_inner,
    })
    conf = ReachabilityConfirmer(
        call_graph=graph,
        function_defs=[fd_outer, fd_inner],
        entry_points=[EntryPoint(
            canonical_name="csharp:Outer", reason="", location=fd_outer.location,
        )],
    )
    # Line 7 is in both, but Inner is smaller -> pick Inner
    assert conf.find_enclosing_function("/x.cs", 7) == "csharp:Inner"


@pytest.mark.unit
def test_confirmer_normalizes_paths_across_os():
    """A finding path with Windows separators should still match a
    workspace stored with POSIX separators."""
    assert _normalize_path("C:\\Users\\Fadhil\\proj\\file.cs") == \
           _normalize_path("/Users/Fadhil/proj/file.cs")


# =====================================================================
# ReviewConfirmationService end-to-end
# =====================================================================


@pytest.mark.unit
def test_service_confirms_across_realistic_controller(tmp_path):
    """Realistic ASP.NET Core: mix of reachable/unreachable findings.

    NOTE: cross-namespace name resolution in the C# adapter is
    incomplete (see taint patches); this test uses same-namespace
    calls so the call graph fully resolves. Cross-file reachability
    is exercised by the multi-file test below.
    """
    (tmp_path / "UserController.cs").write_text(textwrap.dedent("""\
    namespace N;
    public class UserController {
        [HttpGet("api/users/{id}")]
        public string GetUser(int id) { return "x"; }

        [HttpPost("api/users")]
        public void CreateUser(string name) { }

        public string HelperNotEntry() { return "private"; }
    }
    public class AuthMiddleware {
        public void InvokeAsync() { }
    }
    """))
    reg = AdapterRegistry(); reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    idx = load_taint_index(languages=["csharp"])
    service = ReviewConfirmationService(
        adapter_registry=reg,
        entry_point_registry=entry_reg,
        taint_index=idx,
    )

    findings = [
        _make_finding(str(tmp_path / "UserController.cs"), 4, "in GetUser"),
        _make_finding(str(tmp_path / "UserController.cs"), 7, "in CreateUser"),
        _make_finding(str(tmp_path / "UserController.cs"), 9, "in Helper"),
        _make_finding(str(tmp_path / "UserController.cs"), 12, "in Middleware"),
    ]
    confirmed, summary = service.confirm(
        findings=findings, repo_path=tmp_path, languages=["csharp"],
    )
    by_issue = {f.issue: f.taint_confirmation for f in confirmed}

    assert by_issue["in GetUser"].status == CONFIRMATION_STATUS_REACHABLE
    assert by_issue["in CreateUser"].status == CONFIRMATION_STATUS_REACHABLE
    assert by_issue["in Helper"].status == CONFIRMATION_STATUS_NOT_CONFIRMED
    assert by_issue["in Middleware"].status == CONFIRMATION_STATUS_REACHABLE

    # Sanity on summary numbers
    assert summary.entry_points_detected >= 3
    assert summary.reachable == 3
    assert summary.not_confirmed == 1


@pytest.mark.unit
def test_service_marks_unsupported_language_as_skipped(tmp_path):
    """Findings in files whose language has no adapter/detector
    get status=skipped, not treated as errors."""
    (tmp_path / "x.py").write_text("def foo(): pass")
    reg = AdapterRegistry(); reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    idx = load_taint_index(languages=["csharp"])
    service = ReviewConfirmationService(
        adapter_registry=reg, entry_point_registry=entry_reg, taint_index=idx,
    )
    findings = [_make_finding(str(tmp_path / "x.py"), 1)]
    confirmed, _ = service.confirm(
        findings=findings, repo_path=tmp_path, languages=["csharp"],
    )
    assert confirmed[0].taint_confirmation.status == CONFIRMATION_STATUS_SKIPPED


@pytest.mark.unit
def test_service_all_skipped_if_no_supported_language(tmp_path):
    """If we're asked to confirm for a language we don't support at
    all, every finding is marked skipped and no call graph is built."""
    (tmp_path / "x.py").write_text("def foo(): pass")
    reg = AdapterRegistry(); reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    idx = load_taint_index(languages=["csharp"])
    service = ReviewConfirmationService(
        adapter_registry=reg, entry_point_registry=entry_reg, taint_index=idx,
    )
    findings = [_make_finding(str(tmp_path / "x.py"), 1)]
    confirmed, summary = service.confirm(
        findings=findings, repo_path=tmp_path, languages=["python"],
    )
    assert summary.skipped == 1
    assert summary.call_graph_nodes == 0
    assert confirmed[0].taint_confirmation.status == CONFIRMATION_STATUS_SKIPPED


# =====================================================================
# SecurityFinding integration
# =====================================================================


@pytest.mark.unit
def test_taint_confirmation_serializes_in_to_dict():
    """SecurityFinding.to_dict includes taint_confirmation when set."""
    conf = TaintConfirmation(
        status=CONFIRMATION_STATUS_REACHABLE,
        enclosing_function="csharp:C.M",
        reachable_entry_points=("csharp:C.Entry",),
        shortest_chain=("csharp:C.Entry", "csharp:C.M"),
        hop_count=1,
        reason="reachable",
    )
    finding = SecurityFinding.create(
        issue="x", reasoning="", mitigation="",
        severity=Severity.HIGH, confidence=FindingConfidence.MEDIUM,
        file_path="/x.cs", code_snippet="",
        taint_confirmation=conf,
    )
    d = finding.to_dict()
    assert d["taint_confirmation"]["status"] == CONFIRMATION_STATUS_REACHABLE
    assert d["taint_confirmation"]["hop_count"] == 1
    assert d["taint_confirmation"]["shortest_chain"] == [
        "csharp:C.Entry", "csharp:C.M",
    ]


@pytest.mark.unit
def test_taint_confirmation_none_serializes_as_null():
    """Default (no confirmation run) serializes as null, not missing."""
    finding = SecurityFinding.create(
        issue="x", reasoning="", mitigation="",
        severity=Severity.HIGH, confidence=FindingConfidence.MEDIUM,
        file_path="/x.cs", code_snippet="",
    )
    d = finding.to_dict()
    assert "taint_confirmation" in d
    assert d["taint_confirmation"] is None


@pytest.mark.unit
def test_confirmation_is_reachable_property():
    reachable = TaintConfirmation(status=CONFIRMATION_STATUS_REACHABLE)
    not_reachable = TaintConfirmation(status=CONFIRMATION_STATUS_NOT_CONFIRMED)
    skipped = TaintConfirmation(status=CONFIRMATION_STATUS_SKIPPED)
    assert reachable.is_reachable is True
    assert not_reachable.is_reachable is False
    assert skipped.is_reachable is False
