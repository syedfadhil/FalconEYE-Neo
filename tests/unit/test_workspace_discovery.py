"""Unit tests for the shared workspace walker and fair-share discovery.

Coverage:
  * iter_workspace_files: skip fragments are relative to the root (a
    workspace under ``build/`` is still walked), skipped trees are
    pruned rather than enumerated, order is sorted, extensions are
    case-insensitive, size cap, missing root, multi-segment fragments
  * GrepRunner: works under ``build/``; total cap shared fairly so a
    late, light pattern isn't starved; pattern-major output order;
    each file read once per run; trimmed patterns named in a warning
  * _fair_share allocation table
  * ReviewConfirmationService / TaintScanService: workspaces under
    ``build/`` are walked (confirmation no longer collapses to
    no_entry_points)
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path

import pytest

from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.application.services.taint_scan_service import TaintScanService
from falconeye.domain.models.confirmation import CONFIRMATION_STATUS_REACHABLE
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    Severity,
)
from falconeye.domain.models.skill import DiscoveryPattern
from falconeye.domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)
from falconeye.infrastructure import workspace_files
from falconeye.infrastructure.skills.grep_runner import GrepRunner, _fair_share
from falconeye.infrastructure.taint_adapters.adapter_registry import (
    AdapterRegistry,
)
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index
from falconeye.infrastructure.workspace_files import iter_workspace_files


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _names(paths):
    return [p.name for p in paths]


# ---------------------------------------------------------------
# iter_workspace_files
# ---------------------------------------------------------------


@pytest.mark.unit
def test_walker_ignores_skip_names_above_the_root(tmp_path):
    """The workspace's own location must never cause a skip."""
    ws = tmp_path / "build" / "Portal"
    _touch(ws / "src" / "Repo.cs")
    found = list(iter_workspace_files(
        ws, extensions={".cs"}, skip_fragments=("/build/", "/src/x/"),
    ))
    assert _names(found) == ["Repo.cs"]


@pytest.mark.unit
def test_walker_skips_directories_inside_the_root(tmp_path):
    _touch(tmp_path / "src" / "keep.cs")
    _touch(tmp_path / "node_modules" / "pkg" / "drop.cs")
    _touch(tmp_path / "bin" / "Debug" / "drop2.cs")
    found = list(iter_workspace_files(
        tmp_path, extensions={".cs"},
        skip_fragments=("/node_modules/", "/bin/"),
    ))
    assert _names(found) == ["keep.cs"]


@pytest.mark.unit
def test_walker_prunes_skipped_trees_instead_of_enumerating_them(
    tmp_path, monkeypatch,
):
    """os.walk must never be handed a skipped directory -- that's what
    makes a 100k-file node_modules free instead of a full walk."""
    _touch(tmp_path / "src" / "a.cs")
    _touch(tmp_path / "node_modules" / "deep" / "deeper" / "b.cs")

    visited = []
    real_walk = os.walk

    def spy(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            visited.append(Path(dirpath).name)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(workspace_files.os, "walk", spy)
    list(iter_workspace_files(
        tmp_path, extensions={".cs"}, skip_fragments=("/node_modules/",),
    ))
    assert "src" in visited
    assert not {"node_modules", "deep", "deeper"} & set(visited)


@pytest.mark.unit
def test_walker_order_is_sorted_and_stable(tmp_path):
    for rel in ("z/b.cs", "b.cs", "m/a.cs", "a.cs", "m/c.cs"):
        _touch(tmp_path / rel)
    found = [p.relative_to(tmp_path).as_posix()
             for p in iter_workspace_files(tmp_path, extensions={".cs"})]
    assert found == ["a.cs", "b.cs", "m/a.cs", "m/c.cs", "z/b.cs"]


@pytest.mark.unit
def test_walker_extensions_are_case_insensitive_and_dot_optional(tmp_path):
    _touch(tmp_path / "Upper.CS")
    _touch(tmp_path / "lower.cs")
    _touch(tmp_path / "other.py")
    found = list(iter_workspace_files(tmp_path, extensions=("CS",)))
    assert sorted(_names(found)) == ["Upper.CS", "lower.cs"]


@pytest.mark.unit
def test_walker_size_cap(tmp_path):
    _touch(tmp_path / "small.cs", "x")
    _touch(tmp_path / "big.cs", "x" * 5000)
    found = list(iter_workspace_files(
        tmp_path, extensions={".cs"}, max_file_bytes=100,
    ))
    assert _names(found) == ["small.cs"]


@pytest.mark.unit
def test_walker_missing_root_or_no_extensions_yields_nothing(tmp_path):
    _touch(tmp_path / "a.cs")
    assert list(iter_workspace_files(tmp_path / "nope", extensions={".cs"})) == []
    assert list(iter_workspace_files(tmp_path, extensions=())) == []


@pytest.mark.unit
def test_walker_multi_segment_fragment(tmp_path):
    """``/src/gen/`` skips src/gen but not a top-level gen/."""
    _touch(tmp_path / "src" / "gen" / "skip.cs")
    _touch(tmp_path / "gen" / "keep.cs")
    found = list(iter_workspace_files(
        tmp_path, extensions={".cs"}, skip_fragments=("/src/gen/",),
    ))
    assert _names(found) == ["keep.cs"]


# ---------------------------------------------------------------
# GrepRunner
# ---------------------------------------------------------------


@pytest.mark.unit
def test_grep_runner_finds_hits_when_workspace_is_under_build_dir(tmp_path):
    """Regression: matched against the absolute path, ``/build/`` in
    the workspace's own location dropped every file."""
    ws = tmp_path / "build" / "Portal"
    _touch(ws / "Repo.cs", 'db.ExecuteSqlRaw("SELECT " + id);')
    hits = GrepRunner(workspace_root=ws).discover([
        DiscoveryPattern(regex="ExecuteSqlRaw", extensions=(".cs",)),
    ])
    assert len(hits) == 1


def _heavy_and_light_workspace(tmp_path):
    """Three patterns with 40 hits each, then a light one with 3."""
    lines = []
    for i in range(40):
        lines += [f"alpha{i}", f"beta{i}", f"gamma{i}"]
    lines += ["delta0", "delta1", "delta2"]
    _touch(tmp_path / "big.py", "\n".join(lines))
    return [
        DiscoveryPattern(regex=r"alpha\d", extensions=(".py",), description="A"),
        DiscoveryPattern(regex=r"beta\d", extensions=(".py",), description="B"),
        DiscoveryPattern(regex=r"gamma\d", extensions=(".py",), description="C"),
        DiscoveryPattern(regex=r"delta\d", extensions=(".py",), description="D"),
    ]


@pytest.mark.unit
def test_total_cap_is_shared_fairly_across_patterns(tmp_path):
    """First-come-first-served gave A=40, B=10, C=0, D=0: the light,
    late pattern was starved. Fair share keeps all of D and splits
    the rest evenly."""
    patterns = _heavy_and_light_workspace(tmp_path)
    hits = GrepRunner(
        workspace_root=tmp_path, max_hits_per_pattern=40, max_total_hits=50,
    ).discover(patterns)

    by_pattern = {d: sum(1 for h in hits if h.pattern_description == d) for d in "ABCD"}
    assert len(hits) == 50
    assert by_pattern["D"] == 3
    assert by_pattern == {"A": 16, "B": 16, "C": 15, "D": 3}


@pytest.mark.unit
def test_hits_are_returned_in_pattern_order(tmp_path):
    """Files are scanned once for all patterns, but output stays
    pattern-major as before: every A hit precedes every B hit."""
    _touch(tmp_path / "a.py", "beta1\nalpha1")
    _touch(tmp_path / "b.py", "alpha2\nbeta2")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="alpha", extensions=(".py",), description="A"),
        DiscoveryPattern(regex="beta", extensions=(".py",), description="B"),
    ])
    assert [h.pattern_description for h in hits] == ["A", "A", "B", "B"]
    assert [Path(h.file_path).name for h in hits] == ["a.py", "b.py", "a.py", "b.py"]


@pytest.mark.unit
def test_each_file_is_read_once_per_discovery_run(tmp_path, monkeypatch):
    """Nine injection-prevention patterns used to mean nine reads of
    every matching file; now it is one."""
    for i in range(3):
        _touch(tmp_path / f"f{i}.py", "alpha beta gamma")
    opened = []
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex=word, extensions=(".py",))
        for word in ("alpha", "beta", "gamma")
    ])
    assert len(hits) == 9
    assert sorted(opened) == ["f0.py", "f1.py", "f2.py"]


@pytest.mark.unit
def test_trim_warning_names_the_trimmed_patterns(tmp_path, caplog):
    patterns = _heavy_and_light_workspace(tmp_path)
    with caplog.at_level(logging.WARNING):
        GrepRunner(
            workspace_root=tmp_path, max_hits_per_pattern=40, max_total_hits=50,
        ).discover(patterns)
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "capped at 50" in message
    assert "'A' 40->16" in message
    assert "'D'" not in message  # D kept everything


@pytest.mark.unit
@pytest.mark.parametrize("counts, budget, expected", [
    ([40, 40, 40, 3], 50, [16, 16, 15, 3]),
    ([5, 5], 100, [5, 5]),             # fits: keep everything
    ([0, 10], 4, [0, 4]),              # empty patterns take nothing
    ([10, 10, 10], 0, [0, 0, 0]),
    ([1, 1, 1, 1, 1], 3, [1, 1, 1, 0, 0]),  # remainder to earlier patterns
    ([], 10, []),
])
def test_fair_share_allocation(counts, budget, expected):
    assert _fair_share(counts, budget) == expected


# ---------------------------------------------------------------
# Confirmation and taint scan walkers
# ---------------------------------------------------------------


@pytest.mark.unit
def test_confirmation_under_build_dir_still_finds_entry_points(tmp_path):
    """Regression: the call graph used to be empty for a repo under
    ``/build/``, so every finding came back no_entry_points."""
    ws = tmp_path / "build" / "Portal"
    ctrl = _touch(ws / "Ctrl.cs", textwrap.dedent("""\
        namespace N;
        public class Ctrl {
            [HttpGet("/x")]
            public string Get() { return "x"; }
        }
        """))
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    service = ReviewConfirmationService(
        adapter_registry=reg,
        entry_point_registry=entry_reg,
        taint_index=load_taint_index(languages=["csharp"]),
    )
    finding = SecurityFinding.create(
        issue="in Get", reasoning="", mitigation="",
        severity=Severity.HIGH, confidence=FindingConfidence.MEDIUM,
        file_path=str(ctrl), code_snippet="", line_start=4,
    )
    confirmed, summary = service.confirm(
        findings=[finding], repo_path=ws, languages=["csharp"],
    )
    assert summary.entry_points_detected == 1
    assert confirmed[0].taint_confirmation.status == CONFIRMATION_STATUS_REACHABLE


@pytest.mark.unit
def test_taint_scan_walk_under_build_dir(tmp_path):
    ws = tmp_path / "build" / "Portal"
    _touch(ws / "src" / "Repo.cs")
    _touch(ws / "obj" / "Generated.cs")
    service = TaintScanService(
        adapter_registry=AdapterRegistry(),
        taint_index=load_taint_index(languages=["csharp"]),
        verifier=None,
    )
    assert _names(service._walk_files(ws, {".cs", ".csx"})) == ["Repo.cs"]
