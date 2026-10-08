"""Unit tests for the --confirm-taint integration in ``falconeye review``.

The review pipeline itself needs an LLM and is out of scope for
unit tests, so this file exercises the injection helper
``_run_taint_confirmation`` in isolation. That helper is what
review_command calls when ``--confirm-taint`` is passed.

Coverage:
  * Successful confirmation replaces the findings list in place
  * Empty findings list is a no-op that doesn't error
  * No taint-enabled languages -> warning, no-op
  * Confirmation service raises -> non-fatal, findings preserved
  * Console summary line shows the counts
"""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from falconeye.adapters.cli.commands import _run_taint_confirmation
from falconeye.domain.models.confirmation import (
    CONFIRMATION_STATUS_REACHABLE,
    TaintConfirmation,
)
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    Severity,
)
from falconeye.domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)
from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.infrastructure.taint_adapters.adapter_registry import (
    AdapterRegistry,
)
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_review(findings):
    """Stand-in for SecurityReview: only .findings is used by the helper."""
    return SimpleNamespace(findings=list(findings))


def _make_config(taint_langs=("csharp",), enabled_langs=("csharp", "python")):
    """Config stand-in shaped exactly like the FalconEyeConfig fields
    the helper reads. Avoids constructing a full Pydantic model."""
    per_language = {}
    for lang in taint_langs:
        per_language[lang] = SimpleNamespace(
            taint=SimpleNamespace(enabled=True),
        )
    # Also include enabled non-taint languages so the intersection logic
    # (taint.enabled AND lang in enabled) exercises correctly.
    languages = SimpleNamespace(
        enabled=list(enabled_langs),
        per_language=per_language,
    )
    return SimpleNamespace(languages=languages)


def _make_service():
    reg = AdapterRegistry(); reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    idx = load_taint_index(languages=["csharp"])
    return ReviewConfirmationService(
        adapter_registry=reg,
        entry_point_registry=entry_reg,
        taint_index=idx,
    )


def _make_finding(file_path: str, line: int, issue: str = "test"):
    return SecurityFinding.create(
        issue=issue, reasoning="", mitigation="",
        severity=Severity.HIGH, confidence=FindingConfidence.MEDIUM,
        file_path=file_path, code_snippet="",
        line_start=line,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


@pytest.mark.unit
def test_confirmation_replaces_findings_in_place(tmp_path):
    """After confirmation, review.findings contains new SecurityFinding
    instances with taint_confirmation populated."""
    (tmp_path / "Ctrl.cs").write_text(textwrap.dedent("""\
    namespace N;
    public class Ctrl {
        [HttpGet("/x")]
        public string Get() { return "x"; }
    }
    """))
    review = _make_review([
        _make_finding(str(tmp_path / "Ctrl.cs"), 4, "in Get")
    ])
    original_id = id(review.findings)  # confirm we mutate in-place

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _run_taint_confirmation(
        review=review,
        repo_path=tmp_path,
        confirmation_service=_make_service(),
        config=_make_config(),
        console=console,
    )

    # Same list object, but contents replaced
    assert id(review.findings) == original_id
    assert len(review.findings) == 1
    tc = review.findings[0].taint_confirmation
    assert tc is not None
    assert tc.status == CONFIRMATION_STATUS_REACHABLE

    # One-line diagnostic was printed
    output = buf.getvalue()
    assert "Taint confirmation" in output
    assert "reachable" in output


@pytest.mark.unit
def test_confirmation_no_findings_is_no_op(tmp_path):
    """An empty findings list shouldn't error or print noise."""
    review = _make_review([])
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _run_taint_confirmation(
        review=review,
        repo_path=tmp_path,
        confirmation_service=_make_service(),
        config=_make_config(),
        console=console,
    )
    assert review.findings == []


@pytest.mark.unit
def test_confirmation_no_taint_language_warns_and_skips(tmp_path):
    """If no taint-enabled language is configured, print a warning
    and leave findings untouched (findings kept, no taint_confirmation)."""
    (tmp_path / "x.cs").write_text("class C { }")
    orig_finding = _make_finding(str(tmp_path / "x.cs"), 1)
    review = _make_review([orig_finding])

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _run_taint_confirmation(
        review=review,
        repo_path=tmp_path,
        confirmation_service=_make_service(),
        config=_make_config(taint_langs=(), enabled_langs=("csharp",)),
        console=console,
    )

    assert "skipping --confirm-taint" in buf.getvalue().lower()
    # Findings preserved unchanged
    assert len(review.findings) == 1
    assert review.findings[0].taint_confirmation is None


@pytest.mark.unit
def test_confirmation_exception_is_non_fatal(tmp_path):
    """If the confirmation service raises, print a warning but let
    the review continue with its original findings intact.

    Rationale: the review's primary output is the findings. Failing
    the whole run because call-graph construction crashed on one
    odd source file would be a bad trade.
    """
    orig_finding = _make_finding(str(tmp_path / "x.cs"), 1)
    review = _make_review([orig_finding])

    class BrokenService:
        def confirm(self, **kwargs):
            raise RuntimeError("call graph build blew up")

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    _run_taint_confirmation(
        review=review,
        repo_path=tmp_path,
        confirmation_service=BrokenService(),
        config=_make_config(),
        console=console,
    )

    output = buf.getvalue().lower()
    assert "taint confirmation failed" in output
    assert "runtimeerror" in output
    assert "call graph build blew up" in output
    # Findings intact and un-mutated
    assert len(review.findings) == 1
    assert review.findings[0] is orig_finding


@pytest.mark.unit
def test_confirmation_summary_line_shows_all_counts(tmp_path):
    """The one-line diagnostic includes every status count so the
    operator can see what happened at a glance."""
    (tmp_path / "Ctrl.cs").write_text(textwrap.dedent("""\
    namespace N;
    public class Ctrl {
        [HttpGet("/x")]
        public string Get() { return "x"; }

        public string Helper() { return "y"; }
    }
    """))
    review = _make_review([
        _make_finding(str(tmp_path / "Ctrl.cs"), 4, "in Get"),
        _make_finding(str(tmp_path / "Ctrl.cs"), 6, "in Helper"),
    ])
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    _run_taint_confirmation(
        review=review,
        repo_path=tmp_path,
        confirmation_service=_make_service(),
        config=_make_config(),
        console=console,
    )
    output = buf.getvalue()
    # Should mention both reachable and not_confirmed counts
    assert "1 reachable" in output
    assert "1 not_confirmed" in output
    assert "entry points" in output
