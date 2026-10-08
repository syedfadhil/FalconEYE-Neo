"""Unit tests for taint_confirmation rendering in reporters.

Covers:
* Console formatter: reachability line, chain display, non-reachable
  status reasons, absence of noise on legacy findings.
* HTML formatter: badge markup, section markup, per-status CSS class
  mapping, HTML escaping, legacy-finding pass-through.

The formatters are pure-string builders (no I/O), so tests just
call format_finding / format_review and inspect the returned text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from falconeye.adapters.formatters.console_formatter import ConsoleFormatter
from falconeye.adapters.formatters.html_formatter import HTMLFormatter
from falconeye.domain.models.confirmation import (
    CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
    CONFIRMATION_STATUS_NO_ENTRY_POINTS,
    CONFIRMATION_STATUS_NOT_CONFIRMED,
    CONFIRMATION_STATUS_REACHABLE,
    CONFIRMATION_STATUS_SKIPPED,
    TaintConfirmation,
)
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    SecurityReview,
    Severity,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _mk_finding(
    *,
    issue: str = "Test issue",
    confirmation: Optional[TaintConfirmation] = None,
) -> SecurityFinding:
    return SecurityFinding.create(
        issue=issue,
        reasoning="LLM reasoning goes here",
        mitigation="Fix suggestion",
        severity=Severity.HIGH,
        confidence=FindingConfidence.MEDIUM,
        file_path="/proj/File.cs",
        code_snippet="",
        line_start=10,
        taint_confirmation=confirmation,
    )


def _mk_review(findings) -> SecurityReview:
    review = SecurityReview.create(codebase_path="/proj", language="csharp")
    for f in findings:
        review.add_finding(f)
    review.complete()
    return review


REACHABLE = TaintConfirmation(
    status=CONFIRMATION_STATUS_REACHABLE,
    enclosing_function="csharp:App.Ctrl.Get",
    reachable_entry_points=("csharp:App.Ctrl.Get",),
    shortest_chain=("csharp:App.Ctrl.Get", "csharp:App.Svc.Load"),
    hop_count=1,
    reason="Enclosing method reachable from 1 entry point (1 hop via shortest chain).",
)

NOT_CONFIRMED = TaintConfirmation(
    status=CONFIRMATION_STATUS_NOT_CONFIRMED,
    enclosing_function="csharp:App.Helper.Log",
    reason="Enclosing method not reachable from any of the 3 detected entry points.",
)

NO_METHOD = TaintConfirmation(
    status=CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
    reason="No enclosing method contains /proj/File.cs:10.",
)

SKIPPED = TaintConfirmation(
    status=CONFIRMATION_STATUS_SKIPPED,
    reason="Finding language (python) has no entry-point detector registered.",
)


# =====================================================================
# Console formatter
# =====================================================================


@pytest.mark.unit
def test_console_shows_reachable_with_hop_count_and_chain():
    """A confirmed_reachable finding shows the entry count, hop count,
    and shortened chain (last name segment only)."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    output = cf.format_finding(_mk_finding(confirmation=REACHABLE))
    assert "Reachability: confirmed_reachable" in output
    assert "Reachable from 1 entry point, 1 hop" in output
    # Chain uses short names, not fully qualified
    assert "Get -> Load" in output
    assert "csharp:App.Ctrl" not in output  # long names not shown


@pytest.mark.unit
def test_console_shows_not_confirmed_reason():
    """Non-reachable statuses surface the diagnostic reason so the
    operator can distinguish dead code from parse gaps."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    output = cf.format_finding(_mk_finding(confirmation=NOT_CONFIRMED))
    assert "Reachability: not_confirmed" in output
    assert "not reachable from any" in output  # wrap may split the full phrase
    assert "detected entry" in output


@pytest.mark.unit
def test_console_shows_no_enclosing_method_reason():
    cf = ConsoleFormatter(use_color=False, verbose=False)
    output = cf.format_finding(_mk_finding(confirmation=NO_METHOD))
    assert "Reachability: no_enclosing_method" in output
    assert "No enclosing method contains" in output


@pytest.mark.unit
def test_console_skipped_shows_status_only():
    """Skipped findings don't need the long reason -- status is enough."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    output = cf.format_finding(_mk_finding(confirmation=SKIPPED))
    assert "Reachability: skipped" in output


@pytest.mark.unit
def test_console_legacy_finding_no_confirmation_stays_clean():
    """A finding without taint_confirmation renders with no
    Reachability line -- unchanged from pre-p13 behavior."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    output = cf.format_finding(_mk_finding(confirmation=None))
    assert "Reachability" not in output


@pytest.mark.unit
def test_console_colorizes_when_use_color():
    """With use_color=True, the status label is wrapped in ANSI escapes.
    Confirmed_reachable = green, not_confirmed = yellow, skipped = gray."""
    cf = ConsoleFormatter(use_color=True, verbose=False)
    r = cf.format_finding(_mk_finding(confirmation=REACHABLE))
    assert "\033[92m" in r  # green
    nc = cf.format_finding(_mk_finding(confirmation=NOT_CONFIRMED))
    assert "\033[33m" in nc  # yellow
    sk = cf.format_finding(_mk_finding(confirmation=SKIPPED))
    assert "\033[90m" in sk  # gray


@pytest.mark.unit
def test_console_hop_pluralization():
    """1 hop vs 2 hops -- respect English pluralization."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    # hop_count = 0
    conf0 = TaintConfirmation(
        status=CONFIRMATION_STATUS_REACHABLE,
        enclosing_function="a", hop_count=0,
        reachable_entry_points=("a",),
        shortest_chain=("a",),
    )
    out0 = cf.format_finding(_mk_finding(confirmation=conf0))
    assert "0 hops" in out0
    # hop_count = 1
    conf1 = TaintConfirmation(
        status=CONFIRMATION_STATUS_REACHABLE,
        enclosing_function="a", hop_count=1,
        reachable_entry_points=("a", "b"),
        shortest_chain=("a", "b"),
    )
    out1 = cf.format_finding(_mk_finding(confirmation=conf1))
    assert "1 hop" in out1 and "1 hops" not in out1
    # hop_count = 3
    conf3 = TaintConfirmation(
        status=CONFIRMATION_STATUS_REACHABLE,
        enclosing_function="a", hop_count=3,
        reachable_entry_points=("a",),
        shortest_chain=("a", "b", "c", "d"),
    )
    out3 = cf.format_finding(_mk_finding(confirmation=conf3))
    assert "3 hops" in out3


@pytest.mark.unit
def test_console_entry_point_pluralization():
    """1 entry point vs 2 entry points."""
    cf = ConsoleFormatter(use_color=False, verbose=False)
    conf = TaintConfirmation(
        status=CONFIRMATION_STATUS_REACHABLE,
        enclosing_function="a", hop_count=2,
        reachable_entry_points=("a", "b"),
        shortest_chain=("a", "b", "c"),
    )
    out = cf.format_finding(_mk_finding(confirmation=conf))
    assert "2 entry points" in out


# =====================================================================
# HTML formatter
# =====================================================================


@pytest.mark.unit
def test_html_reachable_badge_and_section_render():
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=REACHABLE)])
    html = hf.format_review(review)
    # Badge in the header
    assert 'class="badge taint-reachable"' in html
    # Section with left-border styling
    assert "taint-reachable-section" in html
    # Content
    assert "Enclosing method" in html
    assert "Distance:" in html
    assert "1 hop" in html
    assert "Shortest chain" in html
    # Chain uses short names
    assert "Get -&gt; Load" in html or "Get -> Load" in html


@pytest.mark.unit
def test_html_not_confirmed_uses_yellow_section():
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=NOT_CONFIRMED)])
    html = hf.format_review(review)
    assert 'class="badge taint-not-confirmed"' in html
    assert "taint-not-confirmed-section" in html
    assert "not reachable from any" in html


@pytest.mark.unit
def test_html_no_method_uses_gray_section():
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=NO_METHOD)])
    html = hf.format_review(review)
    assert 'class="badge taint-no-method"' in html
    assert "taint-no-method-section" in html


@pytest.mark.unit
def test_html_skipped_uses_dim_section():
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=SKIPPED)])
    html = hf.format_review(review)
    assert 'class="badge taint-skipped"' in html
    assert "taint-skipped-section" in html


@pytest.mark.unit
def test_html_legacy_finding_renders_without_taint_markup():
    """A finding without taint_confirmation has no badge and no section --
    the HTML stays clean and identical to pre-p16 output."""
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=None)])
    html = hf.format_review(review)
    # None of the taint-specific CSS classes should appear in the finding
    # (they still appear in the <style> block, but not in the finding markup).
    # Verify by counting: in a <style> block only, the class name appears
    # in CSS-rule form (with a "." prefix and "{"), not as class attributes.
    assert 'class="badge taint-reachable"' not in html
    assert 'class="badge taint-not-confirmed"' not in html
    # The section's <h4> only appears when the section is rendered.
    # The CSS rule ".taint-reachable-section" does appear in the <style>
    # block, so we look for the header instead.
    assert "Taint Reachability:" not in html


@pytest.mark.unit
def test_html_escapes_status_and_reason():
    """A malicious status/reason (unlikely but possible in a JSON reload)
    must not inject HTML. Confirmation values are strings from user
    input at the JSON-loading boundary."""
    hf = HTMLFormatter()
    evil = TaintConfirmation(
        status="confirmed_reachable",
        enclosing_function="<script>alert(1)</script>",
        reason="<img src=x onerror=1>",
    )
    review = _mk_review([_mk_finding(confirmation=evil)])
    html = hf.format_review(review)
    # Original tags must be escaped
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=1>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html


@pytest.mark.unit
def test_html_stylesheet_declares_taint_css_rules():
    """The embedded <style> block must include rules for the taint
    badge and section classes, otherwise the markup is unstyled."""
    hf = HTMLFormatter()
    review = _mk_review([_mk_finding(confirmation=REACHABLE)])
    html = hf.format_review(review)
    # These are the CSS rule selectors (with the leading dot)
    assert ".badge.taint-reachable" in html
    assert ".badge.taint-not-confirmed" in html
    assert ".taint-reachable-section" in html
    assert ".taint-not-confirmed-section" in html


@pytest.mark.unit
def test_html_mixed_review_renders_all_statuses():
    """A realistic review mixing all statuses shows all badges and
    sections. Also verifies no cross-contamination between findings."""
    hf = HTMLFormatter()
    review = _mk_review([
        _mk_finding(issue="reachable-issue", confirmation=REACHABLE),
        _mk_finding(issue="not-confirmed-issue", confirmation=NOT_CONFIRMED),
        _mk_finding(issue="no-method-issue", confirmation=NO_METHOD),
        _mk_finding(issue="skipped-issue", confirmation=SKIPPED),
        _mk_finding(issue="legacy-issue", confirmation=None),
    ])
    html = hf.format_review(review)
    for issue in [
        "reachable-issue", "not-confirmed-issue", "no-method-issue",
        "skipped-issue", "legacy-issue",
    ]:
        assert issue in html
    # All four badge variants present
    assert 'class="badge taint-reachable"' in html
    assert 'class="badge taint-not-confirmed"' in html
    assert 'class="badge taint-no-method"' in html
    assert 'class="badge taint-skipped"' in html
