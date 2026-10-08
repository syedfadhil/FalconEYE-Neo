"""Regression tests for the shape-tolerant _finding_from_dict loader.

The JSON reporter (``falconeye review --output json``) emits findings
with wrapped ``confidence``, wrapped ``location``, and wrapped
``verifier`` fields. SecurityFinding.to_dict() (used by
``falconeye taint-confirm --output json``) emits the same data with
flat fields. Both shapes must load cleanly through _finding_from_dict
because the operator can pipe either into ``falconeye taint-confirm
--findings <file>``.

The bug this file covers: prior to this patch, the loader used
``FindingConfidence(d.get("confidence"))`` which failed with
``{'value': 'high', 'level': 'high'} is not a valid FindingConfidence``
whenever the input was JSON reporter output.
"""

from __future__ import annotations

import json

import pytest

from falconeye.adapters.cli.commands_taint_confirm import (
    _confidence_from_json,
    _finding_from_dict,
    _location_from_json,
    _severity_from_json,
    _verifier_from_json,
)
from falconeye.adapters.formatters.json_formatter import JSONFormatter
from falconeye.domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    SecurityReview,
    Severity,
)


# ---------------------------------------------------------------
# _confidence_from_json
# ---------------------------------------------------------------


@pytest.mark.unit
def test_confidence_bare_string():
    assert _confidence_from_json("high") == FindingConfidence.HIGH
    assert _confidence_from_json("medium") == FindingConfidence.MEDIUM
    assert _confidence_from_json("low") == FindingConfidence.LOW


@pytest.mark.unit
def test_confidence_wrapped_uses_value_field():
    """{"value": "high", "level": "high"} shape from JSON reporter."""
    assert _confidence_from_json(
        {"value": "high", "level": "high"}
    ) == FindingConfidence.HIGH


@pytest.mark.unit
def test_confidence_wrapped_falls_back_to_level():
    """If ``value`` is missing but ``level`` is present, use level."""
    assert _confidence_from_json(
        {"level": "medium"}
    ) == FindingConfidence.MEDIUM


@pytest.mark.unit
def test_confidence_none_defaults_to_medium():
    assert _confidence_from_json(None) == FindingConfidence.MEDIUM


@pytest.mark.unit
def test_confidence_empty_dict_defaults_to_medium():
    """A ``{}`` should NOT raise; default to medium."""
    assert _confidence_from_json({}) == FindingConfidence.MEDIUM


# ---------------------------------------------------------------
# _severity_from_json
# ---------------------------------------------------------------


@pytest.mark.unit
def test_severity_bare_string_common_values():
    assert _severity_from_json("critical") == Severity.CRITICAL
    assert _severity_from_json("high") == Severity.HIGH
    assert _severity_from_json("medium") == Severity.MEDIUM
    assert _severity_from_json("low") == Severity.LOW


@pytest.mark.unit
def test_severity_wrapped_shape_also_accepted():
    """Future-proofing: if the reporter ever wraps severity, we accept it."""
    assert _severity_from_json(
        {"value": "high", "level": "high"}
    ) == Severity.HIGH


@pytest.mark.unit
def test_severity_none_defaults_to_medium():
    assert _severity_from_json(None) == Severity.MEDIUM


# ---------------------------------------------------------------
# _location_from_json
# ---------------------------------------------------------------


@pytest.mark.unit
def test_location_flat_shape():
    """SecurityFinding.to_dict() output: flat fields at top level."""
    d = {
        "file_path": "/x.cs",
        "line_start": 42,
        "line_end": 44,
    }
    assert _location_from_json(d) == ("/x.cs", 42, 44)


@pytest.mark.unit
def test_location_wrapped_shape():
    """JSON reporter output: location nested under ``location``."""
    d = {
        "location": {
            "file_path": "/y.cs",
            "line_start": 10,
            "line_end": 15,
        },
    }
    assert _location_from_json(d) == ("/y.cs", 10, 15)


@pytest.mark.unit
def test_location_wrapped_preferred_when_both_present():
    """If both wrapped and flat are somehow present, wrapped wins.

    This can happen in a partially-migrated report or a hand-authored
    JSON that mixes the shapes. Wrapping is the more common source
    (JSON reporter) so it takes precedence.
    """
    d = {
        "file_path": "/flat.cs",
        "line_start": 1,
        "line_end": 2,
        "location": {
            "file_path": "/wrapped.cs",
            "line_start": 100,
            "line_end": 200,
        },
    }
    assert _location_from_json(d) == ("/wrapped.cs", 100, 200)


@pytest.mark.unit
def test_location_missing_all_fields():
    """No location info at all -> empty path, None lines."""
    assert _location_from_json({}) == ("", None, None)


# ---------------------------------------------------------------
# _verifier_from_json
# ---------------------------------------------------------------


@pytest.mark.unit
def test_verifier_flat_shape():
    d = {
        "verifier_verdict": "confirmed",
        "verifier_reasoning": "clear",
        "verifier_counterargument": "none",
    }
    assert _verifier_from_json(d) == ("confirmed", "clear", "none")


@pytest.mark.unit
def test_verifier_wrapped_shape():
    d = {
        "verifier": {
            "verdict": "downgrade",
            "reasoning": "might be safe",
            "counterargument": "or might not",
        },
    }
    assert _verifier_from_json(d) == (
        "downgrade", "might be safe", "or might not",
    )


@pytest.mark.unit
def test_verifier_missing_returns_defaults():
    v, r, c = _verifier_from_json({})
    assert v is None
    assert r == ""
    assert c == ""


# ---------------------------------------------------------------
# _finding_from_dict end-to-end
# ---------------------------------------------------------------


@pytest.mark.unit
def test_finding_from_dict_wrapped_shape_operator_case():
    """The exact input shape from the operator's bug report.

    Prior to this patch, this raised:
        {'value': 'high', 'level': 'high'} is not a valid FindingConfidence
    """
    d = {
        "id": "00000000-0000-0000-0000-000000000001",
        "issue": "SQL injection",
        "severity": "high",
        "confidence": {"value": "high", "level": "high"},
        "reasoning": "traced from HTTP query param",
        "mitigation": "use parameterized queries",
        "location": {
            "file_path": "/proj/UserController.cs",
            "line_start": 42,
            "line_end": 44,
        },
        "code_snippet": "cursor.execute(...)",
        "cwe_id": "CWE-89",
    }
    f = _finding_from_dict(d)
    assert f.confidence == FindingConfidence.HIGH
    assert f.severity == Severity.HIGH
    assert f.file_path == "/proj/UserController.cs"
    assert f.line_start == 42
    assert f.line_end == 44
    assert f.cwe_id == "CWE-89"
    assert f.issue == "SQL injection"


@pytest.mark.unit
def test_finding_from_dict_flat_shape_still_works():
    """Backwards compatibility: bare-string / flat-field JSON also loads.

    Users who hand-author findings JSON or use taint-confirm's own
    --output json (which emits SecurityFinding.to_dict()) rely on
    this shape.
    """
    d = {
        "id": "00000000-0000-0000-0000-000000000002",
        "issue": "Path traversal",
        "severity": "medium",
        "confidence": "low",
        "file_path": "/proj/File.cs",
        "line_start": 10,
        "line_end": 12,
        "verifier_verdict": "downgrade",
    }
    f = _finding_from_dict(d)
    assert f.confidence == FindingConfidence.LOW
    assert f.severity == Severity.MEDIUM
    assert f.file_path == "/proj/File.cs"
    assert f.verifier_verdict == "downgrade"


@pytest.mark.unit
def test_finding_from_dict_round_trip_through_json_reporter():
    """The critical end-to-end: JSON reporter output loads cleanly.

    This is what the operator was doing: run
        falconeye review Portal --output json --output-file r.json
        falconeye taint-confirm Portal --findings r.json
    The pipeline must complete without raising.
    """
    original = SecurityFinding.create(
        issue="RT test", reasoning="r", mitigation="m",
        severity=Severity.CRITICAL, confidence=FindingConfidence.HIGH,
        file_path="/x.cs", code_snippet="s",
        line_start=1, line_end=2,
        cwe_id="CWE-79",
    )
    review = SecurityReview.create(codebase_path="/x", language="csharp")
    review.add_finding(original)
    review.complete()

    # Format via the reporter (this is where the wrap happens)
    json_text = JSONFormatter().format_review(review)
    parsed = json.loads(json_text)
    finding_dict = parsed["findings"][0]

    # Confirm the reporter really is producing the wrapped shape
    assert isinstance(finding_dict["confidence"], dict)
    assert "location" in finding_dict

    # Load through our loader - must not raise
    reloaded = _finding_from_dict(finding_dict)
    assert reloaded.confidence == FindingConfidence.HIGH
    assert reloaded.severity == Severity.CRITICAL
    assert reloaded.file_path == "/x.cs"
    assert reloaded.line_start == 1
    assert reloaded.line_end == 2
    assert reloaded.cwe_id == "CWE-79"


@pytest.mark.unit
def test_finding_from_dict_tolerates_missing_id():
    """Empty or missing id -> synthetic all-zeros UUID (not a crash)."""
    from uuid import UUID
    d = {"issue": "no id", "severity": "medium", "confidence": "medium"}
    f = _finding_from_dict(d)
    assert f.id == UUID(int=0)


@pytest.mark.unit
def test_finding_from_dict_missing_optional_fields():
    """Only the required identity fields present; everything else defaults."""
    d = {
        "id": "00000000-0000-0000-0000-000000000003",
        "issue": "minimal",
        "severity": "low",
        "confidence": "low",
    }
    f = _finding_from_dict(d)
    assert f.issue == "minimal"
    assert f.file_path == ""
    assert f.line_start is None
    assert f.line_end is None
    assert f.verifier_verdict is None
    assert f.tags == []
    assert f.evidence_lines == []
    assert f.validation_warnings == []
