"""Unit tests for output formatter schemas and report content."""

import json

from falconeye.adapters.formatters.console_formatter import ConsoleFormatter
from falconeye.adapters.formatters.html_formatter import HTMLFormatter
from falconeye.adapters.formatters.json_formatter import JSONFormatter
from falconeye.adapters.formatters.sarif_formatter import SARIFFormatter
from falconeye.domain.models.security import FindingConfidence, SecurityFinding, SecurityReview, Severity


def sample_review():
    review = SecurityReview.create("/tmp/project", "python")
    findings = [
        SecurityFinding.create(
            issue="SQL Injection",
            reasoning="User input reaches SQL string construction.",
            mitigation="Use parameterized queries.",
            severity=Severity.CRITICAL,
            confidence=FindingConfidence.HIGH,
            file_path="src/db.py",
            code_snippet="cursor.execute(query)",
            line_start=10,
            line_end=10,
            cwe_id="CWE-89",
            tags=["sql"],
        ),
        SecurityFinding.create(
            issue="Hardcoded Secret",
            reasoning="A token is embedded in source.",
            mitigation="Load secret from environment.",
            severity=Severity.MEDIUM,
            confidence=FindingConfidence.MEDIUM,
            file_path="src/config.py",
            code_snippet="TOKEN='abc'",
            line_start=2,
            line_end=2,
            cwe_id="CWE-798",
            tags=["secret"],
        ),
        SecurityFinding.create(
            issue="Verbose Error",
            reasoning="Errors leak implementation details.",
            mitigation="Return generic errors to users.",
            severity=Severity.LOW,
            confidence=FindingConfidence.LOW,
            file_path="src/api.py",
            code_snippet="return str(exc)",
            line_start=7,
            line_end=8,
            tags=["error-handling"],
        ),
    ]
    for finding in findings:
        review.add_finding(finding)
    review.files_analyzed = 3
    review.complete()
    return review


def test_json_formatter_outputs_required_review_and_finding_fields():
    data = json.loads(JSONFormatter(pretty=True).format_review(sample_review()))

    assert data["tool"]["name"] == "FalconEYE"
    assert data["review"]["files_analyzed"] == 3
    assert data["summary"] == {"total_findings": 3, "critical": 1, "high": 0, "medium": 1, "low": 1}
    assert len(data["findings"]) == 3

    required = {"id", "issue", "severity", "confidence", "reasoning", "mitigation", "location", "code_snippet"}
    for finding in data["findings"]:
        assert required.issubset(finding)
        assert {"file_path", "line_start", "line_end"}.issubset(finding["location"])
        assert {"value", "level"}.issubset(finding["confidence"])


def test_json_formatter_single_finding_and_extension():
    finding = sample_review().findings[0]
    data = json.loads(JSONFormatter(pretty=False).format_finding(finding))

    assert data["issue"] == "SQL Injection"
    assert data["severity"] == "critical"
    assert JSONFormatter().get_file_extension() == ".json"


def test_sarif_formatter_outputs_valid_sarif_2_1_shape():
    sarif = json.loads(SARIFFormatter().format_review(sample_review()))

    assert sarif["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "FalconEYE"
    assert run["tool"]["driver"]["rules"]
    assert len(run["results"]) == 3

    first = run["results"][0]
    assert first["ruleId"] == "falconeye-critical"
    assert first["level"] == "error"
    assert first["message"]["text"] == "SQL Injection"
    region = first["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 10
    assert region["snippet"]["text"] == "cursor.execute(query)"


def test_sarif_formatter_single_finding_and_extension():
    result = json.loads(SARIFFormatter().format_finding(sample_review().findings[1]))

    assert result["ruleId"] == "falconeye-medium"
    assert result["level"] == "warning"
    assert SARIFFormatter().get_file_extension() == ".sarif"


def test_console_formatter_outputs_human_readable_report_without_color():
    output = ConsoleFormatter(use_color=False, verbose=True).format_review(sample_review())

    assert "FalconEYE Security Review" in output
    assert "Critical: 1" in output
    assert "Medium: 1" in output
    assert "Low: 1" in output
    assert "SQL Injection" in output
    assert "Hardcoded Secret" in output
    assert "cursor.execute(query)" in output
    assert ConsoleFormatter().get_file_extension() == ".txt"


def test_console_formatter_empty_review_reports_no_findings():
    review = SecurityReview.create("/tmp/empty", "python")
    review.complete()
    output = ConsoleFormatter(use_color=False).format_review(review)

    assert "No security issues found" in output
    assert "Total: 0 issues" in output


def test_html_formatter_outputs_report_structure_titles_and_severity_badges():
    html = HTMLFormatter().format_review(sample_review())

    assert html.startswith("<!DOCTYPE html>")
    assert "FalconEYE Security Report" in html
    assert "SQL Injection" in html
    assert "Hardcoded Secret" in html
    assert "Verbose Error" in html
    assert "critical" in html.lower()
    assert "medium" in html.lower()
    assert "low" in html.lower()
    assert "severity-badge" in html or "severity" in html
    assert "Use parameterized queries" in html
    assert HTMLFormatter().get_file_extension() == ".html"


def test_html_formatter_single_finding_contains_escaped_code_and_location():
    html = HTMLFormatter().format_finding(sample_review().findings[0])

    assert "SQL Injection" in html
    assert "src/db.py" in html
    assert "cursor.execute" in html
