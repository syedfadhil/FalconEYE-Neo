"""JSON formatter for machine-readable output."""

import json
from typing import Dict, Any
from ...domain.models.security import SecurityReview, SecurityFinding
from .base_formatter import OutputFormatter


class JSONFormatter(OutputFormatter):
    """
    Format security review results as JSON.

    Provides machine-readable output for programmatic consumption,
    CI/CD integration, and further processing.
    """

    def __init__(self, pretty: bool = True):
        """
        Initialize JSON formatter.

        Args:
            pretty: Enable pretty-printing with indentation
        """
        self.pretty = pretty

    def format_review(self, review: SecurityReview) -> str:
        """
        Format complete security review as JSON.

        Args:
            review: SecurityReview to format

        Returns:
            JSON string
        """
        data = self._review_to_dict(review)

        if self.pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)

    def format_finding(self, finding: SecurityFinding) -> str:
        """
        Format single security finding as JSON.

        Args:
            finding: SecurityFinding to format

        Returns:
            JSON string
        """
        data = self._finding_to_dict(finding)

        if self.pretty:
            return json.dumps(data, indent=2, default=str)
        else:
            return json.dumps(data, default=str)

    def get_file_extension(self) -> str:
        """Get file extension."""
        return ".json"

    def _review_to_dict(self, review: SecurityReview) -> Dict[str, Any]:
        """
        Convert SecurityReview to dictionary.

        Args:
            review: SecurityReview to convert

        Returns:
            Dictionary representation
        """
        return {
            "tool": {
                "name": "FalconEYE",
                "version": "2.0",
                "analysis_type": "AI-powered security analysis (NO pattern matching)",
            },
            "review": {
                "id": str(review.id),
                "codebase_path": review.codebase_path,
                "language": review.language,
                "languages": review.get_all_languages(),
                "started_at": review.started_at.isoformat() if review.started_at else None,
                "completed_at": review.completed_at.isoformat() if review.completed_at else None,
                "files_analyzed": review.files_analyzed,
            },
            "summary": {
                "total_findings": len(review.findings),
                "critical": review.get_critical_count(),
                "high": review.get_high_count(),
                "medium": review.get_medium_count(),
                "low": review.get_low_count(),
            },
            "findings": [self._finding_to_dict(f) for f in review.findings],
        }

    def _finding_to_dict(self, finding: SecurityFinding) -> Dict[str, Any]:
        """
        Convert SecurityFinding to dictionary.

        This is the definitive shape of a finding in the report JSON.
        It must include every meaningful field on SecurityFinding --
        historically this method dropped fields that weren't part of
        the original schema (e.g. cwe_id, tags), and it will drop new
        fields silently as they are added. Every field the model
        produces must be listed here or downstream consumers won't
        see it.

        Fields included:
        - Identity: id, issue, severity, confidence, reasoning, mitigation
        - Location: file_path / line_start / line_end (wrapped in
          `location` for report-consumer convenience)
        - Evidence: code_snippet, evidence_lines, cwe_id, tags
        - Reasoning provenance: reasoning_trace, assumed_context,
          why_not_false_positive
        - Automated grading: validation_warnings
        """
        out = {
            "id": str(finding.id),
            "issue": finding.issue,
            "severity": finding.severity.value,
            "confidence": {
                "value": finding.confidence.value,
                "level": finding.confidence.name.lower(),
            },
            "reasoning": finding.reasoning,
            "mitigation": finding.mitigation,
            "location": {
                "file_path": finding.file_path,
                "line_start": finding.line_start,
                "line_end": finding.line_end,
            },
            "code_snippet": finding.code_snippet,
        }

        # Extended fields. Use getattr with defaults so this still works
        # on older SecurityFinding objects (e.g. hydrated from a legacy
        # checkpoint) that don't have these attributes.
        cwe_id = getattr(finding, "cwe_id", None)
        if cwe_id:
            out["cwe_id"] = cwe_id

        tags = getattr(finding, "tags", None)
        if tags:
            out["tags"] = list(tags)

        # New reasoning provenance fields from the prompt-engineering
        # patch. Always emitted (even when empty) so downstream tooling
        # can rely on their presence and grade on non-emptiness.
        out["reasoning_trace"] = getattr(finding, "reasoning_trace", "") or ""
        out["evidence_lines"] = list(getattr(finding, "evidence_lines", []) or [])
        out["assumed_context"] = getattr(finding, "assumed_context", "") or ""
        out["why_not_false_positive"] = (
            getattr(finding, "why_not_false_positive", "") or ""
        )

        # Structured validation output. Empty list when the finding
        # passed all checks -- reviewers/automation can filter on
        # non-empty to surface findings that need extra scrutiny.
        out["validation_warnings"] = list(
            getattr(finding, "validation_warnings", []) or []
        )

        # Two-pass verifier output. Only emitted when the verifier ran
        # (verifier_verdict is not None). Downstream tooling can filter
        # on verifier_verdict=="confirmed" for high-confidence findings,
        # or verifier_verdict=="flag_uncertain" for human review.
        verifier_verdict = getattr(finding, "verifier_verdict", None)
        if verifier_verdict:
            out["verifier"] = {
                "verdict": verifier_verdict,
                "reasoning": getattr(finding, "verifier_reasoning", "") or "",
                "counterargument": getattr(finding, "verifier_counterargument", "") or "",
            }

        return out