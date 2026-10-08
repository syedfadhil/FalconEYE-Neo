"""Security-related domain models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

# Late import via TYPE_CHECKING is not needed here: the module
# structure has models importing each other freely.
from .confirmation import TaintConfirmation
from uuid import UUID, uuid4


class Severity(str, Enum):
    """Security finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingConfidence(str, Enum):
    """AI confidence level in the finding."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class SecurityFinding:
    """
    Immutable value object representing a security finding.

    IMPORTANT: This finding is generated purely by AI analysis,
    not by pattern matching or static analysis rules.
    """
    id: UUID
    issue: str
    reasoning: str
    mitigation: str
    severity: Severity
    confidence: FindingConfidence
    file_path: str
    code_snippet: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    cwe_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # --- Fields added by the prompt-engineering/validation patch. ---
    # All optional so this model remains backwards-compatible with
    # older checkpoints and older LLM prompts that don't produce them.
    #
    # evidence_lines: specific line numbers the model cites as proof
    #     of the vulnerability. Verified against the file so the model
    #     can't confabulate arbitrary numbers.
    # assumed_context: what the model explicitly says it's assuming
    #     about the caller / runtime / external state. Load-bearing
    #     assumptions are often where false positives hide.
    # why_not_false_positive: the model's own case for why the finding
    #     is real. Structured self-review at prompt time.
    # reasoning_trace: chain-of-thought reasoning the model wrote out
    #     BEFORE committing to the verdict. Present for auditability.
    # validation_warnings: attached post-hoc by the finding-validation
    #     pass. Human-readable descriptions of structural problems
    #     with the finding (line number out of range, snippet not
    #     found in file, unrecognized CVE reference, ...).
    evidence_lines: List[int] = field(default_factory=list)
    assumed_context: str = ""
    why_not_false_positive: str = ""
    reasoning_trace: str = ""
    validation_warnings: List[str] = field(default_factory=list)

    # --- Fields added by the two-pass verifier patch. ---
    # verifier_verdict: outcome of an adversarial second-pass review.
    #     One of: "confirmed", "downgrade", "false_positive",
    #     "flag_uncertain", or None (verifier did not run).
    # verifier_reasoning: the verifier's justification for the verdict.
    # verifier_counterargument: the strongest argument AGAINST the
    #     finding the verifier could construct. Present even for
    #     "confirmed" verdicts -- the verifier is asked to always
    #     attempt a counterargument, and "confirmed" means the
    #     counterargument was not credible.
    verifier_verdict: Optional[str] = None
    verifier_reasoning: str = ""
    verifier_counterargument: str = ""

    # --- Fields added by the review-confirmation patch. ---
    # taint_confirmation: reachability confirmation result attached by
    #     ReviewConfirmationService. When present, indicates whether
    #     the finding's enclosing method is reachable from a known
    #     entry point via call-graph analysis. See TaintConfirmation
    #     docstring for status semantics. None means confirmation was
    #     not run (older findings, non-confirmable finding, etc.).
    taint_confirmation: Optional[TaintConfirmation] = None

    @classmethod
    def create(
        cls,
        issue: str,
        reasoning: str,
        mitigation: str,
        severity: Severity,
        confidence: FindingConfidence,
        file_path: str,
        code_snippet: str,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        cwe_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        evidence_lines: Optional[List[int]] = None,
        assumed_context: str = "",
        why_not_false_positive: str = "",
        reasoning_trace: str = "",
        validation_warnings: Optional[List[str]] = None,
        verifier_verdict: Optional[str] = None,
        verifier_reasoning: str = "",
        verifier_counterargument: str = "",
        taint_confirmation: Optional[TaintConfirmation] = None,
    ) -> "SecurityFinding":
        """Factory method to create a security finding."""
        return cls(
            id=uuid4(),
            issue=issue,
            reasoning=reasoning,
            mitigation=mitigation,
            severity=severity,
            confidence=confidence,
            file_path=file_path,
            code_snippet=code_snippet,
            line_start=line_start,
            line_end=line_end,
            cwe_id=cwe_id,
            tags=tags or [],
            evidence_lines=evidence_lines or [],
            assumed_context=assumed_context,
            why_not_false_positive=why_not_false_positive,
            reasoning_trace=reasoning_trace,
            validation_warnings=validation_warnings or [],
            verifier_verdict=verifier_verdict,
            verifier_reasoning=verifier_reasoning,
            verifier_counterargument=verifier_counterargument,
            taint_confirmation=taint_confirmation,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "issue": self.issue,
            "reasoning": self.reasoning,
            "mitigation": self.mitigation,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "file_path": self.file_path,
            "code_snippet": self.code_snippet,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "cwe_id": self.cwe_id,
            "tags": self.tags,
            "evidence_lines": self.evidence_lines,
            "assumed_context": self.assumed_context,
            "why_not_false_positive": self.why_not_false_positive,
            "reasoning_trace": self.reasoning_trace,
            "validation_warnings": self.validation_warnings,
            "verifier_verdict": self.verifier_verdict,
            "verifier_reasoning": self.verifier_reasoning,
            "verifier_counterargument": self.verifier_counterargument,
            "taint_confirmation": (
                {
                    "status": self.taint_confirmation.status,
                    "enclosing_function": self.taint_confirmation.enclosing_function,
                    "reachable_entry_points": list(self.taint_confirmation.reachable_entry_points),
                    "shortest_chain": list(self.taint_confirmation.shortest_chain),
                    "hop_count": self.taint_confirmation.hop_count,
                    "reason": self.taint_confirmation.reason,
                }
                if self.taint_confirmation is not None else None
            ),
        }


@dataclass
class SecurityReview:
    """
    Aggregate root for a security review session.

    Contains all findings from an AI-powered security analysis.
    NO pattern matching involved - purely AI-driven.
    """
    id: UUID
    codebase_path: str
    language: str
    started_at: datetime
    findings: List[SecurityFinding] = field(default_factory=list)
    completed_at: Optional[datetime] = None
    files_analyzed: int = 0

    @classmethod
    def create(cls, codebase_path: str, language: str) -> "SecurityReview":
        """Factory method to create a new security review."""
        return cls(
            id=uuid4(),
            codebase_path=codebase_path,
            language=language,
            started_at=datetime.now(timezone.utc),
        )

    def add_finding(self, finding: SecurityFinding) -> None:
        """Add a security finding to the review."""
        self.findings.append(finding)

    def complete(self) -> None:
        """Mark the review as completed."""
        self.completed_at = datetime.now(timezone.utc)

    def get_findings_by_severity(self, severity: Severity) -> List[SecurityFinding]:
        """Get all findings of a specific severity."""
        return [f for f in self.findings if f.severity == severity]

    def get_critical_count(self) -> int:
        """Count critical findings."""
        return len(self.get_findings_by_severity(Severity.CRITICAL))

    def get_high_count(self) -> int:
        """Count high severity findings."""
        return len(self.get_findings_by_severity(Severity.HIGH))

    def get_medium_count(self) -> int:
        """Count medium severity findings."""
        return len(self.get_findings_by_severity(Severity.MEDIUM))

    def get_low_count(self) -> int:
        """Count low severity findings."""
        return len(self.get_findings_by_severity(Severity.LOW))

    def get_all_languages(self) -> List[str]:
        """
        Get all unique languages from analyzed files.
        
        Extracts languages from file paths in findings by checking file extensions.
        Returns list sorted by frequency (most common first).
        
        Returns:
            List of unique language names
        """
        from collections import Counter
        from pathlib import Path
        
        # Extension to language mapping (same as LanguageDetector)
        EXTENSION_TO_LANGUAGE = {
            ".c": "c", ".h": "c",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
            ".py": "python",
            ".rs": "rust",
            ".go": "go",
            ".php": "php",
            ".java": "java",
            ".dart": "dart",
            ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".rb": "ruby", ".rake": "ruby",
        }
        
        # Count languages from file paths in findings
        language_counts = Counter()
        for finding in self.findings:
            if finding.file_path:
                ext = Path(finding.file_path).suffix.lower()
                if ext in EXTENSION_TO_LANGUAGE:
                    language_counts[EXTENSION_TO_LANGUAGE[ext]] += 1
        
        # If no languages found from findings, return the primary language
        if not language_counts:
            return [self.language]
        
        # Sort by count (descending) and return language names
        sorted_languages = [lang for lang, _ in language_counts.most_common()]
        return sorted_languages

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "codebase_path": self.codebase_path,
            "language": self.language,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "files_analyzed": self.files_analyzed,
            "total_findings": len(self.findings),
            "critical": self.get_critical_count(),
            "high": self.get_high_count(),
            "medium": len(self.get_findings_by_severity(Severity.MEDIUM)),
            "low": len(self.get_findings_by_severity(Severity.LOW)),
            "findings": [f.to_dict() for f in self.findings],
        }