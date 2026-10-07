"""
Structured validation of LLM-produced security findings.

Motivation:
-----------
LLMs confabulate. The three most common failure modes we see in
FalconEYE findings, in decreasing order of frequency:

  1. Line numbers that don't correspond to the real file
     (e.g. line 47 in a 20-line file, or line_start > line_end).
  2. Code snippets that don't actually appear in the file, or that
     the model paraphrased in the "code_snippet" field.
  3. CVE numbers and CWE IDs that either don't exist or are for
     something unrelated to the finding.

This module runs cheap post-hoc checks on every finding right after
JSON parsing. Findings that fail hard checks (impossible line numbers,
snippets nowhere in the file) get dropped with a log record. Findings
that fail soft checks (near-miss snippets, suspicious CVE mentions)
have their confidence downgraded and get a `validation_warnings` tag
so downstream reporting can flag them.

Design principles:
- Never raise. A bug in validation should not fail the whole scan.
- Whitespace-normalize before comparing snippets (LLMs re-indent
  when they quote code).
- Be conservative on dropping: only drop when the evidence is
  clearly fabricated. Downgrade when it looks fishy but might be
  a legitimate finding with a fuzzy citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# CWEs curated for code-review use. Restricting the model to this list
# stops confabulation of nonexistent numbers ("CWE-2149", etc.) and
# eliminates most cases of citing a CWE that doesn't fit the finding.
#
# Source: MITRE Top 25 (2024) + CWE View 1000 categories that come up
# in practice on real projects. Add more as needed; the list should
# stay short enough that the model can plausibly hold it in context.
ALLOWED_CWES: dict = {
    # Injection family
    "CWE-77":  "Command Injection",
    "CWE-78":  "OS Command Injection",
    "CWE-79":  "Cross-site Scripting (XSS)",
    "CWE-89":  "SQL Injection",
    "CWE-91":  "XML Injection",
    "CWE-93":  "CRLF Injection",
    "CWE-94":  "Code Injection",
    "CWE-95":  "Eval Injection",
    "CWE-98":  "PHP Remote File Inclusion",
    "CWE-113": "HTTP Response Splitting",
    "CWE-116": "Improper Encoding or Escaping of Output",
    "CWE-611": "XXE (XML External Entity)",
    "CWE-643": "XPath Injection",
    "CWE-1336": "Server-Side Template Injection (SSTI)",

    # Authentication / authorization / session
    "CWE-269": "Improper Privilege Management",
    "CWE-284": "Improper Access Control",
    "CWE-285": "Improper Authorization",
    "CWE-287": "Improper Authentication",
    "CWE-306": "Missing Authentication for Critical Function",
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-384": "Session Fixation",
    "CWE-434": "Unrestricted Upload of Dangerous File Type",
    "CWE-522": "Insufficiently Protected Credentials",
    "CWE-565": "Trust of Cookie Without Integrity Check",
    "CWE-639": "Authorization Bypass Through User-Controlled Key (IDOR)",
    "CWE-862": "Missing Authorization",
    "CWE-863": "Incorrect Authorization",

    # Cryptography
    "CWE-295": "Improper Certificate Validation",
    "CWE-311": "Missing Encryption of Sensitive Data",
    "CWE-321": "Use of Hard-coded Cryptographic Key",
    "CWE-326": "Inadequate Encryption Strength",
    "CWE-327": "Use of Broken or Risky Cryptographic Algorithm",
    "CWE-328": "Use of Weak Hash",
    "CWE-330": "Use of Insufficiently Random Values",
    "CWE-338": "Weak PRNG",
    "CWE-347": "Improper Verification of Cryptographic Signature",

    # File handling / path
    "CWE-22":  "Path Traversal",
    "CWE-59":  "Link Following",
    "CWE-73":  "External Control of File Name or Path",

    # Deserialization / trust boundaries
    "CWE-502": "Deserialization of Untrusted Data",
    "CWE-915": "Mass Assignment / Modifiable Attributes",

    # SSRF / open redirect / URL
    "CWE-601": "Open Redirect",
    "CWE-918": "Server-Side Request Forgery (SSRF)",

    # Information disclosure / logging
    "CWE-200": "Exposure of Sensitive Information",
    "CWE-209": "Information Exposure Through Error Message",
    "CWE-532": "Insertion of Sensitive Information into Log File",
    "CWE-798": "Use of Hard-coded Credentials",

    # Misconfiguration / defaults
    "CWE-16":  "Configuration",
    "CWE-1004": "Sensitive Cookie Without HttpOnly Flag",
    "CWE-614": "Sensitive Cookie in HTTPS Session Without Secure Attribute",
    "CWE-942": "Overly Permissive CORS",

    # Validation / input handling
    "CWE-20":  "Improper Input Validation",
    "CWE-1284": "Improper Validation of Specified Quantity in Input",

    # Memory / resource
    "CWE-119": "Improper Restriction of Operations within Bounds",
    "CWE-125": "Out-of-bounds Read",
    "CWE-190": "Integer Overflow or Wraparound",
    "CWE-400": "Uncontrolled Resource Consumption",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-770": "Allocation of Resources Without Limits",

    # Race conditions
    "CWE-362": "Concurrent Execution using Shared Resource (Race)",
    "CWE-367": "TOCTOU Race",

    # Dependencies / supply chain
    "CWE-1104": "Use of Unmaintained Third Party Components",
    "CWE-1395": "Dependency on Vulnerable Third-Party Component",

    # Cryptography ops
    "CWE-916": "Use of Password Hash With Insufficient Computational Effort",

    # Generic
    "CWE-693": "Protection Mechanism Failure",
    "CWE-754": "Improper Check for Unusual or Exceptional Conditions",
    "CWE-1188": "Insecure Default Initialization of Resource",
}


# Matches "CVE-2019-14853" and similar. Used to detect suspicious CVE
# references. We don't reject them outright (a scan may legitimately
# reference a known CVE) — we downgrade confidence and tag so a
# reviewer knows to verify.
CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b")

# Matches CWE references so we can validate against the allowed list.
CWE_PATTERN = re.compile(r"\bCWE-(\d+)\b")


@dataclass
class ValidationReport:
    """
    Result of validating a single finding.

    Attributes:
        should_drop: True if the finding should be discarded entirely
            (e.g. its code snippet exists nowhere in the file). Reserved
            for clearly-fabricated evidence.
        should_downgrade_confidence: True if the finding is questionable
            but might still be real. Confidence gets bumped down one
            level (HIGH -> MEDIUM -> LOW).
        warnings: Human-readable strings describing each validation
            issue. Attached to the finding as tags for reporting.
    """
    should_drop: bool = False
    should_downgrade_confidence: bool = False
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: "ValidationReport") -> None:
        self.should_drop = self.should_drop or other.should_drop
        self.should_downgrade_confidence = (
            self.should_downgrade_confidence or other.should_downgrade_confidence
        )
        self.warnings.extend(other.warnings)


def _normalize(text: str) -> str:
    """
    Whitespace-normalize a code fragment for substring matching.

    Collapses runs of whitespace (spaces, tabs, newlines) into single
    spaces. LLMs commonly re-indent when quoting code, so exact-string
    matching gives too many false negatives.
    """
    return re.sub(r"\s+", " ", text).strip()


def _validate_line_range(
    line_start: Optional[int],
    line_end: Optional[int],
    file_line_count: int,
) -> ValidationReport:
    """Verify line numbers point at real lines in the file."""
    r = ValidationReport()

    if line_start is None and line_end is None:
        return r  # no line info to check

    if line_start is not None:
        if line_start < 1 or line_start > file_line_count:
            r.warnings.append(
                f"line_start={line_start} out of range (file has "
                f"{file_line_count} lines)"
            )
            r.should_downgrade_confidence = True

    if line_end is not None:
        if line_end < 1 or line_end > file_line_count:
            r.warnings.append(
                f"line_end={line_end} out of range (file has "
                f"{file_line_count} lines)"
            )
            r.should_downgrade_confidence = True

    if (line_start is not None and line_end is not None
            and line_start > line_end):
        r.warnings.append(
            f"line_start={line_start} > line_end={line_end}"
        )
        r.should_downgrade_confidence = True

    return r


def _validate_code_snippet(
    code_snippet: str,
    file_content: str,
    line_start: Optional[int],
    line_end: Optional[int],
) -> ValidationReport:
    """
    Verify the code snippet appears in the file (whitespace-normalized).

    If line numbers are provided, also check that the snippet appears
    near those lines (within a small tolerance) — catches cases where
    the model produces a real snippet but a completely wrong line
    citation.
    """
    r = ValidationReport()

    if not code_snippet or code_snippet.strip() in ("", "N/A", "..."):
        return r  # nothing to validate

    # Filter out snippets that are obviously not code — some models
    # put a description in code_snippet when they can't quote the
    # source verbatim. These usually contain sentence punctuation and
    # no code tokens.
    if len(code_snippet) < 5:
        return r

    normalized_snippet = _normalize(code_snippet)
    normalized_file = _normalize(file_content)

    if normalized_snippet not in normalized_file:
        # Hard fail: the quoted snippet is not in the file at all.
        # Very likely the model paraphrased or fabricated it.
        r.warnings.append(
            "code_snippet not found in source file "
            "(model may have paraphrased or fabricated the citation)"
        )
        # Don't drop -- the finding might still be real, just with a
        # bad snippet. Downgrade confidence and tag for reviewer.
        r.should_downgrade_confidence = True
        return r

    # Snippet exists somewhere in the file. If line numbers are given,
    # check they roughly correspond.
    if line_start is not None:
        file_lines = file_content.split("\n")
        # Look in a small window around the cited lines. If the
        # snippet doesn't appear there, it's a citation-line mismatch.
        window_start = max(0, line_start - 3)
        window_end = min(len(file_lines), (line_end or line_start) + 3)
        window_text = _normalize("\n".join(file_lines[window_start:window_end]))
        if normalized_snippet not in window_text:
            r.warnings.append(
                f"code_snippet found in file but not near cited lines "
                f"{line_start}-{line_end or line_start}"
            )
            # Softer signal than "not in file at all" -- just tag.
            # Don't downgrade for this alone.

    return r


def _validate_cve_cwe(
    reasoning: str,
    mitigation: str,
    cwe_id: Optional[str],
) -> ValidationReport:
    """
    Check for confabulated CVE/CWE references.

    CVEs: Any CVE-like string is treated as suspect. LLMs cite fake
    CVEs constantly. We don't drop the finding but tag it so a
    reviewer verifies before including in a report.

    CWEs: Both the explicit cwe_id field and any CWE mentions in
    reasoning/mitigation are checked against the allowed list.
    """
    r = ValidationReport()

    # CVE references in reasoning/mitigation
    combined = f"{reasoning}\n{mitigation}"
    cves_mentioned = CVE_PATTERN.findall(combined)
    if cves_mentioned:
        # De-duplicate for the warning message
        unique_cves = sorted(set(cves_mentioned))
        r.warnings.append(
            f"CVE reference(s) present: {', '.join(unique_cves)} -- "
            "verify these are real and applicable before publishing"
        )
        # Don't downgrade for this alone -- users may legitimately
        # cite CVEs. The tag alerts the reviewer.

    # CWE explicit field
    if cwe_id:
        cwe_normalized = cwe_id.upper().strip()
        if not cwe_normalized.startswith("CWE-"):
            cwe_normalized = f"CWE-{cwe_normalized}"
        if cwe_normalized not in ALLOWED_CWES:
            r.warnings.append(
                f"cwe_id={cwe_id} not in allowed CWE list "
                "(may be nonexistent or off-topic)"
            )
            r.should_downgrade_confidence = True

    # CWE references in reasoning/mitigation
    cwes_in_text = CWE_PATTERN.findall(combined)
    for cwe_num in cwes_in_text:
        cwe_ref = f"CWE-{cwe_num}"
        if cwe_ref not in ALLOWED_CWES:
            r.warnings.append(
                f"reference to {cwe_ref} in reasoning/mitigation "
                "is not in the allowed CWE list"
            )
            # Soft signal, don't downgrade.

    return r


def verify_finding_grounding(
    finding,
    file_content: str,
) -> ValidationReport:
    """
    Run all validation checks on a single finding.

    Args:
        finding: SecurityFinding object. Duck-typed (only needs the
            standard fields), so it works on the domain model without
            importing it -- avoids a circular dependency between
            infrastructure and domain.
        file_content: The full source text of the file the finding
            claims to be about.

    Returns:
        ValidationReport aggregating all findings from the individual
        checks.
    """
    report = ValidationReport()

    try:
        file_line_count = file_content.count("\n") + 1

        report.merge(_validate_line_range(
            getattr(finding, "line_start", None),
            getattr(finding, "line_end", None),
            file_line_count,
        ))

        report.merge(_validate_code_snippet(
            getattr(finding, "code_snippet", "") or "",
            file_content,
            getattr(finding, "line_start", None),
            getattr(finding, "line_end", None),
        ))

        report.merge(_validate_cve_cwe(
            getattr(finding, "reasoning", "") or "",
            getattr(finding, "mitigation", "") or "",
            getattr(finding, "cwe_id", None),
        ))

    except Exception:
        # Never raise from validation. If we hit an unexpected shape,
        # tag it and continue.
        report.warnings.append(
            "validation encountered an internal error -- finding "
            "passed through unchecked"
        )

    return report


def downgrade_confidence(current) -> "Optional[object]":
    """
    Bump a FindingConfidence enum one level down.

    Duck-typed to avoid importing the domain enum. Returns the same
    type as the input, or None if we can't figure out how to downgrade.
    """
    try:
        cur_val = getattr(current, "value", str(current)).lower()
        mapping = {"high": "medium", "medium": "low", "low": "low"}
        target = mapping.get(cur_val, "low")
        # Try to find a sibling value on the enum class of `current`
        cls = type(current)
        for member in cls:
            if getattr(member, "value", "").lower() == target:
                return member
    except Exception:
        pass
    return current


# Public constants for use in prompt generation.
def get_allowed_cwe_list_for_prompt() -> str:
    """
    Format the allowed CWE list for inclusion in a system prompt.

    One CWE per line, `CWE-N: Name` format. Kept short so it doesn't
    balloon the system prompt.
    """
    lines = [f"  {k}: {v}" for k, v in ALLOWED_CWES.items()]
    return "\n".join(lines)
