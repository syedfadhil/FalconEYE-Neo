"""
Two-pass finding verifier.

Motivation
----------
The primary security analysis pass is fundamentally biased *toward*
finding vulnerabilities:

- Its system prompt says "find security bugs"
- Its temperature is low but not zero, so it prefers positive
  interpretations of ambiguous evidence
- It's rewarded (by chain-of-thought self-review) for constructing
  the *positive case* for each finding

The two-pass verifier is a deliberately adversarial second call that
tries to argue AGAINST each finding. Same LLM, different framing.
Empirically this catches a substantial fraction of false positives
that survive primary analysis, because the model asked "why is this
NOT a real bug?" gets to notice sanitizers, unreachable paths, and
load-bearing assumptions the proposing pass overlooked.

Design notes
------------
- The verifier is a completely separate call, running on the same or
  a different LLM. See analysis.verifier.chat_base_url in config to
  point it at a different endpoint (useful for cross-model
  verification with e.g. a Qwen model verifying Gemma findings).
- Each finding is verified independently. The verifier does NOT see
  other findings from the same file, so it can't be biased by
  "well the model found N things, they can't all be false positives."
- The verifier's response schema is enforced by guided_json when
  supported, so we're guaranteed to get a verdict + reasoning from
  every call.
- Failure to reach the verifier LLM keeps the finding as-is with a
  warning tag -- verifier bugs never drop legitimate findings.
- Verifier verdicts and severity/confidence adjustments are applied
  to the finding via dataclass replacement. The original values are
  preserved in tags so the audit trail is intact.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace as _dc_replace
from typing import List, Optional

from ..models.security import SecurityFinding, Severity, FindingConfidence
from .llm_service import LLMService
from ...infrastructure.logging import FalconEyeLogger


# Verdict enum values used by the verifier LLM. Kept as a plain list
# (not an Enum) because they end up in a JSON schema and get compared
# to string values from responses.
VERDICT_CONFIRMED = "confirmed"
VERDICT_DOWNGRADE = "downgrade"
VERDICT_FALSE_POSITIVE = "false_positive"
VERDICT_FLAG_UNCERTAIN = "flag_uncertain"
_ALL_VERDICTS = [
    VERDICT_CONFIRMED,
    VERDICT_DOWNGRADE,
    VERDICT_FALSE_POSITIVE,
    VERDICT_FLAG_UNCERTAIN,
]


_VERIFIER_SYSTEM_PROMPT = """You are a security expert conducting an ADVERSARIAL review of a
vulnerability finding proposed by another analyst. Your goal is NOT
to confirm the finding, but to argue AGAINST it and identify reasons
it might be a false positive.

Read the code carefully. Then, for the finding you are shown, try to
find AT LEAST ONE credible reason it might be wrong:

- Is the vulnerable code path actually reachable with attacker input?
  Or does control flow prevent that input from reaching the sink?
- Is there a sanitizer, validator, or framework protection the
  original analyst missed?
- Are the assumptions in "assumed_context" actually true? Or are they
  load-bearing assumptions that would fail in practice?
- Does the "why_not_false_positive" argument actually address the
  real question, or does it dodge into general observations?
- Does the claimed CWE match what the code actually does, or is the
  taxonomy wrong?
- Would this finding trigger a real security incident, or is it a
  best-practice observation that has no exploit?

Then deliver a verdict:

- "confirmed": The finding is real. Your adversarial review could not
  find a credible counterargument. You STILL must provide a
  specific_counterargument -- the best case you could construct AGAINST
  the finding -- even though you rejected it.
- "downgrade": The finding might be real, but its severity or
  confidence was overstated. Provide revised_severity and/or
  revised_confidence.
- "false_positive": You have strong reasons to believe this is not a
  real vulnerability. Explain them in specific_counterargument.
- "flag_uncertain": The evidence is genuinely mixed. Not clearly a
  false positive, but you cannot confirm it as real. Human review
  needed.

Be honest. If the finding is clearly real, say "confirmed". If it's
clearly bogus, say "false_positive". Don't split the difference just
to appear thorough. If you're truly uncertain, "flag_uncertain" is
better than a wrong verdict either way.

Respond ONLY with a single JSON object matching this schema. Do NOT
include any text before or after the JSON. Do NOT use markdown fences.
"""


def _verifier_response_schema() -> dict:
    """
    JSON schema for the verifier's response. Enforced by guided_json.

    All string fields required so the verifier can't skip its own
    reasoning. Severity/confidence overrides are optional (verifier
    only fills them when verdict is downgrade or confirmed with a
    correction).
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": _ALL_VERDICTS,
            },
            "verifier_reasoning": {"type": "string"},
            "specific_counterargument": {"type": "string"},
            "revised_severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "info"],
            },
            "revised_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "verdict",
            "verifier_reasoning",
            "specific_counterargument",
        ],
        "additionalProperties": False,
    }


class FindingVerifier:
    """
    Adversarial second-pass verifier for security findings.

    Operates on findings that have already been through:
      1. Primary analysis (produced the finding)
      2. Structured validation (grounded line numbers, snippet, CWE)
      3. Enrichment (filled/adjusted missing fields, severity)
      4. Context enhancement (added line numbers to snippet)

    The verifier is the last gate before findings hit the report.
    """

    def __init__(
        self,
        llm_service: LLMService,
        context_lines: int = 10,
        concurrency: int = 4,
    ):
        """
        Args:
            llm_service: LLM service to call for verification. If you
                want to use a different model or endpoint for
                verification than for primary analysis, wire that in
                at the DI container level (a separate LLMService
                instance) and pass it here.
            context_lines: How many lines of code above/below the
                cited range to include in the verifier's prompt. Larger
                values give the verifier more context to notice missed
                sanitizers, but cost more tokens per call.
            concurrency: How many findings to verify in parallel.
                Bounded to avoid exhausting the LLM backend when a
                single file produces many findings.
        """
        self.llm_service = llm_service
        self.context_lines = max(0, context_lines)
        self.concurrency = max(1, concurrency)
        self.logger = FalconEyeLogger.get_instance()

    async def verify_findings(
        self,
        findings: List[SecurityFinding],
        file_content: str,
    ) -> List[SecurityFinding]:
        """
        Verify every finding, returning the list with false positives
        removed and downgrades/confirmations applied.

        Args:
            findings: List of findings to verify.
            file_content: Full source of the file the findings are
                about. Used to extract the context window around each
                finding's cited lines.

        Returns:
            Filtered/adjusted findings. Empty list is a valid result if
            every finding was rejected as a false positive.
        """
        if not findings:
            return findings

        self.logger.info(
            "Two-pass verifier starting",
            extra={"finding_count": len(findings)},
        )

        sem = asyncio.Semaphore(self.concurrency)

        async def _one(f: SecurityFinding) -> Optional[SecurityFinding]:
            async with sem:
                return await self._verify_one(f, file_content)

        results = await asyncio.gather(*(_one(f) for f in findings))

        # Filter out None (dropped false positives)
        out = [r for r in results if r is not None]

        # Log a compact summary so operators can see verifier impact
        # without wading through per-finding logs.
        verdict_counts: dict = {}
        for r in out:
            v = r.verifier_verdict or "unverified"
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        dropped = len(findings) - len(out)
        self.logger.info(
            "Two-pass verifier summary",
            extra={
                "input_findings": len(findings),
                "kept": len(out),
                "dropped_false_positives": dropped,
                "verdict_counts": verdict_counts,
            },
        )
        return out

    async def _verify_one(
        self,
        finding: SecurityFinding,
        file_content: str,
    ) -> Optional[SecurityFinding]:
        """
        Verify a single finding. Returns:
        - the finding with verifier fields populated (verdict !=
          false_positive), or
        - None if the verifier determined the finding is a false
          positive AND we should drop it.

        Verifier failures (LLM errors, malformed responses) leave the
        finding as-is with a warning tag; we never drop findings on
        verifier failure.
        """
        code_window = self._extract_context_window(finding, file_content)
        user_prompt = self._build_user_prompt(finding, code_window)

        try:
            raw_response = await self.llm_service.chat_with_schema(
                system_prompt=_VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=_verifier_response_schema(),
            )
        except Exception as e:
            self.logger.warning(
                "Verifier LLM call failed; keeping finding as-is",
                extra={
                    "finding_issue": finding.issue[:100],
                    "error_type": type(e).__name__,
                    "error": str(e)[:200],
                },
            )
            return _dc_replace(
                finding,
                tags=list(finding.tags) + ["verifier:call_failed"],
            )

        verdict_data = self._parse_verdict(raw_response)
        if verdict_data is None:
            self.logger.warning(
                "Verifier response could not be parsed; keeping finding as-is",
                extra={
                    "finding_issue": finding.issue[:100],
                    "raw_response_prefix": raw_response[:200],
                },
            )
            return _dc_replace(
                finding,
                tags=list(finding.tags) + ["verifier:parse_failed"],
            )

        return self._apply_verdict(finding, verdict_data)

    def _extract_context_window(
        self,
        finding: SecurityFinding,
        file_content: str,
    ) -> str:
        """
        Pull the finding's cited lines plus a ± context_lines window
        as a numbered-line string. Falls back to sending the whole
        file for very small files or when we can't determine a window.
        """
        lines = file_content.split("\n")
        n = len(lines)

        # Determine the range to show. Prefer evidence_lines (specific
        # cited lines), then line_start/line_end (bounding range),
        # then fall back to whole file.
        if finding.evidence_lines:
            lo = min(finding.evidence_lines)
            hi = max(finding.evidence_lines)
        elif finding.line_start is not None:
            lo = finding.line_start
            hi = finding.line_end or finding.line_start
        else:
            # No line info -- send the whole file (up to a limit) so
            # the verifier has something to work with.
            return self._format_lines(lines, 1, min(n, 200))

        window_start = max(1, lo - self.context_lines)
        window_end = min(n, hi + self.context_lines)
        return self._format_lines(lines, window_start, window_end)

    @staticmethod
    def _format_lines(lines: List[str], start: int, end: int) -> str:
        """Number lines for the verifier prompt. `start`/`end` are 1-based inclusive."""
        pad = len(str(end))
        out = []
        for i in range(start, end + 1):
            # lines list is 0-indexed
            line_text = lines[i - 1] if 0 <= (i - 1) < len(lines) else ""
            out.append(f"{str(i).rjust(pad)} | {line_text}")
        return "\n".join(out)

    def _build_user_prompt(
        self,
        finding: SecurityFinding,
        code_window: str,
    ) -> str:
        """
        Assemble the user turn for the verifier. Includes:
        - Finding metadata (issue, severity, CWE)
        - Finding's own justification (reasoning, why_not_false_positive)
        - Model's assumed_context (so verifier can attack those specifically)
        - The relevant code lines
        """
        cwe_line = (
            f"CWE: {finding.cwe_id}\n" if finding.cwe_id else ""
        )
        return (
            f"FINDING UNDER REVIEW\n"
            f"====================\n"
            f"Issue: {finding.issue}\n"
            f"Proposed severity: {finding.severity.value}\n"
            f"Proposed confidence: {finding.confidence.value}\n"
            f"{cwe_line}"
            f"File: {finding.file_path}\n"
            f"Cited lines: "
            f"{finding.line_start}-{finding.line_end or finding.line_start}\n"
            f"\n"
            f"REASONING (original analyst)\n"
            f"----------------------------\n"
            f"{finding.reasoning}\n"
            f"\n"
            f"REASONING TRACE (original analyst)\n"
            f"----------------------------------\n"
            f"{finding.reasoning_trace or '(none provided)'}\n"
            f"\n"
            f"ASSUMED CONTEXT (original analyst)\n"
            f"----------------------------------\n"
            f"{finding.assumed_context or '(none provided)'}\n"
            f"\n"
            f"WHY NOT A FALSE POSITIVE (original analyst)\n"
            f"-------------------------------------------\n"
            f"{finding.why_not_false_positive or '(none provided)'}\n"
            f"\n"
            f"PROPOSED MITIGATION\n"
            f"-------------------\n"
            f"{finding.mitigation}\n"
            f"\n"
            f"CODE UNDER REVIEW\n"
            f"=================\n"
            f"{code_window}\n"
            f"\n"
            f"Now conduct your adversarial review. Respond with a "
            f"JSON object as specified in the system prompt."
        )

    def _parse_verdict(self, raw: str) -> Optional[dict]:
        """
        Parse the verifier response into a dict.

        With guided_json enforcement, the response should be pure JSON.
        We still allow for some tolerance (leading/trailing whitespace,
        accidental markdown fences) because some backends behave
        differently when the schema mode is active.
        """
        if not raw:
            return None

        # Strip common markdown wrappers just in case.
        text = raw.strip()
        # Remove leading ```json / ``` and trailing ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

        # Try direct parse
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try extracting the first {...} block
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                return None

        if not isinstance(data, dict):
            return None
        # Minimal shape check: verdict must be one of the allowed values.
        if data.get("verdict") not in _ALL_VERDICTS:
            return None
        return data

    def _apply_verdict(
        self,
        finding: SecurityFinding,
        data: dict,
    ) -> Optional[SecurityFinding]:
        """
        Apply the verifier's verdict to the finding.

        Returns None if the verdict is "false_positive" -- caller
        removes the finding entirely.

        For "downgrade" and "confirmed" verdicts (which may still
        include revised severity/confidence), we return a new finding
        with the updated fields.
        """
        verdict = data["verdict"]
        verifier_reasoning = data.get("verifier_reasoning", "")
        counterargument = data.get("specific_counterargument", "")

        if verdict == VERDICT_FALSE_POSITIVE:
            # Dropped. Log at INFO so a reviewer can see what the
            # verifier rejected and why (useful for tuning the
            # verifier prompt).
            self.logger.info(
                "Verifier rejected finding as false positive",
                extra={
                    "finding_issue": finding.issue[:120],
                    "file_path": finding.file_path,
                    "counterargument": counterargument[:300],
                },
            )
            return None

        # Kept. Attach verifier fields.
        new_severity = finding.severity
        new_confidence = finding.confidence
        new_tags = list(finding.tags)

        if verdict == VERDICT_DOWNGRADE:
            rev_sev = data.get("revised_severity")
            if rev_sev:
                try:
                    new_severity = Severity(rev_sev)
                except ValueError:
                    pass
            rev_conf = data.get("revised_confidence")
            if rev_conf:
                try:
                    new_confidence = FindingConfidence(rev_conf)
                except ValueError:
                    pass
            # Preserve the original severity as a tag so the audit
            # trail is not lost.
            if new_severity != finding.severity:
                new_tags.append(f"verifier:downgraded_from_{finding.severity.value}")
            if new_confidence != finding.confidence:
                new_tags.append(
                    f"verifier:confidence_from_{finding.confidence.value}"
                )

        if verdict == VERDICT_FLAG_UNCERTAIN:
            new_tags.append("verifier:uncertain")

        return _dc_replace(
            finding,
            severity=new_severity,
            confidence=new_confidence,
            tags=new_tags,
            verifier_verdict=verdict,
            verifier_reasoning=verifier_reasoning,
            verifier_counterargument=counterargument,
        )
