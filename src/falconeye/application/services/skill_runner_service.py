"""
Skill runner service (Phase 1b of the skills architecture).

Takes a :class:`SkillDefinition`, runs its Step 1 grep discovery against
a workspace, batches the hits by file, invokes the LLM once per file
using the skill's body as the system prompt, parses the model's
findings (JSON matching the shared schema), converts each into a
:class:`SecurityFinding`, and returns a populated
:class:`SecurityReview`.

Design notes
------------

Skills without discovery patterns are rejected here with a clear
error. Whole-workspace LLM-only skills (input-validation-style) need
a different execution model -- they either iterate every relevant
file (expensive) or ask the LLM to select files (fragile). Phase 1c
will add that mode; this patch focuses on the discovery-driven path,
which covers the two heaviest skills in the operator's library
(injection-prevention, deserialization-xxe).

Batching is by-file, not by-hit. Sending each hit separately would
multiply prompt tokens (each call would re-include the skill body).
One call per file with all its hits reuses the skill body once and
gives the LLM enough context to reason about the file as a whole
rather than one line at a time.

Failure isolation: an LLM error on file N does not stop processing
of file N+1. Each file's error is recorded and shipped as a
zero-severity note in the final report so the operator knows what
didn't get audited. This matches the failure-tolerance philosophy of
the confirmation service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ...domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    SecurityReview,
    Severity,
)
from ...domain.models.skill import DiscoveryHit, SkillDefinition
from ...domain.services.llm_service import LLMService
from ...infrastructure.skills.grep_runner import GrepRunner


_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------


@dataclass(frozen=True)
class FileEvaluationOutcome:
    """Per-file evaluation record for diagnostics.

    Not shipped to the report itself (that stays in SecurityFinding
    shape) but returned from ``run()`` so the CLI can print a summary
    of what worked and what didn't.
    """
    file_path: str
    discovery_hit_count: int
    findings_emitted: int
    error: Optional[str] = None


@dataclass
class SkillRunSummary:
    """Aggregate diagnostics from a skill run.

    Populated as the runner works; the CLI renders this before the
    per-finding output so the operator can see coverage at a glance.
    """
    skill_name: str
    total_discovery_hits: int = 0
    files_evaluated: int = 0
    files_with_llm_errors: int = 0
    total_findings: int = 0
    outcomes: List[FileEvaluationOutcome] = field(default_factory=list)


# --------------------------------------------------------------------
# Batch-run aggregation models
# --------------------------------------------------------------------


#: Batch outcome status constants. Kept as strings (not an Enum) so
#: they serialize cleanly to JSON and templates without extra work.
BATCH_STATUS_RAN = "ran"
BATCH_STATUS_SKIPPED_NOT_SCAN = "skipped_not_scan"
BATCH_STATUS_SKIPPED_NO_PATTERNS = "skipped_no_patterns"
BATCH_STATUS_ERROR = "error"


@dataclass(frozen=True)
class SkillBatchOutcome:
    """Per-skill result in a ``run_all`` batch.

    Populated for every skill in the batch regardless of whether it
    executed. ``status`` distinguishes the four cases; the CLI uses
    it to color-code the summary and print skip reasons.

    ``summary`` is populated only when the skill actually ran (status
    == BATCH_STATUS_RAN). For skipped or errored skills it is None.
    """
    skill_name: str
    status: str
    reason: str = ""
    summary: "Optional[SkillRunSummary]" = None


@dataclass
class SkillBatchSummary:
    """Aggregate counts across a batch run.

    ``total_findings`` counts pre-deduplication; ``findings_kept`` is
    the count after dedup (may equal total_findings when dedup is off
    or when no duplicates were found).
    """
    total_skills_seen: int = 0
    skills_ran: int = 0
    skills_skipped: int = 0
    skills_errored: int = 0
    total_findings: int = 0
    findings_kept: int = 0
    duplicates_dropped: int = 0
    outcomes: List[SkillBatchOutcome] = field(default_factory=list)


# --------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------


#: System-prompt suffix appended to the skill's body. Forces JSON
#: output regardless of the YAML shape the skill's markdown shows --
#: many LLM backends handle JSON schema enforcement but not YAML.
#: The Python side translates the JSON back into SecurityFinding
#: shape so operators reading skill markdown see the familiar YAML.
_JSON_OUTPUT_INSTRUCTION = """

---

## Output Requirements

You MUST respond with a single JSON object matching this schema:

```
{
  "findings": [
    {
      "issue":          "short one-line summary of the vulnerability",
      "severity":       "Critical" | "High" | "Medium" | "Low" | "Info",
      "confidence":     "High" | "Medium" | "Low",
      "file":           "relative or absolute path to the file",
      "line":           <integer, 1-based, or null>,
      "description":    "detailed explanation of the issue",
      "recommended_fix": "specific remediation guidance",
      "cwe":            "CWE-XXX or null",
      "category":       "injection | secrets | auth | crypto | ... or null",
      "evidence_lines": [<integer>, ...] // lines that support the finding
    }
  ]
}
```

If NO findings are present in the code shown to you, respond with
`{"findings": []}`.

Do not include Markdown fences, explanations, or any text outside
the JSON object. The JSON must be valid and parseable.
"""


#: JSON schema for backends that support structured output. Kept
#: permissive enough that a model that's slightly off-schema (extra
#: fields, missing optionals) doesn't fail the whole call.
def _skill_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "severity": {"type": "string"},
                        "confidence": {"type": "string"},
                        "file": {"type": "string"},
                        "line": {"type": ["integer", "null"]},
                        "description": {"type": "string"},
                        "recommended_fix": {"type": "string"},
                        "cwe": {"type": ["string", "null"]},
                        "category": {"type": ["string", "null"]},
                        "evidence_lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["issue", "severity", "confidence", "description"],
                },
            },
        },
        "required": ["findings"],
    }


def _build_user_prompt(
    file_path: str,
    file_content: str,
    hits: List[DiscoveryHit],
) -> str:
    """Assemble the per-file user prompt.

    Includes the file with line numbers (so the model can cite lines
    accurately) and a compact listing of the discovery hits that
    flagged this file, so the model knows which specific spots the
    grep pass considered suspicious.
    """
    numbered = _number_lines(file_content)
    hit_lines = "\n".join(
        f"  - line {h.line_number}: {h.line_text.strip()[:120]}"
        f"{'  [' + h.pattern_description + ']' if h.pattern_description else ''}"
        for h in hits
    )
    return (
        f"## File\n\n`{file_path}`\n\n"
        f"## Discovery hits\n\n{hit_lines or '  (none)'}\n\n"
        f"## Source (numbered)\n\n```\n{numbered}\n```\n"
    )


def _number_lines(content: str) -> str:
    """Prefix each source line with its 1-based line number.

    Limited to source shorter than ~4000 lines; longer files get
    truncated with a marker so oversized inputs don't blow the
    context window. In practice code files > 4000 lines are rare and
    usually generated; skipping is safer than truncating mid-block.
    """
    max_lines = 4000
    lines = content.splitlines()
    if len(lines) > max_lines:
        head = lines[:max_lines]
        return "\n".join(f"{i + 1: >5}  {l}" for i, l in enumerate(head)) + (
            f"\n... ({len(lines) - max_lines} more lines omitted)"
        )
    return "\n".join(f"{i + 1: >5}  {l}" for i, l in enumerate(lines))


# --------------------------------------------------------------------
# LLM response parsing
# --------------------------------------------------------------------


def _parse_llm_findings(raw_response: str) -> Optional[List[dict]]:
    """Best-effort JSON extraction from the LLM response.

    Returns the list of finding dicts, or None if the response could
    not be interpreted. Common tolerated variations:

      * The model wraps its output in ```json fences (strip them).
      * The model prefaces its JSON with a natural-language sentence
        (find the first ``{`` and try from there).
      * The model returns a bare array instead of ``{"findings": [...]}``.
    """
    text = raw_response.strip()
    if not text:
        return None

    # Strip fenced-code blocks
    if text.startswith("```"):
        # Drop first line (the ```json or ```) and last ``` fence
        parts = text.split("```")
        # Whichever middle segment is the biggest is usually the payload
        candidates = [p.strip() for p in parts if p.strip()]
        # Take the one that starts with { or [
        for cand in candidates:
            head = cand.lstrip()
            # Some models emit ``json\n{...}`` inside the fence
            if head.startswith("json"):
                head = head[4:].lstrip()
            if head.startswith("{") or head.startswith("["):
                text = head
                break

    # Try direct parse
    parsed = _try_parse_json(text)
    if parsed is None:
        # Try to slice from first { to last }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            parsed = _try_parse_json(text[first:last + 1])
    if parsed is None:
        # Try bare array
        first = text.find("[")
        last = text.rfind("]")
        if first != -1 and last > first:
            arr = _try_parse_json(text[first:last + 1])
            if isinstance(arr, list):
                parsed = {"findings": arr}

    if parsed is None:
        return None
    if isinstance(parsed, dict):
        findings = parsed.get("findings")
        if isinstance(findings, list):
            return findings
    if isinstance(parsed, list):
        return parsed
    return None


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# --------------------------------------------------------------------
# Finding conversion
# --------------------------------------------------------------------


def _finding_from_llm_dict(
    d: dict,
    *,
    file_path_from_context: str,
    skill_name: str,
) -> Optional[SecurityFinding]:
    """Turn one entry from the LLM's ``findings`` array into a
    :class:`SecurityFinding`.

    Best-effort field mapping with defaults for anything the model
    might drop. Returns None if the entry is unusable (e.g. no issue
    text) so the caller can skip it without crashing the whole batch.

    ``file_path_from_context`` is used when the model returns a
    relative path or omits the file field entirely. The evaluation
    is per-file, so we already know which file this was reasoning
    about.
    """
    issue = (d.get("issue") or "").strip()
    if not issue:
        return None

    severity = _severity_from_llm(d.get("severity"))
    confidence = _confidence_from_llm(d.get("confidence"))

    file_path = (d.get("file") or "").strip() or file_path_from_context
    line_raw = d.get("line")
    line_start = int(line_raw) if isinstance(line_raw, int) else None

    description = (d.get("description") or "").strip()
    fix = (d.get("recommended_fix") or "").strip()
    cwe = d.get("cwe")
    if cwe in ("", "null", "None"):
        cwe = None
    category = d.get("category")

    tags = [f"skill:{skill_name}"]
    if category:
        tags.append(f"category:{category}")

    evidence_raw = d.get("evidence_lines") or []
    if isinstance(evidence_raw, list):
        evidence_lines = [int(e) for e in evidence_raw if isinstance(e, int)]
    else:
        evidence_lines = []

    return SecurityFinding(
        id=uuid4(),
        issue=issue,
        reasoning=description,
        mitigation=fix,
        severity=severity,
        confidence=confidence,
        file_path=file_path,
        code_snippet="",  # LLM saw the code but we don't ship it inline here
        line_start=line_start,
        line_end=None,
        cwe_id=cwe,
        tags=tags,
        evidence_lines=evidence_lines,
    )


def _severity_from_llm(value) -> Severity:
    """Map an LLM severity string to the Severity enum.

    Skills' schema uses TitleCase ("Critical", "High"); Severity's
    enum uses lowercase values. Also tolerate variants ("info" -> low
    since Severity doesn't have Info; log the coercion so we don't
    silently lose information).
    """
    if not value:
        return Severity.MEDIUM
    s = str(value).strip().lower()
    for candidate in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        if candidate.value.lower() == s:
            return candidate
    if s in ("info", "informational", "note"):
        _logger.debug("Coercing severity=%s to LOW", value)
        return Severity.LOW
    _logger.debug("Unknown severity=%s; defaulting to MEDIUM", value)
    return Severity.MEDIUM


def _confidence_from_llm(value) -> FindingConfidence:
    if not value:
        return FindingConfidence.MEDIUM
    s = str(value).strip().lower()
    for candidate in (FindingConfidence.HIGH, FindingConfidence.MEDIUM, FindingConfidence.LOW):
        if candidate.value.lower() == s:
            return candidate
    return FindingConfidence.MEDIUM


# --------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------


class SkillRunnerService:
    """Runs one skill against a workspace and produces a SecurityReview.

    Constructed via the DI container so it picks up the same
    ``LLMService`` the rest of the pipeline uses. Stateless across
    runs; a single instance can be reused for different skills and
    workspaces.
    """

    def __init__(self, *, llm_service: LLMService) -> None:
        self._llm = llm_service

    async def run(
        self,
        *,
        skill: SkillDefinition,
        workspace: Path,
        max_hits_per_pattern: int = GrepRunner.DEFAULT_MAX_HITS_PER_PATTERN,
        max_total_hits: int = GrepRunner.DEFAULT_MAX_TOTAL_HITS,
    ) -> "tuple[SecurityReview, SkillRunSummary]":
        """Execute the skill and return findings + a diagnostic summary.

        Raises :class:`ValueError` when the skill has no discovery
        patterns (LLM-only skills are not supported in Phase 1b; see
        the module docstring). All other errors during evaluation are
        caught per-file and recorded in the summary rather than
        propagated.
        """
        if not skill.is_runnable_today:
            raise ValueError(
                f"Skill '{skill.name}' has function='{skill.metadata.function}'; "
                "only function=scan skills can be executed via `skill run`. "
                "verify skills route to fp-check (Phase 3); triage skills "
                "route to the triage stage (Phase 4)."
            )
        if not skill.discovery_patterns:
            raise ValueError(
                f"Skill '{skill.name}' has no Step 1 discovery patterns. "
                "Whole-workspace LLM-only execution is deferred to "
                "Phase 1c. Add at least one `grep -rn -E \"...\" "
                "--include=\"*.<ext>\" .` command inside a "
                "```bash``` block under Step 1."
            )

        # ---------- Discovery ----------
        grep = GrepRunner(
            workspace_root=workspace,
            max_hits_per_pattern=max_hits_per_pattern,
            max_total_hits=max_total_hits,
        )
        hits = grep.discover(skill.discovery_patterns)

        summary = SkillRunSummary(
            skill_name=skill.name,
            total_discovery_hits=len(hits),
        )

        # Group by file
        hits_by_file: Dict[str, List[DiscoveryHit]] = {}
        for h in hits:
            hits_by_file.setdefault(h.file_path, []).append(h)

        # ---------- Per-file LLM evaluation ----------
        review = SecurityReview.create(
            codebase_path=str(workspace),
            language=(skill.metadata.category or "unknown"),
        )
        for file_path, file_hits in sorted(hits_by_file.items()):
            outcome = await self._evaluate_file(
                skill=skill,
                file_path=file_path,
                hits=file_hits,
                review=review,
            )
            summary.outcomes.append(outcome)
            summary.files_evaluated += 1
            if outcome.error is not None:
                summary.files_with_llm_errors += 1
            summary.total_findings += outcome.findings_emitted

        review.complete()
        return review, summary

    async def run_all(
        self,
        *,
        skills: "List[SkillDefinition]",
        workspace: Path,
        dedupe: bool = True,
    ) -> "tuple[SecurityReview, SkillBatchSummary]":
        """Run every runnable skill in ``skills`` against ``workspace``.

        Skills that aren't runnable (function != scan, or no Step 1
        discovery patterns) are recorded in the batch summary as
        skipped rather than failing the whole batch. Skills that
        raise unexpectedly during execution are recorded as errored.
        Both cases keep the batch flowing so a broken skill doesn't
        block the operator from seeing what everything else found.

        Findings from every skill land in ONE :class:`SecurityReview`
        with the per-skill ``skill:<name>`` tag from the single-skill
        runner preserved. When ``dedupe`` is True, findings with the
        same (normalized_file_path, line_start, category) are
        collapsed: the highest-severity survivor wins, tiebreaking
        first by highest confidence and second by first-seen order.
        This handles overlaps between related skills (e.g. injection-
        prevention and input-validation both flagging the same call
        site) without dropping information -- the survivor accumulates
        the dropped tags so provenance is visible in the report.
        """
        review = SecurityReview.create(
            codebase_path=str(workspace),
            language="multi",  # batch spans multiple skill categories
        )
        batch = SkillBatchSummary(total_skills_seen=len(skills))

        for skill in skills:
            # Skip non-scan skills up front with a clear status
            if not skill.is_runnable_today:
                batch.outcomes.append(SkillBatchOutcome(
                    skill_name=skill.name,
                    status=BATCH_STATUS_SKIPPED_NOT_SCAN,
                    reason=(
                        f"function={skill.metadata.function or 'unknown'} is "
                        "not executable via run_all; routes to fp-check "
                        "(Phase 3) or triage (Phase 4)."
                    ),
                ))
                batch.skills_skipped += 1
                continue

            if not skill.discovery_patterns:
                batch.outcomes.append(SkillBatchOutcome(
                    skill_name=skill.name,
                    status=BATCH_STATUS_SKIPPED_NO_PATTERNS,
                    reason=(
                        "no Step 1 discovery patterns; whole-workspace "
                        "LLM-only execution is Phase 1c work."
                    ),
                ))
                batch.skills_skipped += 1
                continue

            try:
                sub_review, sub_summary = await self.run(
                    skill=skill, workspace=workspace,
                )
            except Exception as e:
                _logger.warning(
                    "Skill %s raised during batch run: %s", skill.name, e,
                )
                batch.outcomes.append(SkillBatchOutcome(
                    skill_name=skill.name,
                    status=BATCH_STATUS_ERROR,
                    reason=f"{type(e).__name__}: {e}",
                ))
                batch.skills_errored += 1
                continue

            batch.outcomes.append(SkillBatchOutcome(
                skill_name=skill.name,
                status=BATCH_STATUS_RAN,
                summary=sub_summary,
            ))
            batch.skills_ran += 1
            for f in sub_review.findings:
                review.add_finding(f)

        # Deduplicate findings if requested. Aggregate first so we
        # accumulate ALL findings; then collapse duplicates. The
        # non-dedupe path just tallies and moves on.
        batch.total_findings = len(review.findings)
        if dedupe and review.findings:
            deduped, dropped = _dedupe_findings(review.findings)
            # Reassign in-place so any reference to review.findings
            # stays valid (matches the pattern used by
            # ReviewConfirmationService).
            review.findings.clear()
            review.findings.extend(deduped)
            batch.duplicates_dropped = dropped
        batch.findings_kept = len(review.findings)

        review.complete()
        return review, batch

    # ---- internals ----

    async def _evaluate_file(
        self,
        *,
        skill: SkillDefinition,
        file_path: str,
        hits: List[DiscoveryHit],
        review: SecurityReview,
    ) -> FileEvaluationOutcome:
        """Evaluate a single file and append its findings to the review.

        Never raises. Returns an outcome record; errors go in
        ``outcome.error`` for diagnostic display.
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return FileEvaluationOutcome(
                file_path=file_path,
                discovery_hit_count=len(hits),
                findings_emitted=0,
                error=f"Could not read file: {e}",
            )

        system_prompt = skill.body + _JSON_OUTPUT_INSTRUCTION
        user_prompt = _build_user_prompt(file_path, content, hits)

        try:
            raw = await self._llm.chat_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=_skill_response_schema(),
            )
        except Exception as e:
            _logger.warning(
                "Skill %s: LLM call failed on %s: %s",
                skill.name, file_path, e,
            )
            return FileEvaluationOutcome(
                file_path=file_path,
                discovery_hit_count=len(hits),
                findings_emitted=0,
                error=f"LLM call failed: {type(e).__name__}: {e}",
            )

        finding_dicts = _parse_llm_findings(raw)
        if finding_dicts is None:
            _logger.warning(
                "Skill %s: could not parse LLM response for %s (%d bytes)",
                skill.name, file_path, len(raw),
            )
            return FileEvaluationOutcome(
                file_path=file_path,
                discovery_hit_count=len(hits),
                findings_emitted=0,
                error="LLM response was not valid JSON",
            )

        emitted = 0
        for d in finding_dicts:
            finding = _finding_from_llm_dict(
                d,
                file_path_from_context=file_path,
                skill_name=skill.name,
            )
            if finding is not None:
                review.add_finding(finding)
                emitted += 1

        return FileEvaluationOutcome(
            file_path=file_path,
            discovery_hit_count=len(hits),
            findings_emitted=emitted,
        )



# --------------------------------------------------------------------
# Deduplication (batch runs)
# --------------------------------------------------------------------


#: Severity ordering, high-to-low. Used to pick the survivor when
#: multiple findings hit the same (file, line, category) key.
_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
}

_CONFIDENCE_RANK = {
    FindingConfidence.HIGH: 3,
    FindingConfidence.MEDIUM: 2,
    FindingConfidence.LOW: 1,
}


def _dedupe_findings(findings):
    """Collapse findings sharing (normalized_file, line, category).

    Returns (kept, dropped_count). The kept list preserves the
    original order of first-seen survivors so operators reading the
    report in top-to-bottom order see findings in a stable sequence
    across runs.

    Survivor selection:
      1. Highest severity wins.
      2. Ties broken by highest confidence.
      3. Further ties broken by first-seen order.

    The survivor accumulates the ``skill:<name>`` tags of every
    dropped duplicate so provenance is visible (an operator can see
    that both injection-prevention AND input-validation flagged the
    same site).
    """
    # Group by key preserving first-seen order.
    from collections import OrderedDict
    groups = OrderedDict()
    for f in findings:
        key = _dedupe_key(f)
        groups.setdefault(key, []).append(f)

    kept = []
    dropped = 0
    for key, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Pick survivor
        survivor = _pick_survivor(group)
        # Merge tags from dropped
        merged_tags = list(survivor.tags)
        for other in group:
            if other is survivor:
                continue
            for t in other.tags:
                if t not in merged_tags:
                    merged_tags.append(t)
        # Reconstruct survivor with merged tags (frozen dataclass ->
        # dataclasses.replace)
        import dataclasses as _dc
        survivor = _dc.replace(survivor, tags=merged_tags)
        kept.append(survivor)
        dropped += len(group) - 1
    return kept, dropped


def _dedupe_key(finding):
    """(normalized_file_path, line_start, category) key for grouping.

    Category is extracted from the ``category:<name>`` tag that the
    per-skill runner attaches. Falls back to "unknown" so a finding
    without an explicit category still participates in dedup (against
    other uncategorized findings on the same line).
    """
    file_key = _normalize_path_for_dedup(finding.file_path or "")
    line_key = finding.line_start or 0
    category = "unknown"
    for tag in finding.tags:
        if tag.startswith("category:"):
            category = tag[len("category:"):]
            break
    return (file_key, line_key, category)


def _normalize_path_for_dedup(p: str) -> str:
    """Normalize path for cross-OS deduplication comparison.

    Replaces backslashes, drops drive letters, lowercases. Same
    strategy as the reachability confirmer to keep behavior
    consistent across the confirmation and dedup passes.
    """
    import os as _os
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[2:]
    return _os.path.normpath(p.lower())


def _pick_survivor(group):
    """Return the highest-severity + highest-confidence finding.

    Preserves first-seen order for ties by relying on stable sort.
    """
    def sort_key(idx_finding):
        idx, f = idx_finding
        # Negative ranks so ``min`` picks the highest.
        return (
            -_SEVERITY_RANK.get(f.severity, 0),
            -_CONFIDENCE_RANK.get(f.confidence, 0),
            idx,  # tiebreak: first-seen
        )
    indexed = list(enumerate(group))
    indexed.sort(key=sort_key)
    return indexed[0][1]
