"""
Taint path verifier.

Consumes candidate :class:`TaintPath` entries from the path finder and
asks an LLM to confirm whether the taint actually flows from source to
sink through the caller chain, based on the actual method bodies.

The verifier is the ADVERSARIAL confirmer in the taint pipeline. The
path finder over-emits (structural candidates); the verifier reads
the code and prunes candidates that don't hold up under scrutiny.
This mirrors the two-pass design of the existing FindingVerifier but
adapted for cross-method flow reasoning rather than single-file
findings.

The verifier makes one LLM call per candidate path. Failures (network
errors, malformed responses) leave the path tagged rather than dropped
-- the operator sees "verifier_call_failed" in the output and can
decide manually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..models.taint import FunctionDef, TaintPath
from ..models.security import FindingConfidence, Severity

if TYPE_CHECKING:
    from .call_graph import CallGraph
    from .llm_service import LLMService
    from ...infrastructure.taint_index import TaintIndex


# Verdicts the LLM may return. The taint pipeline has narrower needs
# than the general finding verifier -- we only care about confirmed /
# false positive / uncertain. Downgrade (adjust severity) is deferred.
VERIFIER_VERDICT_CONFIRMED = "confirmed"
VERIFIER_VERDICT_FALSE_POSITIVE = "false_positive"
VERIFIER_VERDICT_UNCERTAIN = "uncertain"
_ALL_VERDICTS = [
    VERIFIER_VERDICT_CONFIRMED,
    VERIFIER_VERDICT_FALSE_POSITIVE,
    VERIFIER_VERDICT_UNCERTAIN,
]


_VERIFIER_SYSTEM_PROMPT = """You are a security expert reviewing a candidate taint flow that a
static analyzer identified based on call structure. The analyzer saw
that a "source" (untrusted input) is called somewhere in a chain of
methods that ends at a "sink" (dangerous operation). You must read
the actual method bodies and determine whether tainted data really
flows from the source to the sink.

The candidate is structural, not semantic. It may be a false positive
if any of the following apply:

- The tainted return value from the source is discarded or never
  reaches the sink call.
- A sanitizer, validator, or type conversion cleans the value between
  source and sink.
- The sink is called with a constant, a different variable, or a
  value that came from a different (untainted) source.
- Framework-level protection (parameterized queries, escaping) makes
  the sink call safe despite receiving user input.
- The call chain is real but no data is actually passed along it
  (e.g. the intermediate method ignores its argument).

Deliver one of three verdicts:

- "confirmed": Tainted data from the source really does reach the
  sink. Provide reasoning citing specific lines showing the flow.
- "false_positive": You have a specific reason the flow is not real.
  Explain in reasoning.
- "uncertain": The evidence is genuinely mixed or the code is too
  complex to be sure. Say why.

Be honest. Do not split the difference to look thorough. If the
finding is clearly real, say "confirmed". If it's clearly bogus,
say "false_positive". "uncertain" is for genuine ambiguity.

Respond ONLY with a single JSON object matching the schema. Do NOT
include any text before or after the JSON. Do NOT use markdown fences.
"""


def _verifier_response_schema() -> dict:
    """JSON schema for the verifier's response."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": _ALL_VERDICTS},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "reasoning": {"type": "string"},
            "evidence_lines": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["verdict", "confidence", "reasoning"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class VerifierResult:
    """Structured LLM verdict on one candidate TaintPath.

    ``verdict`` is one of the three module-level constants. Callers
    that want to emit a SecurityFinding should check for
    ``VERIFIER_VERDICT_CONFIRMED`` (definite hit) and typically also
    include ``VERIFIER_VERDICT_UNCERTAIN`` results (flagged for human
    review). ``VERIFIER_VERDICT_FALSE_POSITIVE`` should be dropped
    unless the operator has explicitly asked to see them.

    ``call_failed`` is True when the LLM call itself failed (network
    error, malformed response). The verdict field will be
    ``VERIFIER_VERDICT_UNCERTAIN`` in this case; the caller should
    tag the finding with something like ``verifier:call_failed``
    rather than trusting the verdict.
    """
    verdict: str
    confidence: FindingConfidence
    reasoning: str
    evidence_lines: Tuple[int, ...] = ()
    call_failed: bool = False


class TaintVerifier:
    """LLM-based confirmer of candidate taint paths.

    Stateless with respect to input. Uses the LLM service passed to
    the constructor; a single verifier can be reused across many
    scans and paths.
    """

    def __init__(self, *, llm_service: "LLMService") -> None:
        self._llm = llm_service

    async def verify(
        self,
        *,
        path: TaintPath,
        graph: "CallGraph",
        taint_index: "TaintIndex",
    ) -> VerifierResult:
        """Ask the LLM whether ``path`` represents a real taint flow.

        Assembles method bodies for every step in ``path.via`` and
        includes them in the user prompt. Method bodies are read
        directly from the file on disk using the location info from
        the graph's :class:`FunctionDef` for that node.

        Handles LLM failure modes conservatively: on any exception
        or parse failure, returns ``verdict=UNCERTAIN`` with
        ``call_failed=True`` and the caller decides how to tag.
        """
        user_prompt = self._build_user_prompt(path, graph, taint_index)

        try:
            raw = await self._llm.chat_with_schema(
                system_prompt=_VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=_verifier_response_schema(),
            )
        except Exception:
            return VerifierResult(
                verdict=VERIFIER_VERDICT_UNCERTAIN,
                confidence=FindingConfidence.LOW,
                reasoning="Verifier LLM call failed.",
                call_failed=True,
            )

        parsed = self._parse_response(raw)
        if parsed is None:
            return VerifierResult(
                verdict=VERIFIER_VERDICT_UNCERTAIN,
                confidence=FindingConfidence.LOW,
                reasoning="Verifier response could not be parsed as JSON matching the schema.",
                call_failed=True,
            )
        return parsed

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        path: TaintPath,
        graph: "CallGraph",
        taint_index: "TaintIndex",
    ) -> str:
        """Compose the code + question prompt.

        Structure:
          - Overview of the candidate flow (source, sink, chain).
          - Metadata about source and sink (kind, CWE, severity).
          - Method bodies for every step in the chain, with line
            numbers so the LLM can cite specific lines as evidence.
        """
        source_spec = taint_index.get_source(path.source)
        sink_spec = taint_index.get_sink(path.sink)

        source_kind = source_spec.kind if source_spec else "unknown"
        sink_kind = sink_spec.kind if sink_spec else "unknown"
        sink_cwe = sink_spec.cwe if sink_spec else "unknown"
        sink_severity = sink_spec.severity if sink_spec else "unknown"

        parts: List[str] = []
        parts.append("CANDIDATE TAINT FLOW")
        parts.append("=" * 60)
        parts.append(f"Source: {path.source}")
        parts.append(f"  kind: {source_kind}")
        parts.append(f"  called by: {path.source_caller}")
        parts.append("")
        parts.append(f"Sink: {path.sink}")
        parts.append(f"  kind: {sink_kind}")
        parts.append(f"  cwe: {sink_cwe}")
        parts.append(f"  severity: {sink_severity}")
        parts.append(f"  called by: {path.sink_caller}")
        parts.append("")
        parts.append(f"Hop count: {path.hop_count}")
        parts.append(f"Chain: {' -> '.join(path.via)}")
        parts.append("")
        parts.append("=" * 60)
        parts.append("METHOD BODIES")
        parts.append("=" * 60)

        for step in path.via:
            fn = graph.function(step)
            if fn is None:
                # No FunctionDef -- this is a spec-only endpoint (source
                # or sink itself, not a user method). Skip; the caller
                # methods are what matter for verification.
                continue
            body = self._read_method_body(fn)
            parts.append("")
            parts.append(f"--- {step} ---")
            parts.append(f"file: {fn.location.file_path}")
            parts.append(f"lines: {fn.location.line_start}-{fn.location.line_end}")
            parts.append("")
            parts.append(body)

        parts.append("")
        parts.append("=" * 60)
        parts.append("QUESTION")
        parts.append("=" * 60)
        parts.append(
            "Does tainted data from the source actually reach the sink "
            "through this call chain? Cite specific line numbers in "
            "your reasoning and in evidence_lines."
        )

        return "\n".join(parts)

    def _read_method_body(self, fn: FunctionDef) -> str:
        """Read the method's source lines from disk, numbered.

        Silent on I/O errors: returns a placeholder so verification
        can still proceed. The verifier prompt already warns the LLM
        that code may be truncated / unavailable.
        """
        try:
            content = Path(fn.location.file_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return f"(method body not readable at {fn.location.file_path})"

        lines = content.splitlines()
        start = max(fn.location.line_start - 1, 0)
        end = min(fn.location.line_end, len(lines))
        if start >= end:
            return "(empty method body)"

        # Number each line with its actual file line number so the LLM
        # can cite them in evidence_lines. Zero-pad width based on the
        # largest line number for consistent alignment.
        width = len(str(end))
        numbered = []
        for i, line in enumerate(lines[start:end], start=fn.location.line_start):
            numbered.append(f"{i:>{width}}  {line}")
        return "\n".join(numbered)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Optional[VerifierResult]:
        """Parse the LLM's JSON response. Returns None on failure.

        Robust to:
          - Trailing / leading whitespace
          - Markdown code fences the LLM slipped in despite the prompt
          - Empty responses
          - Missing optional fields (evidence_lines)
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip fenced-code block: ``` ... ``` or ```json ... ```
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return None

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None

        verdict = data.get("verdict")
        if verdict not in _ALL_VERDICTS:
            return None
        confidence_raw = str(data.get("confidence", "low")).lower()
        if confidence_raw not in ("high", "medium", "low"):
            confidence_raw = "low"
        reasoning = str(data.get("reasoning", ""))
        evidence_raw = data.get("evidence_lines", [])
        if not isinstance(evidence_raw, list):
            evidence_raw = []
        # Coerce to ints, dropping anything that isn't an int-compatible value.
        evidence: List[int] = []
        for e in evidence_raw:
            try:
                evidence.append(int(e))
            except (TypeError, ValueError):
                continue

        return VerifierResult(
            verdict=verdict,
            confidence=FindingConfidence(confidence_raw),
            reasoning=reasoning,
            evidence_lines=tuple(evidence),
        )
