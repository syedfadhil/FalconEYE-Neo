"""Unit tests for the skill runner service (Phase 1b).

Covers:
* Prompt construction (line numbering, hit listing, truncation).
* LLM response parsing (fenced blocks, prefaced text, bare arrays,
  malformed input, empty).
* Finding conversion (severity/confidence normalization, missing
  fields, file-path fallback, tags).
* Full end-to-end run with a mock LLM (real skill file, real
  workspace, verify SecurityReview shape).
* Per-file failure isolation (LLM error on one file doesn't stop
  processing of others).
* Gating: LLM-only skills and non-scan skills are rejected up front.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest

from falconeye.application.services.skill_runner_service import (
    SkillRunnerService,
    _build_user_prompt,
    _confidence_from_llm,
    _finding_from_llm_dict,
    _number_lines,
    _parse_llm_findings,
    _severity_from_llm,
)
from falconeye.domain.models.security import FindingConfidence, Severity
from falconeye.domain.models.skill import DiscoveryHit, DiscoveryPattern
from falconeye.domain.services.llm_service import LLMService
from falconeye.infrastructure.skills.skill_loader import SkillLoader


# ---------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------


class MockLLM(LLMService):
    """Configurable mock returning canned responses.

    ``responses`` is a list of strings; each call consumes one. If
    a callable is used instead, it receives (system_prompt,
    user_prompt) and returns the response string. Errors can be
    injected by making a callable raise.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def analyze_code_security(self, context, system_prompt, user_prompt, **kw):
        return "{}"

    async def generate_embedding(self, text): return [0.0]
    async def generate_embeddings_batch(self, texts): return [[0.0] for _ in texts]
    async def validate_findings(self, *a, **k): return []
    async def summarize_findings(self, *a, **k): return ""
    async def health_check(self): return True
    def count_tokens(self, text): return len(text) // 4

    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        self.calls.append((system_prompt[:60], user_prompt[:100]))
        if not self._responses:
            return '{"findings": []}'
        r = self._responses.pop(0)
        if callable(r):
            return r(system_prompt, user_prompt)
        return r


# ---------------------------------------------------------------
# _number_lines
# ---------------------------------------------------------------


@pytest.mark.unit
def test_number_lines_prefixes_each_line_with_number():
    src = "def a():\n    return 1\n"
    out = _number_lines(src)
    # 1-based, right-justified
    assert "    1  def a():" in out
    assert "    2      return 1" in out


@pytest.mark.unit
def test_number_lines_truncates_very_long_files():
    """Files over the 4000-line cap get a truncation marker."""
    src = "\n".join(f"line_{i}" for i in range(5000))
    out = _number_lines(src)
    assert "4000 more lines omitted" in out or "1000 more lines omitted" in out


# ---------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------


@pytest.mark.unit
def test_user_prompt_includes_file_hits_and_numbered_source():
    hits = [
        DiscoveryHit(
            file_path="/x.py", line_number=2,
            line_text="cursor.execute(sql + uid)",
            matched_span=(0, 14), pattern_description="raw sql exec",
        ),
    ]
    prompt = _build_user_prompt(
        file_path="/x.py",
        file_content="import x\ncursor.execute(sql + uid)\n",
        hits=hits,
    )
    # Header markers
    assert "## File" in prompt
    assert "## Discovery hits" in prompt
    assert "## Source (numbered)" in prompt
    # Hit description surfaces
    assert "raw sql exec" in prompt
    assert "line 2:" in prompt
    # Source numbered
    assert "1  import x" in prompt
    assert "2  cursor.execute" in prompt


@pytest.mark.unit
def test_user_prompt_with_no_hits_shows_empty_marker():
    prompt = _build_user_prompt("/x.py", "code\n", [])
    assert "(none)" in prompt


# ---------------------------------------------------------------
# _parse_llm_findings
# ---------------------------------------------------------------


@pytest.mark.unit
def test_parse_direct_json_object():
    raw = json.dumps({"findings": [{"issue": "x"}]})
    result = _parse_llm_findings(raw)
    assert result == [{"issue": "x"}]


@pytest.mark.unit
def test_parse_empty_findings_array():
    raw = json.dumps({"findings": []})
    assert _parse_llm_findings(raw) == []


@pytest.mark.unit
def test_parse_fenced_json_block():
    raw = '```json\n{"findings": [{"issue": "x"}]}\n```'
    result = _parse_llm_findings(raw)
    assert result == [{"issue": "x"}]


@pytest.mark.unit
def test_parse_bare_fence_without_language_tag():
    raw = '```\n{"findings": [{"issue": "x"}]}\n```'
    result = _parse_llm_findings(raw)
    assert result == [{"issue": "x"}]


@pytest.mark.unit
def test_parse_with_preface_text():
    """Model prefaces JSON with a natural-language sentence."""
    raw = 'Sure, here are the findings:\n{"findings": [{"issue": "x"}]}'
    result = _parse_llm_findings(raw)
    assert result == [{"issue": "x"}]


@pytest.mark.unit
def test_parse_bare_array_wraps_as_findings():
    raw = '[{"issue": "x"}, {"issue": "y"}]'
    result = _parse_llm_findings(raw)
    assert result == [{"issue": "x"}, {"issue": "y"}]


@pytest.mark.unit
def test_parse_empty_string_returns_none():
    assert _parse_llm_findings("") is None
    assert _parse_llm_findings("   \n") is None


@pytest.mark.unit
def test_parse_malformed_returns_none():
    assert _parse_llm_findings("{ not json") is None


@pytest.mark.unit
def test_parse_non_findings_object_returns_none():
    """An object without the ``findings`` key is not a valid response."""
    raw = json.dumps({"other_key": []})
    assert _parse_llm_findings(raw) is None


# ---------------------------------------------------------------
# _severity_from_llm / _confidence_from_llm
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("input_str,expected", [
    ("Critical", Severity.CRITICAL),
    ("critical", Severity.CRITICAL),
    ("HIGH", Severity.HIGH),
    ("Medium", Severity.MEDIUM),
    ("Low", Severity.LOW),
    ("Info", Severity.LOW),  # Info coerces to Low
    ("informational", Severity.LOW),
    ("nonsense", Severity.MEDIUM),  # unknown -> default
    (None, Severity.MEDIUM),
    ("", Severity.MEDIUM),
])
def test_severity_normalization(input_str, expected):
    assert _severity_from_llm(input_str) == expected


@pytest.mark.unit
@pytest.mark.parametrize("input_str,expected", [
    ("High", FindingConfidence.HIGH),
    ("high", FindingConfidence.HIGH),
    ("Medium", FindingConfidence.MEDIUM),
    ("Low", FindingConfidence.LOW),
    ("nonsense", FindingConfidence.MEDIUM),
    (None, FindingConfidence.MEDIUM),
])
def test_confidence_normalization(input_str, expected):
    assert _confidence_from_llm(input_str) == expected


# ---------------------------------------------------------------
# _finding_from_llm_dict
# ---------------------------------------------------------------


@pytest.mark.unit
def test_finding_from_llm_dict_full_fields():
    d = {
        "issue": "SQL injection",
        "severity": "High",
        "confidence": "High",
        "file": "/x.py",
        "line": 42,
        "description": "Concatenated input",
        "recommended_fix": "Parameterize",
        "cwe": "CWE-89",
        "category": "injection",
        "evidence_lines": [42, 43],
    }
    f = _finding_from_llm_dict(d, file_path_from_context="/other.py", skill_name="injection-prevention")
    assert f is not None
    assert f.issue == "SQL injection"
    assert f.severity == Severity.HIGH
    assert f.confidence == FindingConfidence.HIGH
    assert f.file_path == "/x.py"  # from LLM, not context
    assert f.line_start == 42
    assert f.cwe_id == "CWE-89"
    assert f.mitigation == "Parameterize"
    assert "skill:injection-prevention" in f.tags
    assert "category:injection" in f.tags
    assert f.evidence_lines == [42, 43]


@pytest.mark.unit
def test_finding_from_llm_dict_missing_issue_returns_none():
    """No issue text -> unusable finding, skip it."""
    d = {"severity": "High", "confidence": "High"}
    assert _finding_from_llm_dict(d, file_path_from_context="/x.py", skill_name="s") is None


@pytest.mark.unit
def test_finding_from_llm_dict_uses_context_when_file_missing():
    d = {"issue": "X"}
    f = _finding_from_llm_dict(d, file_path_from_context="/ctx.py", skill_name="s")
    assert f is not None
    assert f.file_path == "/ctx.py"


@pytest.mark.unit
def test_finding_from_llm_dict_null_cwe_becomes_none():
    for null_form in ("null", "None", ""):
        d = {"issue": "x", "cwe": null_form}
        f = _finding_from_llm_dict(d, file_path_from_context="/x.py", skill_name="s")
        assert f.cwe_id is None, f"cwe={null_form!r} should have been normalized"


# ---------------------------------------------------------------
# SkillRunnerService.run - end to end with mock LLM
# ---------------------------------------------------------------


def _write_scan_skill(root: Path) -> Path:
    """Minimal runnable skill (function=scan, has grep patterns)."""
    d = root / "test-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(textwrap.dedent("""\
    ---
    name: test-skill
    description: Test skill
    function: scan
    category: test
    ---

    # Test Skill

    ## Step 1 -- Discover

    ```bash
    # dangerous stuff
    grep -rn -E "dangerous_call" --include="*.py" .
    ```
    """), encoding="utf-8")
    return d / "SKILL.md"


@pytest.mark.unit
def test_run_end_to_end_emits_findings(tmp_path):
    """The full happy path: skill loads, discovery finds hits, LLM
    returns findings, SecurityReview is populated."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_scan_skill(skills_dir)
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("def x():\n    dangerous_call(uid)\n")

    skill = SkillLoader(directory=skills_dir).load_all()[0]

    llm = MockLLM([json.dumps({
        "findings": [{
            "issue": "Dangerous call detected",
            "severity": "High", "confidence": "High",
            "file": str(ws / "vuln.py"), "line": 2,
            "description": "call with user input",
            "recommended_fix": "sanitize",
            "cwe": "CWE-77", "category": "injection",
        }],
    })])
    runner = SkillRunnerService(llm_service=llm)
    review, summary = asyncio.run(runner.run(skill=skill, workspace=ws))

    assert len(review.findings) == 1
    assert review.findings[0].issue == "Dangerous call detected"
    assert review.findings[0].severity == Severity.HIGH
    assert summary.total_discovery_hits == 1
    assert summary.files_evaluated == 1
    assert summary.total_findings == 1
    assert summary.files_with_llm_errors == 0


@pytest.mark.unit
def test_run_per_file_llm_error_is_isolated(tmp_path):
    """Failure on one file must not stop processing of the next.

    The failure is recorded in the summary; other files still get
    their findings emitted normally.
    """
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_scan_skill(skills_dir)
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("dangerous_call()\n")
    (ws / "b.py").write_text("dangerous_call()\n")

    skill = SkillLoader(directory=skills_dir).load_all()[0]

    def raise_on_first(system, user):
        raise RuntimeError("mock LLM outage")

    llm = MockLLM([
        raise_on_first,
        json.dumps({"findings": [{
            "issue": "found on b", "severity": "Low", "confidence": "Medium",
            "description": "x", "recommended_fix": "y",
        }]}),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, summary = asyncio.run(runner.run(skill=skill, workspace=ws))

    # Second file's finding emitted
    assert len(review.findings) == 1
    assert review.findings[0].issue == "found on b"
    # Summary records the failure
    assert summary.files_evaluated == 2
    assert summary.files_with_llm_errors == 1
    error_outcomes = [o for o in summary.outcomes if o.error is not None]
    assert len(error_outcomes) == 1
    assert "mock LLM outage" in error_outcomes[0].error


@pytest.mark.unit
def test_run_unparseable_response_records_error(tmp_path):
    """LLM returns garbage -> file's outcome error is recorded, no findings."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_scan_skill(skills_dir)
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skill = SkillLoader(directory=skills_dir).load_all()[0]
    llm = MockLLM(["not valid json at all"])
    runner = SkillRunnerService(llm_service=llm)
    review, summary = asyncio.run(runner.run(skill=skill, workspace=ws))
    assert len(review.findings) == 0
    assert summary.files_with_llm_errors == 1
    assert "not valid JSON" in summary.outcomes[0].error


@pytest.mark.unit
def test_run_batches_by_file_not_by_hit(tmp_path):
    """Multiple hits in one file -> one LLM call, not one per hit.

    Sending each hit separately would re-transmit the skill body per
    call, exploding cost.
    """
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_scan_skill(skills_dir)
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text(
        "dangerous_call(1)\ndangerous_call(2)\ndangerous_call(3)\n"
    )
    skill = SkillLoader(directory=skills_dir).load_all()[0]

    llm = MockLLM([json.dumps({"findings": []})])
    runner = SkillRunnerService(llm_service=llm)
    review, summary = asyncio.run(runner.run(skill=skill, workspace=ws))

    assert summary.total_discovery_hits == 3
    assert summary.files_evaluated == 1
    assert len(llm.calls) == 1  # not 3


@pytest.mark.unit
def test_run_rejects_non_scan_skill(tmp_path):
    """function=verify skill rejected with a message pointing at the
    correct pipeline stage (fp-check / Phase 3)."""
    skills_dir = tmp_path / "skills" / "verify-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
    ---
    name: verify-skill
    description: verifier
    function: verify
    ---
    ## Step 1
    ```bash
    grep -rn -E "x" --include="*.py" .
    ```
    """))
    skill = SkillLoader(directory=skills_dir.parent).load_all()[0]

    runner = SkillRunnerService(llm_service=MockLLM([]))
    with pytest.raises(ValueError, match=r"function='verify'"):
        asyncio.run(runner.run(skill=skill, workspace=tmp_path))


@pytest.mark.unit
def test_run_rejects_skill_without_discovery_patterns(tmp_path):
    """LLM-only skills (no Step 1 grep patterns) are Phase 1c work."""
    skills_dir = tmp_path / "skills" / "no-patterns"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(textwrap.dedent("""\
    ---
    name: no-patterns
    description: LLM-only
    function: scan
    ---
    ## Step 1 -- Locate

    Read every .py file and look for issues.
    """))
    skill = SkillLoader(directory=skills_dir.parent).load_all()[0]
    assert skill.discovery_patterns == ()

    runner = SkillRunnerService(llm_service=MockLLM([]))
    with pytest.raises(ValueError, match="no Step 1 discovery patterns"):
        asyncio.run(runner.run(skill=skill, workspace=tmp_path))


@pytest.mark.unit
def test_run_completes_review_at_end(tmp_path):
    """SecurityReview.completed_at is set after the run, matching
    the invariant used by reporters (uncompleted reviews render
    'In progress' duration)."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_scan_skill(skills_dir)
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")
    skill = SkillLoader(directory=skills_dir).load_all()[0]
    llm = MockLLM([json.dumps({"findings": []})])
    runner = SkillRunnerService(llm_service=llm)
    review, _ = asyncio.run(runner.run(skill=skill, workspace=ws))
    assert review.completed_at is not None



# =====================================================================
# run_all: batch execution
# =====================================================================


from falconeye.application.services.skill_runner_service import (  # noqa: E402
    BATCH_STATUS_ERROR,
    BATCH_STATUS_RAN,
    BATCH_STATUS_SKIPPED_NO_PATTERNS,
    BATCH_STATUS_SKIPPED_NOT_SCAN,
    SkillBatchOutcome,
    SkillBatchSummary,
    _dedupe_findings,
    _dedupe_key,
    _normalize_path_for_dedup,
    _pick_survivor,
)
from falconeye.domain.models.security import SecurityFinding
from uuid import uuid4


def _write_named_scan_skill(root: Path, name: str, category: str = "test") -> Path:
    d = root / name
    d.mkdir()
    (d / "SKILL.md").write_text(textwrap.dedent(f"""\
    ---
    name: {name}
    description: test
    function: scan
    category: {category}
    ---
    ## Step 1
    ```bash
    grep -rn -E "dangerous_call" --include="*.py" .
    ```
    """), encoding="utf-8")
    return d / "SKILL.md"


def _write_verify_skill(root: Path, name: str) -> Path:
    d = root / name; d.mkdir()
    (d / "SKILL.md").write_text(textwrap.dedent(f"""\
    ---
    name: {name}
    description: verifier
    function: verify
    ---
    ## Step 1
    ```bash
    grep -rn -E "x" --include="*.py" .
    ```
    """), encoding="utf-8")
    return d / "SKILL.md"


def _write_llm_only_skill(root: Path, name: str) -> Path:
    d = root / name; d.mkdir()
    (d / "SKILL.md").write_text(textwrap.dedent(f"""\
    ---
    name: {name}
    description: llm only
    function: scan
    ---
    ## Step 1

    Read every file.
    """), encoding="utf-8")
    return d / "SKILL.md"


def _minimal_finding_response(issue: str, severity: str = "High",
                              category: str = "injection",
                              line: int = 2) -> str:
    """Build a minimal ``{"findings": [...]}`` LLM response string."""
    return json.dumps({"findings": [{
        "issue": issue, "severity": severity, "confidence": "High",
        "file": "vuln.py", "line": line,
        "description": "x", "recommended_fix": "y",
        "cwe": None, "category": category,
    }]})


@pytest.mark.unit
def test_run_all_runs_every_runnable_skill(tmp_path):
    """Two runnable skills: both execute and both contribute findings."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "s1", category="a")
    _write_named_scan_skill(skills_dir, "s2", category="b")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("def x():\n    dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()
    assert len(skills) == 2

    llm = MockLLM([
        _minimal_finding_response("issue-a", category="a"),
        _minimal_finding_response("issue-b", category="b"),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(
        skills=skills, workspace=ws, dedupe=False,
    ))

    assert batch.total_skills_seen == 2
    assert batch.skills_ran == 2
    assert batch.skills_skipped == 0
    assert batch.skills_errored == 0
    assert len(review.findings) == 2
    issues = sorted(f.issue for f in review.findings)
    assert issues == ["issue-a", "issue-b"]


@pytest.mark.unit
def test_run_all_skips_non_scan_skills(tmp_path):
    """function=verify -> BATCH_STATUS_SKIPPED_NOT_SCAN."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "runnable")
    _write_verify_skill(skills_dir, "vfy")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([_minimal_finding_response("found")])
    runner = SkillRunnerService(llm_service=llm)
    _, batch = asyncio.run(runner.run_all(skills=skills, workspace=ws))

    assert batch.skills_ran == 1
    assert batch.skills_skipped == 1
    vfy_outcome = next(o for o in batch.outcomes if o.skill_name == "vfy")
    assert vfy_outcome.status == BATCH_STATUS_SKIPPED_NOT_SCAN
    assert "function=verify" in vfy_outcome.reason


@pytest.mark.unit
def test_run_all_skips_llm_only_skills(tmp_path):
    """No Step 1 grep patterns -> BATCH_STATUS_SKIPPED_NO_PATTERNS."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "with-patterns")
    _write_llm_only_skill(skills_dir, "no-patterns")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([_minimal_finding_response("found")])
    runner = SkillRunnerService(llm_service=llm)
    _, batch = asyncio.run(runner.run_all(skills=skills, workspace=ws))

    assert batch.skills_ran == 1
    assert batch.skills_skipped == 1
    no_pat = next(o for o in batch.outcomes if o.skill_name == "no-patterns")
    assert no_pat.status == BATCH_STATUS_SKIPPED_NO_PATTERNS
    assert "no Step 1 discovery patterns" in no_pat.reason


@pytest.mark.unit
def test_run_all_isolates_skill_errors(tmp_path):
    """A skill whose run() raises is marked errored but the batch continues."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "broken", category="a")
    _write_named_scan_skill(skills_dir, "healthy", category="b")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()

    # Both skills invoke the LLM once. Make the FIRST call raise via
    # a callable in the response queue; the second returns a normal
    # response. The `broken` skill is loaded first alphabetically,
    # so it hits the raising call.
    def raise_call(system, user):
        raise RuntimeError("simulated outage")
    llm = MockLLM([raise_call, _minimal_finding_response("from healthy")])

    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(skills=skills, workspace=ws))

    # broken skill's run() succeeded (the LLM error was caught per-file
    # inside run()); the skill produced ZERO findings but did NOT raise.
    # So both skills show BATCH_STATUS_RAN; the broken skill's
    # per-file summary carries files_with_llm_errors == 1.
    broken = next(o for o in batch.outcomes if o.skill_name == "broken")
    healthy = next(o for o in batch.outcomes if o.skill_name == "healthy")
    assert broken.status == BATCH_STATUS_RAN
    assert broken.summary is not None
    assert broken.summary.files_with_llm_errors == 1
    assert broken.summary.total_findings == 0
    assert healthy.status == BATCH_STATUS_RAN
    assert healthy.summary.total_findings == 1

    # Healthy findings still land in the review
    assert len(review.findings) == 1
    assert review.findings[0].issue == "from healthy"


@pytest.mark.unit
def test_run_all_dedupe_collapses_same_key(tmp_path):
    """Two skills flagging (file=vuln.py, line=2, category=injection) -> 1."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "s1", category="injection")
    _write_named_scan_skill(skills_dir, "s2", category="injection")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([
        _minimal_finding_response("dup", severity="High", category="injection"),
        _minimal_finding_response("dup", severity="Medium", category="injection"),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(
        skills=skills, workspace=ws, dedupe=True,
    ))

    assert batch.total_findings == 2
    assert batch.findings_kept == 1
    assert batch.duplicates_dropped == 1
    # Survivor is the High-severity one
    assert review.findings[0].severity == Severity.HIGH


@pytest.mark.unit
def test_run_all_dedupe_survivor_accumulates_skill_tags(tmp_path):
    """When two skills flag the same site, the survivor's tags
    show BOTH skill provenance so the operator sees who found what."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "aaa", category="injection")
    _write_named_scan_skill(skills_dir, "bbb", category="injection")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")

    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([
        _minimal_finding_response("shared"),
        _minimal_finding_response("shared"),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(
        skills=skills, workspace=ws, dedupe=True,
    ))
    assert len(review.findings) == 1
    tags = review.findings[0].tags
    assert "skill:aaa" in tags
    assert "skill:bbb" in tags


@pytest.mark.unit
def test_run_all_no_dedupe_keeps_all_findings(tmp_path):
    """--no-dedupe path: every skill's findings preserved separately."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "s1", category="injection")
    _write_named_scan_skill(skills_dir, "s2", category="injection")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")
    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([
        _minimal_finding_response("dup"),
        _minimal_finding_response("dup"),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(
        skills=skills, workspace=ws, dedupe=False,
    ))
    assert len(review.findings) == 2
    assert batch.total_findings == 2
    assert batch.findings_kept == 2
    assert batch.duplicates_dropped == 0


@pytest.mark.unit
def test_run_all_different_categories_are_not_deduped(tmp_path):
    """(file, line, category) tuple keeps two DIFFERENT categories separate."""
    skills_dir = tmp_path / "skills"; skills_dir.mkdir()
    _write_named_scan_skill(skills_dir, "s1", category="injection")
    _write_named_scan_skill(skills_dir, "s2", category="path-traversal")
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "vuln.py").write_text("dangerous_call()\n")
    skills = SkillLoader(directory=skills_dir).load_all()
    llm = MockLLM([
        _minimal_finding_response("a", category="injection"),
        _minimal_finding_response("b", category="path-traversal"),
    ])
    runner = SkillRunnerService(llm_service=llm)
    review, batch = asyncio.run(runner.run_all(
        skills=skills, workspace=ws, dedupe=True,
    ))
    assert len(review.findings) == 2
    assert batch.duplicates_dropped == 0


@pytest.mark.unit
def test_run_all_empty_skill_list_produces_empty_review(tmp_path):
    """No skills at all -> empty review, empty batch, no crash."""
    ws = tmp_path / "ws"; ws.mkdir()
    runner = SkillRunnerService(llm_service=MockLLM([]))
    review, batch = asyncio.run(runner.run_all(
        skills=[], workspace=ws, dedupe=True,
    ))
    assert len(review.findings) == 0
    assert batch.total_skills_seen == 0
    assert batch.skills_ran == 0


# ---------------------------------------------------------------
# _dedupe_findings helper
# ---------------------------------------------------------------


def _finding(*, file_path="/x.py", line=1, severity=Severity.HIGH,
             confidence=FindingConfidence.HIGH, issue="i",
             category="injection", extra_tags=()) -> SecurityFinding:
    tags = [f"category:{category}", *extra_tags]
    return SecurityFinding(
        id=uuid4(), issue=issue, reasoning="", mitigation="",
        severity=severity, confidence=confidence,
        file_path=file_path, code_snippet="",
        line_start=line, line_end=None,
        tags=tags,
    )


@pytest.mark.unit
def test_dedupe_key_uses_normalized_path():
    """C:\\proj\\file.py and /proj/file.py hash to the same dedup key
    if their normalized forms match."""
    f_win = _finding(file_path="C:\\proj\\file.py", line=5, category="c")
    f_nix = _finding(file_path="/proj/file.py", line=5, category="c")
    assert _dedupe_key(f_win) == _dedupe_key(f_nix)


@pytest.mark.unit
def test_dedupe_key_falls_back_to_unknown_category():
    """A finding without a category tag participates in dedup as
    category=unknown -- still groups with other uncategorized findings."""
    f = SecurityFinding(
        id=uuid4(), issue="i", reasoning="", mitigation="",
        severity=Severity.LOW, confidence=FindingConfidence.LOW,
        file_path="/x", code_snippet="", line_start=1, line_end=None,
        tags=[],  # no category tag
    )
    key = _dedupe_key(f)
    assert key[2] == "unknown"


@pytest.mark.unit
def test_dedupe_picks_highest_severity_survivor():
    findings = [
        _finding(severity=Severity.LOW, issue="low"),
        _finding(severity=Severity.CRITICAL, issue="crit"),
        _finding(severity=Severity.MEDIUM, issue="med"),
    ]
    kept, dropped = _dedupe_findings(findings)
    assert len(kept) == 1
    assert kept[0].issue == "crit"
    assert dropped == 2


@pytest.mark.unit
def test_dedupe_tiebreaks_severity_by_confidence():
    """Same severity -> pick highest confidence."""
    findings = [
        _finding(severity=Severity.HIGH, confidence=FindingConfidence.LOW, issue="h-lo"),
        _finding(severity=Severity.HIGH, confidence=FindingConfidence.HIGH, issue="h-hi"),
        _finding(severity=Severity.HIGH, confidence=FindingConfidence.MEDIUM, issue="h-med"),
    ]
    kept, _ = _dedupe_findings(findings)
    assert len(kept) == 1
    assert kept[0].issue == "h-hi"


@pytest.mark.unit
def test_dedupe_tiebreak_severity_confidence_by_first_seen():
    """Same severity AND confidence -> keep the first-seen."""
    findings = [
        _finding(issue="first"),
        _finding(issue="second"),
        _finding(issue="third"),
    ]
    kept, _ = _dedupe_findings(findings)
    assert len(kept) == 1
    assert kept[0].issue == "first"


@pytest.mark.unit
def test_dedupe_no_duplicates_preserves_order():
    """Findings with distinct keys are all kept in first-seen order."""
    findings = [
        _finding(file_path="/a", issue="a"),
        _finding(file_path="/b", issue="b"),
        _finding(file_path="/c", issue="c"),
    ]
    kept, dropped = _dedupe_findings(findings)
    assert [f.issue for f in kept] == ["a", "b", "c"]
    assert dropped == 0


@pytest.mark.unit
def test_dedupe_survivor_accumulates_dropped_tags():
    """Tags from dropped duplicates merge into the survivor's tags."""
    findings = [
        _finding(severity=Severity.CRITICAL, issue="win",
                 extra_tags=("skill:aaa",)),
        _finding(severity=Severity.MEDIUM, issue="lose1",
                 extra_tags=("skill:bbb",)),
        _finding(severity=Severity.LOW, issue="lose2",
                 extra_tags=("skill:ccc", "custom:tag")),
    ]
    kept, _ = _dedupe_findings(findings)
    tags = kept[0].tags
    assert "skill:aaa" in tags
    assert "skill:bbb" in tags
    assert "skill:ccc" in tags
    assert "custom:tag" in tags


@pytest.mark.unit
def test_dedupe_empty_input_returns_empty():
    kept, dropped = _dedupe_findings([])
    assert kept == []
    assert dropped == 0


@pytest.mark.unit
def test_normalize_path_for_dedup_windows_and_posix_equal():
    """Windows drive letter + backslashes normalize identically to
    lowercase POSIX path -- so a finding produced on Windows and one
    produced on Linux for the same logical file dedupe together."""
    assert (
        _normalize_path_for_dedup("C:\\Users\\Fadhil\\file.py")
        == _normalize_path_for_dedup("/Users/Fadhil/file.py")
    )
