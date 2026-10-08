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
