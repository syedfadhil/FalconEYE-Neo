"""Unit tests for ``falconeye skill lint`` and the loader fixes shipped with it.

Skill markdown is written as raw strings, so each file holds exactly
the shell text a skill author would type; the loader then applies
shell unquoting as usual before the linter sees the regex.

Coverage:
  * Pattern checks: clean skill, BRE-style \\| alternation (and the
    literal \\|\\| it must not flag), Python-invalid regex with the
    bracket-backslash hint, POSIX classes, GNU word boundaries (and an
    escaped backslash it must not flag), empty-matching patterns
  * Step 1 commands the loader ignores, with the reason; unsupported
    search tools; grep commands outside Step 1; LLM-only skills
  * Frontmatter: unterminated, invalid YAML, not a mapping, missing
  * Metadata: unknown function, deferred verify, tool-invocation
    skill_type and sandbox execution.mode; duplicate names
  * Loader fixes: frontmatter behind a UTF-8 BOM, frontmatter closed
    at end of file
  * CLI: output, verbatim regexes, exit code 1 on errors only
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from falconeye.adapters.cli.commands_skill import skill_lint_command
from falconeye.infrastructure.skills.skill_linter import (
    LINT_ERROR,
    LINT_INFO,
    LINT_WARNING,
    lint_skill,
    lint_skills,
)
from falconeye.infrastructure.skills.skill_loader import SkillLoader


_FRONTMATTER = (
    "---\nname: s\ndescription: d\nfunction: scan\n"
    "skill_type: llm-reasoning\nexecution:\n  mode: local-llm\n  tier: T1\n---\n\n"
)


def _grep_skill(*lines: str) -> str:
    """A scan skill (example-library frontmatter) whose Step 1 holds ``lines``."""
    return (
        _FRONTMATTER
        + "## Step 1 -- Discover\n\n```bash\n# probe\n"
        + "\n".join(lines)
        + "\n```\n"
    )


def _lint(tmp_path: Path, text: str, name: str = "s"):
    (tmp_path / f"{name}.md").write_text(text, encoding="utf-8")
    skill = {s.name: s for s in SkillLoader(directory=tmp_path).load_all()}[name]
    return lint_skill(skill)


def _codes(issues):
    return [i.code for i in issues]


def _issue(issues, code):
    return next(i for i in issues if i.code == code)


# ---------------------------------------------------------------
# Discovery patterns
# ---------------------------------------------------------------


@pytest.mark.unit
def test_clean_skill_has_no_issues(tmp_path):
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "(cursor\.execute\(|FromSqlRaw\()" --include="*.{py,cs}" .',
    ))
    assert issues == []


@pytest.mark.unit
def test_bre_alternation_is_flagged(tmp_path):
    """The operator's deserialization-xxe shape: alternatives joined
    with \\| are literal pipes and never match."""
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "pickle\.loads|jsonpickle\.decode\|shelve\.open" --include="*.py" .',
    ))
    assert _codes(issues) == ["bre-alternation"]
    issue = issues[0]
    assert issue.level == LINT_WARNING
    assert issue.message.startswith("pattern [0] (probe):")
    assert r"jsonpickle\.decode\|shelve" in issue.message


@pytest.mark.unit
def test_literal_double_pipe_is_not_flagged(tmp_path):
    """Matching a logical-or (`||`) needs \\|\\|; that's deliberate."""
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "if .* \|\| " --include="*.py" .',
    ))
    assert issues == []


@pytest.mark.unit
def test_invalid_regex_is_an_error_with_bracket_hint(tmp_path):
    """`[/\\\\]` in a double-quoted grep reaches the regex as `[/\\]`:
    fine for POSIX grep (backslash is literal inside brackets), an
    unterminated set for Python -- so the pattern is dead at run time."""
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "\.\.[/\\]" --include="*.cs" .',
    ))
    issue = _issue(issues, "regex-invalid")
    assert issue.level == LINT_ERROR
    assert "skipped at run time" in issue.message
    assert "literal backslash" in issue.message


@pytest.mark.unit
def test_posix_class_is_flagged(tmp_path):
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "[[:alpha:]]+_KEY" --include="*.py" .',
    ))
    assert _codes(issues) == ["posix-class"]


@pytest.mark.unit
def test_gnu_word_boundary_is_flagged(tmp_path):
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "\<eval\>" --include="*.js" .',
    ))
    assert _codes(issues) == ["gnu-word-boundary"]


@pytest.mark.unit
def test_escaped_backslash_before_angle_is_not_flagged(tmp_path):
    """`C:\\\\<x` is a literal backslash followed by `<`, not `\\<`."""
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "C:\\\\<x" --include="*.cs" .',
    ))
    assert issues == []


@pytest.mark.unit
def test_pattern_matching_empty_string_is_flagged(tmp_path):
    issues = _lint(tmp_path, _grep_skill(
        r'grep -rn -E "(eval)?" --include="*.js" .',
    ))
    assert _codes(issues) == ["matches-everything"]


# ---------------------------------------------------------------
# Step 1 commands
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("line, reason", [
    (r'grep -rn "eval(" --include="*.js" .', "no -E flag"),
    (r'grep -rnE "eval\(" --include="*.js" .', "combined flags"),
    (r'grep -rn -E "eval\(" .', "no --include filter"),
    (r'grep -rn -E "eval\(" --include="src/**" .', "is not understood"),
    (r'grep -rn -E "eval\( --include="*.js" .', "cannot be parsed"),
])
def test_ignored_grep_commands_report_why(tmp_path, line, reason):
    issues = _lint(tmp_path, _grep_skill(line))
    issue = _issue(issues, "grep-ignored")
    assert issue.level == LINT_WARNING
    assert reason in issue.message
    # Nothing usable was extracted, so the skill is also LLM-only.
    assert "llm-only" in _codes(issues)


@pytest.mark.unit
def test_unsupported_search_command_is_flagged(tmp_path):
    issues = _lint(tmp_path, _grep_skill(r'rg -n "eval\(" src/'))
    assert _codes(issues) == ["unsupported-command", "llm-only"]


@pytest.mark.unit
def test_grep_outside_step1_is_flagged(tmp_path):
    text = _grep_skill(r'grep -rn -E "alpha" --include="*.py" .') + (
        "\n## Step 2 -- Evaluate\n\n```bash\n"
        'grep -rn -E "beta" --include="*.py" .\n'
        'grep -rn -E "gamma" --include="*.py" .\n'
        "```\n"
    )
    issues = _lint(tmp_path, text)
    assert _codes(issues) == ["grep-outside-step1"]
    assert issues[0].message.startswith("2 grep commands")


@pytest.mark.unit
@pytest.mark.parametrize("body, reason", [
    ("## Audit Workflow\n\n1. Inventory output points.\n", "no 'Step 1' heading"),
    ("## Step 1 -- Discover\n\nLocate and read:\n- handlers\n", "Step 1 has no"),
])
def test_llm_only_skill_reports_reason(tmp_path, body, reason):
    issues = _lint(tmp_path, _FRONTMATTER + body)
    assert _codes(issues) == ["llm-only"]
    assert issues[0].level == LINT_INFO
    assert reason in issues[0].message


# ---------------------------------------------------------------
# Frontmatter and metadata
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text, code, detail", [
    ("---\nname: x\nfunction: verify\n\n# body\n", "frontmatter-unterminated", "never closes"),
    ("---\nname: [unclosed\nfunction: verify\n---\n", "frontmatter-invalid", "not valid YAML"),
    ("---\n- a\n- b\n---\n", "frontmatter-invalid", "not a YAML mapping"),
])
def test_broken_frontmatter_is_an_error(tmp_path, text, code, detail):
    """Broken frontmatter silently turns a verify skill into a runnable
    scan skill named after its file -- an error, not a warning."""
    issues = _lint(tmp_path, text)
    issue = _issue(issues, code)
    assert issue.level == LINT_ERROR
    assert detail in issue.message
    assert "function=scan" in issue.message


@pytest.mark.unit
def test_missing_frontmatter_is_info(tmp_path):
    issues = _lint(tmp_path, "# Plain markdown skill\n")
    assert _issue(issues, "no-frontmatter").level == LINT_INFO


@pytest.mark.unit
def test_unknown_function_warns_and_is_not_llm_only(tmp_path):
    issues = _lint(tmp_path, "---\nname: s\ndescription: d\nfunction: scanner\n---\n")
    assert _codes(issues) == ["function-unknown"]


@pytest.mark.unit
def test_verify_skill_is_reported_as_deferred_only(tmp_path):
    issues = _lint(tmp_path, "---\nname: s\ndescription: d\nfunction: verify\n---\n")
    assert _codes(issues) == ["deferred"]
    assert "fp-check" in issues[0].message


@pytest.mark.unit
def test_tool_invocation_and_sandbox_mode_warn(tmp_path):
    text = _grep_skill(r'grep -rn -E "x" --include="*.py" .').replace(
        "skill_type: llm-reasoning", "skill_type: tool-invocation",
    ).replace("mode: local-llm", "mode: local-sandbox")
    issues = _lint(tmp_path, text)
    assert _codes(issues) == ["skill-type", "execution-mode"]


@pytest.mark.unit
def test_duplicate_names_are_an_error(tmp_path):
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.md").write_text(
            "---\nname: dup\ndescription: d\nfunction: verify\n---\n", encoding="utf-8",
        )
    issues = lint_skills(SkillLoader(directory=tmp_path).load_all())
    dup = _issue(issues, "duplicate-name")
    assert dup.level == LINT_ERROR
    assert dup.skill == "dup"
    assert "2 skills share this name" in dup.message


# ---------------------------------------------------------------
# Loader fixes
# ---------------------------------------------------------------


@pytest.mark.unit
def test_loader_keeps_frontmatter_behind_a_bom(tmp_path):
    """A BOM (common from Windows editors) used to hide the frontmatter:
    this fp-check skill loaded as runnable scan skill 'fp'."""
    (tmp_path / "fp.md").write_bytes(
        "\ufeff---\nname: fp-check\nfunction: verify\n---\n\n# body\n".encode("utf-8")
    )
    skill = SkillLoader(directory=tmp_path).load_all()[0]
    assert skill.name == "fp-check"
    assert skill.metadata.function == "verify"
    assert skill.is_runnable_today is False


@pytest.mark.unit
def test_loader_accepts_frontmatter_closed_at_end_of_file(tmp_path):
    (tmp_path / "t.md").write_text(
        "---\nname: triage\nfunction: triage\n---", encoding="utf-8",
    )
    skill = SkillLoader(directory=tmp_path).load_all()[0]
    assert skill.name == "triage"
    assert skill.metadata.function == "triage"
    assert skill.body == ""


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def _run_lint(skills_dir):
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=500)
    skill_lint_command(config_path=None, console=console, skills_dir_override=skills_dir)
    return buf.getvalue()


@pytest.mark.unit
def test_lint_command_exits_1_on_errors(tmp_path):
    (tmp_path / "bad.md").write_text(
        _grep_skill(r'grep -rn -E "\.\.[/\\]" --include="*.cs" .').replace("name: s", "name: bad"),
        encoding="utf-8",
    )
    (tmp_path / "good.md").write_text(
        _grep_skill(r'grep -rn -E "x" --include="*.py" .').replace("name: s", "name: good"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        _run_lint(tmp_path)
    assert exc.value.code == 1


@pytest.mark.unit
def test_lint_command_output_and_exit_0_on_warnings(tmp_path):
    """Warnings alone don't fail the command; regexes print verbatim."""
    (tmp_path / "xxe.md").write_text(
        _grep_skill(
            r'grep -rn -E "pickle\.loads|jsonpickle\.decode\|shelve\.open" --include="*.py" .',
        ).replace("name: s", "name: xxe"),
        encoding="utf-8",
    )
    (tmp_path / "good.md").write_text(
        _grep_skill(r'grep -rn -E "x" --include="*.py" .').replace("name: s", "name: good"),
        encoding="utf-8",
    )
    out = _run_lint(tmp_path)
    assert "WARNING bre-alternation" in out
    assert r"jsonpickle\.decode\|shelve" in out
    assert "good\n  ok" in out
    assert "2 skills: 0 error(s), 1 warning(s), 0 info" in out
