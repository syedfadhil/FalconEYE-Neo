"""Unit tests for sweeping LLM-only skills (skills without Step 1 patterns).

Coverage:
  * Service: each candidate file evaluated once, skip fragments, the
    file cap (count + warning), ``targets`` globs selecting manifests,
    whole-file prompt wording, per-file failure isolation, discovery
    skills unaffected, ValueError without ``sweep_extensions``
  * run_all: LLM-only skills skipped by default with a reason naming
    the flag; swept with include_llm_only
  * sweep_extensions_for, ``targets`` parsing, iter_workspace_files
    name patterns, SkillsConfig.max_llm_only_files bounds
  * Lint: targets-invalid, targets-ignored, llm-only message
  * CLI: cost preview and cap message, config-driven languages and
    cap, run-all with/without --include-llm-only, typer wiring
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from falconeye.adapters.cli import commands_skill
from falconeye.adapters.cli.commands_skill import (
    skill_run_all_command,
    skill_run_command,
)
from falconeye.application.services.skill_runner_service import (
    BATCH_STATUS_RAN,
    BATCH_STATUS_SKIPPED_NO_PATTERNS,
    RUN_MODE_DISCOVERY,
    RUN_MODE_SWEEP,
    SkillRunnerService,
    sweep_extensions_for,
)
from falconeye.infrastructure.config.config_models import SkillsConfig
from falconeye.infrastructure.skills.skill_linter import lint_skill
from falconeye.infrastructure.skills.skill_loader import SkillLoader
from falconeye.infrastructure.workspace_files import iter_workspace_files


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


_FILE_IN_PROMPT = re.compile(r"## File\n\n`([^`]+)`")


class _RecordingLLM:
    """Returns one finding per file, named after the file it was shown.

    ``fail_on`` names files whose call raises, to exercise isolation.
    """

    def __init__(self, fail_on=()):
        self.files = []
        self.prompts = []
        self._fail_on = set(fail_on)

    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        path = _FILE_IN_PROMPT.search(user_prompt).group(1)
        name = Path(path).name
        self.files.append(name)
        self.prompts.append(user_prompt)
        if name in self._fail_on:
            raise RuntimeError(f"backend down for {name}")
        return json.dumps({"findings": [{
            "issue": f"issue in {name}",
            "severity": "Medium",
            "confidence": "Low",
            "line": 1,
            "description": "d",
        }]})


def _touch(path: Path, content: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_skill(skills_dir: Path, name: str, *, targets=None, grep=None) -> None:
    lines = ["---", f"name: {name}", "description: test", "function: scan"]
    if targets is not None:
        lines.append(f"targets: {json.dumps(targets)}")
    lines += ["---", "", f"# {name}", "", "## Step 1 -- Discover", ""]
    if grep:
        lines += ["```bash", grep, "```"]
    else:
        lines += ["Locate and read request handlers."]
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load(skills_dir: Path, name: str):
    return {s.name: s for s in SkillLoader(directory=skills_dir).load_all()}[name]


def _run(llm, skill, workspace, **kwargs):
    return asyncio.run(SkillRunnerService(llm_service=llm).run(
        skill=skill, workspace=workspace, **kwargs,
    ))


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    _write_skill(d, "llm-only")
    return d


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    _touch(ws / "a.py")
    _touch(ws / "b.py")
    _touch(ws / "web" / "app.js", "let x = 1;\n")
    _touch(ws / "README.md", "# readme\n")
    return ws


# ---------------------------------------------------------------
# Service: sweep
# ---------------------------------------------------------------


@pytest.mark.unit
def test_sweep_evaluates_each_matching_file_once(skills_dir, workspace):
    llm = _RecordingLLM()
    review, summary = _run(
        llm, _load(skills_dir, "llm-only"), workspace, sweep_extensions={".py"},
    )
    assert llm.files == ["a.py", "b.py"]
    assert summary.mode == RUN_MODE_SWEEP
    assert summary.files_evaluated == 2
    assert summary.files_not_evaluated == 0
    assert summary.total_discovery_hits == 0
    # The model omitted "file", so each finding is attributed to the
    # file it was shown.
    assert sorted(Path(f.file_path).name for f in review.findings) == ["a.py", "b.py"]
    assert all("skill:llm-only" in f.tags for f in review.findings)


@pytest.mark.unit
def test_sweep_skips_default_excluded_directories(skills_dir, tmp_path):
    ws = tmp_path / "ws"
    _touch(ws / "src" / "keep.py")
    _touch(ws / "node_modules" / "pkg" / "drop.py")
    _touch(ws / "build" / "drop2.py")
    llm = _RecordingLLM()
    _run(llm, _load(skills_dir, "llm-only"), ws, sweep_extensions={".py"})
    assert llm.files == ["keep.py"]


@pytest.mark.unit
def test_sweep_cap_counts_files_not_evaluated_and_warns(skills_dir, tmp_path, caplog):
    ws = tmp_path / "ws"
    for name in ("e.py", "d.py", "c.py", "b.py", "a.py"):
        _touch(ws / name)
    llm = _RecordingLLM()
    with caplog.at_level(logging.WARNING):
        _, summary = _run(
            llm, _load(skills_dir, "llm-only"), ws,
            sweep_extensions={".py"}, max_sweep_files=2,
        )
    assert llm.files == ["a.py", "b.py"]  # sorted, so the same two every run
    assert summary.files_evaluated == 2
    assert summary.files_not_evaluated == 3
    assert "capped at 2 of 5 files" in caplog.text


@pytest.mark.unit
def test_targets_select_manifests_instead_of_source(tmp_path):
    """A dependency-audit skill reaches manifests, not every .cs file;
    targets win even when sweep_extensions are passed."""
    skills = tmp_path / "skills"
    _write_skill(skills, "deps", targets=["package.json", "*.csproj"])
    ws = tmp_path / "ws"
    _touch(ws / "package.json", "{}\n")
    _touch(ws / "src" / "App.csproj", "<Project/>\n")
    _touch(ws / "src" / "Program.cs", "class P {}\n")
    _touch(ws / "node_modules" / "left-pad" / "package.json", "{}\n")
    llm = _RecordingLLM()
    _run(llm, _load(skills, "deps"), ws, sweep_extensions={".cs"})
    assert llm.files == ["package.json", "App.csproj"]


@pytest.mark.unit
def test_sweep_prompt_asks_for_whole_file_review(skills_dir, workspace):
    llm = _RecordingLLM()
    _run(llm, _load(skills_dir, "llm-only"), workspace, sweep_extensions={".py"})
    assert "no discovery pre-filter; review the whole file" in llm.prompts[0]


@pytest.mark.unit
def test_sweep_failure_on_one_file_does_not_stop_the_rest(skills_dir, workspace):
    llm = _RecordingLLM(fail_on={"a.py"})
    review, summary = _run(
        llm, _load(skills_dir, "llm-only"), workspace, sweep_extensions={".py"},
    )
    assert llm.files == ["a.py", "b.py"]
    assert summary.files_with_llm_errors == 1
    assert [Path(f.file_path).name for f in review.findings] == ["b.py"]
    failed = next(o for o in summary.outcomes if o.error)
    assert "backend down for a.py" in failed.error


@pytest.mark.unit
def test_pattern_less_skill_without_sweep_extensions_still_raises(skills_dir, workspace):
    """Direct callers must opt in; nothing starts a sweep by accident."""
    with pytest.raises(ValueError, match="no Step 1 discovery patterns"):
        _run(_RecordingLLM(), _load(skills_dir, "llm-only"), workspace)


@pytest.mark.unit
def test_discovery_skill_ignores_sweep_extensions(tmp_path, workspace):
    skills = tmp_path / "skills"
    _write_skill(skills, "grep", grep='grep -rn -E "x = 1" --include="*.py" .')
    (workspace / "b.py").write_text("y = 2\n", encoding="utf-8")
    llm = _RecordingLLM()
    _, summary = _run(llm, _load(skills, "grep"), workspace, sweep_extensions={".py", ".js"})
    assert summary.mode == RUN_MODE_DISCOVERY
    assert llm.files == ["a.py"]


# ---------------------------------------------------------------
# Service: run_all
# ---------------------------------------------------------------


def _run_all(llm, skills, workspace, **kwargs):
    return asyncio.run(SkillRunnerService(llm_service=llm).run_all(
        skills=skills, workspace=workspace, **kwargs,
    ))


@pytest.mark.unit
def test_run_all_skips_llm_only_by_default_with_flag_in_reason(skills_dir, workspace):
    llm = _RecordingLLM()
    review, batch = _run_all(
        llm, [_load(skills_dir, "llm-only")], workspace, sweep_extensions={".py"},
    )
    outcome = batch.outcomes[0]
    assert outcome.status == BATCH_STATUS_SKIPPED_NO_PATTERNS
    assert "--include-llm-only" in outcome.reason
    assert llm.files == []
    assert review.findings == []


@pytest.mark.unit
def test_run_all_sweeps_llm_only_skills_when_included(skills_dir, workspace):
    llm = _RecordingLLM()
    review, batch = _run_all(
        llm, [_load(skills_dir, "llm-only")], workspace,
        include_llm_only=True, sweep_extensions={".py"}, max_sweep_files=1,
    )
    from falconeye.adapters.cli.commands_skill import _print_batch_summary
    console, buf = _console()
    _print_batch_summary(console, batch, dedupe=True)
    assert "(sweep: 1 file, 1 over cap)" in buf.getvalue()
    outcome = batch.outcomes[0]
    assert outcome.status == BATCH_STATUS_RAN
    assert outcome.summary.mode == RUN_MODE_SWEEP
    assert outcome.summary.files_not_evaluated == 1
    assert llm.files == ["a.py"]
    assert batch.skills_ran == 1 and len(review.findings) == 1


@pytest.mark.unit
def test_run_all_include_without_extensions_stays_skipped(skills_dir, workspace):
    _, batch = _run_all(
        _RecordingLLM(), [_load(skills_dir, "llm-only")], workspace,
        include_llm_only=True,
    )
    assert batch.outcomes[0].status == BATCH_STATUS_SKIPPED_NO_PATTERNS


# ---------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------


@pytest.mark.unit
def test_sweep_extensions_for_languages():
    assert sweep_extensions_for(["csharp", "Python", "cobol"]) == {
        ".cs", ".csx", ".cshtml", ".razor", ".py",
    }


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ('["package.json", "*.csproj"]', ("package.json", "*.csproj")),
    ('"requirements*.txt"', ("requirements*.txt",)),
    ('["ok.json", 3, "", "  "]', ("ok.json",)),
    ("{a: b}", ()),
    (None, ()),
])
def test_targets_parsing(tmp_path, raw, expected):
    targets_line = f"targets: {raw}\n" if raw is not None else ""
    (tmp_path / "s.md").write_text(
        f"---\nname: s\ndescription: d\n{targets_line}---\n", encoding="utf-8",
    )
    assert _load(tmp_path, "s").metadata.targets == expected


@pytest.mark.unit
def test_walker_name_patterns_are_case_insensitive_and_add_to_extensions(tmp_path):
    _touch(tmp_path / "PACKAGE.JSON")
    _touch(tmp_path / "Program.cs")
    _touch(tmp_path / "other.json")
    found = [p.name for p in iter_workspace_files(
        tmp_path, extensions={".cs"}, name_patterns=("package.json",),
    )]
    assert sorted(found) == ["PACKAGE.JSON", "Program.cs"]
    only_globs = [p.name for p in iter_workspace_files(
        tmp_path, extensions=(), name_patterns=("*.json",),
    )]
    assert sorted(only_globs) == ["PACKAGE.JSON", "other.json"]


@pytest.mark.unit
def test_skills_config_llm_only_file_cap():
    assert SkillsConfig().max_llm_only_files == 50
    for bad in (0, 2001):
        with pytest.raises(ValidationError):
            SkillsConfig(max_llm_only_files=bad)


# ---------------------------------------------------------------
# Lint
# ---------------------------------------------------------------


def _lint_codes(skills, name):
    return [i.code for i in lint_skill(_load(skills, name))]


@pytest.mark.unit
def test_lint_reports_invalid_targets(tmp_path):
    (tmp_path / "s.md").write_text(
        "---\nname: s\ndescription: d\ntargets: [3, package.json]\n---\n",
        encoding="utf-8",
    )
    issues = lint_skill(_load(tmp_path, "s"))
    bad = next(i for i in issues if i.code == "targets-invalid")
    assert "['package.json']" in bad.message


@pytest.mark.unit
def test_lint_reports_targets_ignored_on_discovery_skill(tmp_path):
    _write_skill(
        tmp_path, "g", targets=["*.json"],
        grep='grep -rn -E "x" --include="*.py" .',
    )
    assert _lint_codes(tmp_path, "g") == ["targets-ignored"]


@pytest.mark.unit
def test_lint_llm_only_message_describes_the_sweep(tmp_path):
    _write_skill(tmp_path, "deps", targets=["package.json"])
    issue = next(i for i in lint_skill(_load(tmp_path, "deps")) if i.code == "llm-only")
    assert "sweeps the files matching targets (package.json)" in issue.message
    assert "--include-llm-only" in issue.message


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def _install_container(monkeypatch, llm, *, languages=("python",), cap=None):
    config = SimpleNamespace(
        languages=SimpleNamespace(enabled=list(languages), per_language={}),
        skills=SimpleNamespace(max_llm_only_files=cap) if cap else None,
    )
    container = SimpleNamespace(
        llm_service=llm, skill_loader=None,
        review_confirmation_service=None, config=config,
    )

    class _Stub:
        @classmethod
        def create(cls, **kwargs):
            return container

    monkeypatch.setattr(commands_skill, "DIContainer", _Stub)


def _console():
    buf = StringIO()
    return Console(file=buf, force_terminal=False, width=300), buf


@pytest.mark.unit
def test_cli_skill_run_sweeps_with_preview_and_cap(skills_dir, tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    for name in ("a.py", "b.py", "c.py"):
        _touch(ws / name)
    llm = _RecordingLLM()
    _install_container(monkeypatch, llm)
    console, buf = _console()
    out_file = tmp_path / "out.json"
    skill_run_command(
        skill_name="llm-only", workspace=ws, output_format="json",
        output_file=out_file, config_path=None, skills_dir_override=skills_dir,
        console=console, max_files=2,
    )
    out = buf.getvalue()
    assert "LLM-only skill: evaluating 2 of 3 files" in out
    assert "--max-files 2" in out
    assert "sweep: 2 files evaluated, 1 over the --max-files cap" in out
    assert "1 file not evaluated" in out
    assert len(json.loads(out_file.read_text(encoding="utf-8"))["findings"]) == 2


@pytest.mark.unit
def test_cli_sweep_follows_config_languages_and_cap(skills_dir, tmp_path, monkeypatch):
    """Only enabled languages are swept, and config's cap applies when
    --max-files isn't given."""
    ws = tmp_path / "ws"
    for name in ("A.cs", "B.cs", "C.cs", "main.py"):
        _touch(ws / name)
    llm = _RecordingLLM()
    _install_container(monkeypatch, llm, languages=("csharp",), cap=2)
    console, _ = _console()
    skill_run_command(
        skill_name="llm-only", workspace=ws, output_format="json",
        output_file=tmp_path / "o.json", config_path=None,
        skills_dir_override=skills_dir, console=console,
    )
    assert llm.files == ["A.cs", "B.cs"]


@pytest.mark.unit
@pytest.mark.parametrize("include, expect_text, expect_files", [
    (False, "skipped; pass --include-llm-only", []),
    (True, "swept file by file: up to 50 LLM calls each", ["a.py", "b.py"]),
])
def test_cli_run_all_include_llm_only(
    skills_dir, workspace, tmp_path, monkeypatch, include, expect_text, expect_files,
):
    llm = _RecordingLLM()
    _install_container(monkeypatch, llm)
    console, buf = _console()
    skill_run_all_command(
        workspace=workspace, output_format="json", output_file=tmp_path / "all.json",
        config_path=None, skills_dir_override=skills_dir, dedupe=True,
        console=console, include_llm_only=include,
    )
    out = buf.getvalue()
    assert expect_text in out
    assert llm.files == expect_files
    if include:
        assert "(sweep: 2 files)" in out


@pytest.mark.unit
@pytest.mark.parametrize("argv, target, key, value", [
    (["skill", "run", "x", "{ws}", "--max-files", "7"], "skill_run_command", "max_files", 7),
    (["skill", "run", "x", "{ws}"], "skill_run_command", "max_files", None),
    (["skill", "run-all", "{ws}", "--include-llm-only"], "skill_run_all_command", "include_llm_only", True),
    (["skill", "run-all", "{ws}"], "skill_run_all_command", "include_llm_only", False),
    (["skill", "run-all", "{ws}", "--max-files", "3"], "skill_run_all_command", "max_files", 3),
])
def test_cli_wiring(argv, target, key, value, tmp_path, monkeypatch):
    from falconeye.adapters.cli import main as cli_main

    seen = {}
    monkeypatch.setattr(cli_main, target, lambda **kw: seen.update(kw))
    args = [a.replace("{ws}", str(tmp_path)) for a in argv]
    result = CliRunner().invoke(cli_main.app, args)
    assert result.exit_code == 0, result.output
    assert seen[key] == value
