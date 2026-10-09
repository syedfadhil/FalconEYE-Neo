"""Unit tests for --confirm-taint on ``falconeye skill run`` / ``skill run-all``.

The commands are exercised end to end with a stub DI container: a
mock LLM stands in for the model, while the confirmation service is
the real one (C# adapter + ASP.NET Core entry-point detector), so the
reachability verdicts asserted here come from an actual call graph.

Coverage:
  * skill run --confirm-taint: findings enriched, none dropped
  * Without the flag: no confirmation runs (backwards compat)
  * No taint-enabled language -> warning, findings preserved
  * Confirmation service raises -> non-fatal, findings preserved
  * No findings -> confirmation not attempted
  * skill run-all --confirm-taint: confirmation runs after dedup
  * Typer wiring: --confirm-taint reaches both command functions
"""

from __future__ import annotations

import json
import textwrap
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from falconeye.adapters.cli import commands_skill
from falconeye.adapters.cli.commands_skill import (
    skill_run_all_command,
    skill_run_command,
)
from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.domain.models.confirmation import (
    CONFIRMATION_STATUS_NOT_CONFIRMED,
    CONFIRMATION_STATUS_REACHABLE,
)
from falconeye.domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)
from falconeye.infrastructure.taint_adapters.adapter_registry import (
    AdapterRegistry,
)
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


# ---------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------


# Line 5 sits inside an [HttpGet] action (reachable); line 9 sits
# inside a helper nothing calls (not_confirmed).
_CONTROLLER = textwrap.dedent("""\
    namespace N;
    public class UsersController {
        [HttpGet("/users/{id}")]
        public string Get(string id) {
            return db.ExecuteSqlRaw("SELECT * FROM U WHERE Id = " + id);
        }

        public string Helper(string id) {
            return db.ExecuteSqlRaw("DELETE FROM U WHERE Id = " + id);
        }
    }
    """)


def _skill_md(name: str) -> str:
    return textwrap.dedent(f"""\
        ---
        name: {name}
        description: test skill
        function: scan
        ---

        # {name}

        ## Step 1 -- Discover

        ```bash
        # raw SQL
        grep -rn -E "ExecuteSqlRaw" --include="*.cs" .
        ```

        ## Step 2 -- Evaluate

        Check for SQL built by string concatenation.
        """)


def _finding(line: int, issue: str) -> dict:
    return {
        "issue": issue,
        "severity": "High",
        "confidence": "Medium",
        "line": line,
        "description": "query built by concatenation",
        "recommended_fix": "use parameters",
        "cwe": "CWE-89",
        "category": "injection",
    }


class _MockLLM:
    """Returns the same canned findings for every file evaluated."""

    def __init__(self, findings):
        self._payload = json.dumps({"findings": findings})

    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        return self._payload


class _SpyService:
    """Wraps a confirmation service and records what it was asked."""

    def __init__(self, inner=None, error=None):
        self._inner = inner
        self._error = error
        self.calls = []

    def confirm(self, *, findings, repo_path, languages):
        self.calls.append(list(findings))
        if self._error is not None:
            raise self._error
        return self._inner.confirm(
            findings=findings, repo_path=repo_path, languages=languages,
        )


def _real_service():
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    return ReviewConfirmationService(
        adapter_registry=reg,
        entry_point_registry=entry_reg,
        taint_index=load_taint_index(languages=["csharp"]),
    )


def _config():
    """Only the fields _run_taint_confirmation reads."""
    return SimpleNamespace(languages=SimpleNamespace(
        enabled=["csharp"],
        per_language={"csharp": SimpleNamespace(taint=SimpleNamespace(enabled=True))},
    ))


def _install_container(monkeypatch, *, llm, service):
    """Make DIContainer.create() inside commands_skill return a stub."""
    container = SimpleNamespace(
        llm_service=llm,
        skill_loader=None,
        review_confirmation_service=service,
        config=_config(),
    )

    class _StubContainer:
        @classmethod
        def create(cls, **kwargs):
            return container

    monkeypatch.setattr(commands_skill, "DIContainer", _StubContainer)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "UsersController.cs").write_text(_CONTROLLER, encoding="utf-8")
    return ws


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "sqli-a.md").write_text(_skill_md("sqli-a"), encoding="utf-8")
    return d


def _console():
    buf = StringIO()
    return Console(file=buf, force_terminal=False, width=200), buf


def _run(workspace, skills_dir, out_file, *, confirm_taint):
    """Invoke `skill run sqli-a` with JSON output; return (findings, console text)."""
    console, buf = _console()
    skill_run_command(
        skill_name="sqli-a",
        workspace=workspace,
        output_format="json",
        output_file=out_file,
        config_path=None,
        skills_dir_override=skills_dir,
        console=console,
        confirm_taint=confirm_taint,
    )
    findings = json.loads(out_file.read_text(encoding="utf-8"))["findings"]
    return findings, buf.getvalue()


# ---------------------------------------------------------------
# skill run
# ---------------------------------------------------------------


@pytest.mark.unit
def test_skill_run_confirm_taint_enriches_without_dropping(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """Both findings come back. The one in the [HttpGet] action is
    confirmed reachable; the one in the uncalled helper is
    not_confirmed but keeps its severity (confirmation is additive)."""
    _install_container(
        monkeypatch,
        llm=_MockLLM([_finding(5, "in Get"), _finding(9, "in Helper")]),
        service=_real_service(),
    )
    findings, output = _run(
        workspace, skills_dir, tmp_path / "out.json", confirm_taint=True,
    )

    by_line = {f["line_start"]: f for f in findings}
    assert set(by_line) == {5, 9}
    assert by_line[5]["taint_confirmation"]["status"] == CONFIRMATION_STATUS_REACHABLE
    assert by_line[9]["taint_confirmation"]["status"] == CONFIRMATION_STATUS_NOT_CONFIRMED
    assert by_line[9]["severity"] == "high"
    assert "Taint confirmation" in output


@pytest.mark.unit
def test_skill_run_without_flag_does_not_confirm(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """Backwards compat: no flag means no confirmation call, and the
    JSON carries taint_confirmation: null exactly as before."""
    spy = _SpyService(inner=_real_service())
    _install_container(monkeypatch, llm=_MockLLM([_finding(5, "in Get")]), service=spy)
    findings, output = _run(
        workspace, skills_dir, tmp_path / "out.json", confirm_taint=False,
    )

    assert spy.calls == []
    assert findings[0]["taint_confirmation"] is None
    assert "Taint confirmation" not in output


@pytest.mark.unit
def test_skill_run_confirm_taint_without_service_warns(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """No taint-enabled language means the container has no
    confirmation service: warn, and ship the findings unenriched."""
    _install_container(monkeypatch, llm=_MockLLM([_finding(5, "in Get")]), service=None)
    findings, output = _run(
        workspace, skills_dir, tmp_path / "out.json", confirm_taint=True,
    )

    assert "no taint-enabled language is configured" in output
    assert len(findings) == 1
    assert findings[0]["taint_confirmation"] is None


@pytest.mark.unit
def test_skill_run_confirm_taint_failure_is_non_fatal(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """A crashing confirmation stage must not cost the operator the
    skill's findings: warn and output them unenriched."""
    spy = _SpyService(error=RuntimeError("call graph build blew up"))
    _install_container(monkeypatch, llm=_MockLLM([_finding(5, "in Get")]), service=spy)
    findings, output = _run(
        workspace, skills_dir, tmp_path / "out.json", confirm_taint=True,
    )

    assert len(spy.calls) == 1
    assert "Taint confirmation failed" in output
    assert "call graph build blew up" in output
    assert len(findings) == 1
    assert findings[0]["taint_confirmation"] is None


@pytest.mark.unit
def test_skill_run_confirm_taint_skips_when_no_findings(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """Nothing to confirm: don't build a call graph, don't print noise."""
    spy = _SpyService(inner=_real_service())
    _install_container(monkeypatch, llm=_MockLLM([]), service=spy)
    findings, output = _run(
        workspace, skills_dir, tmp_path / "out.json", confirm_taint=True,
    )

    assert findings == []
    assert spy.calls == []
    assert "Taint confirmation" not in output


# ---------------------------------------------------------------
# skill run-all
# ---------------------------------------------------------------


@pytest.mark.unit
def test_skill_run_all_confirms_after_dedup(
    workspace, skills_dir, tmp_path, monkeypatch,
):
    """Two skills flag the same line. Dedup collapses them first, so
    confirmation sees ONE finding, and the survivor carries both the
    merged skill tags and the reachability verdict."""
    (skills_dir / "sqli-b.md").write_text(_skill_md("sqli-b"), encoding="utf-8")
    spy = _SpyService(inner=_real_service())
    _install_container(monkeypatch, llm=_MockLLM([_finding(5, "in Get")]), service=spy)

    console, _ = _console()
    out_file = tmp_path / "all.json"
    skill_run_all_command(
        workspace=workspace,
        output_format="json",
        output_file=out_file,
        config_path=None,
        skills_dir_override=skills_dir,
        dedupe=True,
        console=console,
        confirm_taint=True,
    )
    findings = json.loads(out_file.read_text(encoding="utf-8"))["findings"]

    assert len(spy.calls) == 1
    assert len(spy.calls[0]) == 1  # deduped before confirmation ran
    assert len(findings) == 1
    assert {"skill:sqli-a", "skill:sqli-b"} <= set(findings[0]["tags"])
    assert findings[0]["taint_confirmation"]["status"] == CONFIRMATION_STATUS_REACHABLE


# ---------------------------------------------------------------
# Typer wiring
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("argv, target", [
    (["skill", "run", "sqli-a"], "skill_run_command"),
    (["skill", "run-all"], "skill_run_all_command"),
])
def test_cli_threads_confirm_taint_flag(argv, target, tmp_path, monkeypatch):
    """--confirm-taint on the typer surface reaches the command
    function; omitting it passes False."""
    from falconeye.adapters.cli import main as cli_main

    seen = []
    monkeypatch.setattr(
        cli_main, target, lambda **kw: seen.append(kw["confirm_taint"]),
    )
    runner = CliRunner()
    for extra in (["--confirm-taint"], []):
        result = runner.invoke(cli_main.app, argv + [str(tmp_path)] + extra)
        assert result.exit_code == 0, result.output
    assert seen == [True, False]
