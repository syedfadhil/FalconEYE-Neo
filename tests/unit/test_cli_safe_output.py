"""Unit tests for markup-safe CLI output.

LLM-written finding text, code snippets, skill regexes and exception
messages all reach the terminal. Printed through Rich's markup parser
they get mangled (``args[name]`` reads as a style tag and vanishes),
crash the command (``a[/b]`` raises MarkupError), turn into emoji
(``:warning:``), or get re-wrapped (long JSON lines split mid-string).

Coverage:
  * skill run -o json to stdout: JSON intact on a narrow, piped console
  * skill run console report: bracketed LLM text rendered literally
  * skill run summary: per-file error text with brackets doesn't crash
  * skill run-all batch summary: errored-skill reason rendered literally
  * skill list --show-patterns: regexes shown verbatim, no MarkupError
  * taint-confirm console: status label visible, LLM text literal
  * taint-confirm -o json to stdout: JSON intact
  * --confirm-taint failure message with brackets doesn't crash
"""

from __future__ import annotations

import json
import textwrap
from io import StringIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from rich.console import Console

from falconeye.adapters.cli import commands_skill, commands_taint_confirm
from falconeye.adapters.cli.commands import _run_taint_confirmation
from falconeye.adapters.cli.commands_skill import (
    _print_batch_summary,
    skill_list_command,
    skill_run_command,
)
from falconeye.adapters.cli.commands_taint_confirm import taint_confirm_command
from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.application.services.skill_runner_service import (
    BATCH_STATUS_ERROR,
    SkillBatchOutcome,
    SkillBatchSummary,
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


#: Text that Rich's markup parser mishandles in each of the known ways,
#: plus a line long enough to be re-wrapped on an 80-column console.
HOSTILE = "items[i] = cache[key] -> a[/b] :warning: " + "x" * 200

_CONTROLLER = textwrap.dedent("""\
    namespace N;
    public class UsersController {
        [HttpGet("/users/{id}")]
        public string Get(string id) {
            return db.ExecuteSqlRaw("SELECT * FROM U WHERE Id = " + id);
        }
    }
    """)

_SKILL = textwrap.dedent("""\
    ---
    name: sqli
    description: test skill
    function: scan
    ---

    # sqli

    ## Step 1 -- Discover

    ```bash
    # raw SQL
    grep -rn -E "ExecuteSqlRaw" --include="*.cs" .
    ```
    """)


class _MockLLM:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        if self._error is not None:
            raise self._error
        return json.dumps(self._payload)


def _hostile_findings():
    return {"findings": [{
        "issue": "Unsanitized input in args[name]",
        "severity": "High",
        "confidence": "Medium",
        "line": 5,
        "description": HOSTILE,
        "recommended_fix": "use [b]parameters[/b]",
        "category": "injection",
    }]}


def _install_skill_container(monkeypatch, llm):
    container = SimpleNamespace(
        llm_service=llm, skill_loader=None,
        review_confirmation_service=None, config=None,
    )

    class _Stub:
        @classmethod
        def create(cls, **kwargs):
            return container

    monkeypatch.setattr(commands_skill, "DIContainer", _Stub)


def _piped_console(width=80):
    """A non-terminal console, like stdout redirected to a file or pipe."""
    buf = StringIO()
    return Console(file=buf, force_terminal=False, width=width), buf


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
    (d / "sqli.md").write_text(_SKILL, encoding="utf-8")
    return d


def _run_skill(workspace, skills_dir, output_format, width=80):
    console, buf = _piped_console(width=width)
    skill_run_command(
        skill_name="sqli",
        workspace=workspace,
        output_format=output_format,
        output_file=None,
        config_path=None,
        skills_dir_override=skills_dir,
        console=console,
    )
    return buf.getvalue()


# ---------------------------------------------------------------
# skill run / run-all
# ---------------------------------------------------------------


@pytest.mark.unit
def test_skill_run_json_stdout_survives_narrow_pipe(workspace, skills_dir, monkeypatch):
    """The JSON block on stdout parses, and hostile LLM text inside it
    round-trips byte for byte (no wrapping, markup or emoji)."""
    _install_skill_container(monkeypatch, _MockLLM(_hostile_findings()))
    out = _run_skill(workspace, skills_dir, "json")

    payload = json.loads(out[out.index('{\n  "codebase_path"'):])
    finding = payload["findings"][0]
    assert finding["reasoning"] == HOSTILE
    assert finding["issue"] == "Unsanitized input in args[name]"
    assert finding["mitigation"] == "use [b]parameters[/b]"


@pytest.mark.unit
def test_skill_run_console_report_renders_brackets_literally(
    workspace, skills_dir, monkeypatch,
):
    """Before the fix, ``args[name]`` was swallowed as a style tag and
    ``a[/b]`` in the description raised MarkupError."""
    _install_skill_container(monkeypatch, _MockLLM(_hostile_findings()))
    out = _run_skill(workspace, skills_dir, "console")

    assert "Unsanitized input in args[name]" in out
    # The description is re-wrapped by ConsoleFormatter itself, so
    # check the fragments rather than one long span.
    for fragment in ("items[i]", "cache[key]", "a[/b]", ":warning:"):
        assert fragment in out


@pytest.mark.unit
def test_skill_run_summary_prints_bracketed_error_text(
    workspace, skills_dir, monkeypatch,
):
    """Per-file LLM errors are echoed in the run summary; exception
    text containing a closing-tag lookalike must not crash it."""
    _install_skill_container(
        monkeypatch, _MockLLM(error=RuntimeError("upstream said [/oops]")),
    )
    out = _run_skill(workspace, skills_dir, "console", width=500)

    assert "upstream said [/oops]" in out


@pytest.mark.unit
def test_batch_summary_prints_error_reason_literally():
    console, buf = _piped_console(width=200)
    batch = SkillBatchSummary(total_skills_seen=1, skills_errored=1)
    batch.outcomes.append(SkillBatchOutcome(
        skill_name="weird[skill]",
        status=BATCH_STATUS_ERROR,
        reason="RuntimeError: regex [/\\] blew up",
    ))
    _print_batch_summary(console, batch, dedupe=True)

    out = buf.getvalue()
    assert "weird[skill]" in out
    assert "RuntimeError: regex [/\\] blew up" in out


# ---------------------------------------------------------------
# skill list --show-patterns
# ---------------------------------------------------------------


@pytest.mark.unit
def test_show_patterns_prints_regexes_verbatim(tmp_path):
    """A ``[/]`` class reads as a closing tag (MarkupError), an escaped
    ``\\[`` used to lose its backslash, and a ``[segments]`` comment
    read as a style tag and vanished."""
    d = tmp_path / "skills"
    d.mkdir()
    (d / "paths.md").write_text(textwrap.dedent("""\
        ---
        name: paths
        description: test
        ---

        ## Step 1 -- Discover

        ```bash
        # dot-dot [segments]
        grep -rn -E "(\\.\\./|[/]etc[/]passwd)" --include="*.cs" .
        # header writes
        grep -rn -E "headers\\[" --include="*.cs" .
        ```
        """), encoding="utf-8")

    console, buf = _piped_console(width=200)
    skill_list_command(
        config_path=None, console=console,
        show_patterns=True, skills_dir_override=d,
    )
    out = buf.getvalue()
    assert r"(\.\./|[/]etc[/]passwd)" in out
    assert r"headers\[" in out
    assert "dot-dot [segments]" in out


# ---------------------------------------------------------------
# taint-confirm
# ---------------------------------------------------------------


def _install_confirm_container(monkeypatch):
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    service = ReviewConfirmationService(
        adapter_registry=reg,
        entry_point_registry=entry_reg,
        taint_index=load_taint_index(languages=["csharp"]),
    )
    container = SimpleNamespace(review_confirmation_service=service)

    class _Stub:
        @classmethod
        def create(cls, **kwargs):
            return container

    monkeypatch.setattr(commands_taint_confirm, "DIContainer", _Stub)


def _findings_file(tmp_path, workspace):
    path = tmp_path / "findings.json"
    path.write_text(json.dumps({"findings": [{
        "id": str(uuid4()),
        "issue": "Unsanitized input in args[name]",
        "reasoning": HOSTILE,
        "mitigation": "",
        "severity": "high",
        "confidence": "medium",
        "file_path": str(workspace / "UsersController.cs"),
        "code_snippet": "",
        "line_start": 5,
    }]}), encoding="utf-8")
    return path


@pytest.mark.unit
def test_taint_confirm_console_shows_status_label_and_literal_issue(
    workspace, tmp_path, monkeypatch,
):
    """The ``[confirmed_reachable]`` label is lowercase, so Rich used to
    read it as an (unknown) style tag and drop it from every line."""
    _install_confirm_container(monkeypatch)
    console, buf = _piped_console(width=500)
    taint_confirm_command(
        path=workspace,
        findings_file=_findings_file(tmp_path, workspace),
        output_format="console",
        output_file=None,
        languages=["csharp"],
        config_path=None,
        console=console,
    )
    out = buf.getvalue()
    assert "[confirmed_reachable]" in out
    assert "Unsanitized input in args[name]" in out


@pytest.mark.unit
def test_taint_confirm_json_stdout_survives_narrow_pipe(
    workspace, tmp_path, monkeypatch,
):
    _install_confirm_container(monkeypatch)
    console, buf = _piped_console()
    taint_confirm_command(
        path=workspace,
        findings_file=_findings_file(tmp_path, workspace),
        output_format="json",
        output_file=None,
        languages=["csharp"],
        config_path=None,
        console=console,
    )
    out = buf.getvalue()
    records = json.loads(out[out.index("[\n  {"):])
    assert records[0]["reasoning"] == HOSTILE
    assert records[0]["taint_confirmation"]["status"] == "confirmed_reachable"


@pytest.mark.unit
def test_confirmation_failure_message_with_brackets_does_not_crash(tmp_path):
    class _Broken:
        def confirm(self, **kwargs):
            raise RuntimeError("bad node [/Ns.Ctrl]")

    review = SimpleNamespace(findings=[object()])
    config = SimpleNamespace(languages=SimpleNamespace(
        enabled=["csharp"],
        per_language={"csharp": SimpleNamespace(taint=SimpleNamespace(enabled=True))},
    ))
    console, buf = _piped_console(width=200)
    _run_taint_confirmation(
        review=review, repo_path=tmp_path, confirmation_service=_Broken(),
        config=config, console=console,
    )
    assert "bad node [/Ns.Ctrl]" in buf.getvalue()
