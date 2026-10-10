"""Unit tests for logging in the skill and taint commands.

Coverage:
  * configure_cli_logging: no logging section -> untouched; non-verbose
    is file only; verbose adds the console
  * skill run: an LLM failure warning lands in the log file, not on
    stderr (it used to print on the console and write no file)
  * taint-confirm: confirmation logs land in the log file
  * -v reaches skill run, skill run-all, taint and taint-confirm
"""

from __future__ import annotations

import json
import logging
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from rich.console import Console
from typer.testing import CliRunner

from falconeye.adapters.cli import commands_skill, commands_taint_confirm
from falconeye.adapters.cli.commands_skill import skill_run_command
from falconeye.adapters.cli.commands_taint_confirm import taint_confirm_command
from falconeye.adapters.cli.logging_setup import configure_cli_logging
from falconeye.application.services.review_confirmation_service import (
    ReviewConfirmationService,
)
from falconeye.domain.services.entry_point_detector import (
    AspNetCoreEntryPointDetector,
    EntryPointDetectorRegistry,
)
from falconeye.infrastructure.logging import FalconEyeLogger
from falconeye.infrastructure.taint_adapters.adapter_registry import AdapterRegistry
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


@pytest.fixture
def shared_logger():
    """The process-wide logger, with its handlers restored afterwards."""
    lg = FalconEyeLogger.get_instance()
    saved_handlers, saved_level = list(lg.logger.handlers), lg.logger.level
    yield lg
    for handler in list(lg.logger.handlers):
        if handler not in saved_handlers:
            handler.close()
    lg.logger.handlers[:] = saved_handlers
    lg.logger.setLevel(saved_level)


def _logging_config(log_file: Path):
    return SimpleNamespace(level="INFO", file=str(log_file), console=True,
                           rotation="none", retention_days=30)


def _records(lg, log_file):
    for handler in lg.logger.handlers:
        handler.flush()
    return [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]


def _stub_container(monkeypatch, module, **attrs):
    container = SimpleNamespace(**attrs)
    monkeypatch.setattr(
        module, "DIContainer",
        type("Stub", (), {"create": classmethod(lambda cls, **kw: container)}),
    )


def _kinds(lg):
    return sorted(type(h).__name__ for h in lg.logger.handlers)


# ---------------------------------------------------------------
# configure_cli_logging
# ---------------------------------------------------------------


@pytest.mark.unit
def test_config_without_logging_section_is_left_alone(shared_logger):
    before = list(shared_logger.logger.handlers)
    configure_cli_logging(SimpleNamespace(), verbose=True)
    configure_cli_logging(None, verbose=False)
    assert shared_logger.logger.handlers == before


@pytest.mark.unit
@pytest.mark.parametrize("verbose, kinds", [
    (False, ["FileHandler"]),
    (True, ["FileHandler", "StreamHandler"]),
])
def test_verbose_controls_the_console(shared_logger, tmp_path, verbose, kinds):
    configure_cli_logging(SimpleNamespace(logging=_logging_config(tmp_path / "f.log")), verbose)
    assert _kinds(shared_logger) == kinds


# ---------------------------------------------------------------
# skill run / taint-confirm write the log file
# ---------------------------------------------------------------


class _FailingLLM:
    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        raise RuntimeError("backend unavailable")


@pytest.mark.unit
def test_skill_run_logs_to_file_not_stderr(shared_logger, tmp_path, monkeypatch, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("danger()\n", encoding="utf-8")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "s.md").write_text(textwrap.dedent("""\
        ---
        name: s
        description: d
        ---

        ## Step 1

        ```bash
        grep -rn -E "danger" --include="*.py" .
        ```
        """), encoding="utf-8")
    log_file = tmp_path / "falconeye.log"
    _stub_container(
        monkeypatch, commands_skill,
        llm_service=_FailingLLM(), skill_loader=None, review_confirmation_service=None,
        config=SimpleNamespace(logging=_logging_config(log_file), skills=None),
    )
    skill_run_command(
        skill_name="s", workspace=ws, output_format="console", output_file=None,
        config_path=None, skills_dir_override=skills,
        console=Console(file=StringIO(), force_terminal=False),
    )
    messages = [r["message"] for r in _records(shared_logger, log_file)]
    assert any("LLM call failed" in m for m in messages)
    assert "LLM call failed" not in capsys.readouterr().err


@pytest.mark.unit
def test_taint_confirm_logs_to_file(shared_logger, tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    ctrl = ws / "Ctrl.cs"
    ctrl.write_text("namespace N;\npublic class C {\n  [HttpGet(\"/x\")]\n  public string Get() { return \"x\"; }\n}\n",
                    encoding="utf-8")
    findings = tmp_path / "findings.json"
    findings.write_text(json.dumps({"findings": [{
        "id": str(uuid4()), "issue": "x", "severity": "high", "confidence": "medium",
        "file_path": str(ctrl), "line_start": 4,
    }]}), encoding="utf-8")
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    entry_reg = EntryPointDetectorRegistry()
    entry_reg.register(AspNetCoreEntryPointDetector())
    log_file = tmp_path / "falconeye.log"
    _stub_container(
        monkeypatch, commands_taint_confirm,
        config=SimpleNamespace(logging=_logging_config(log_file)),
        review_confirmation_service=ReviewConfirmationService(
            adapter_registry=reg, entry_point_registry=entry_reg,
            taint_index=load_taint_index(languages=["csharp"]),
        ),
    )
    taint_confirm_command(
        path=ws, findings_file=findings, output_format="console", output_file=None,
        languages=["csharp"], config_path=None,
        console=Console(file=StringIO(), force_terminal=False),
    )
    messages = [r["message"] for r in _records(shared_logger, log_file)]
    assert any("Entry points detected" in m for m in messages)


# ---------------------------------------------------------------
# -v wiring
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("argv, target", [
    (["skill", "run", "x", "{ws}", "-v"], "skill_run_command"),
    (["skill", "run-all", "{ws}", "--verbose"], "skill_run_all_command"),
    (["taint", "{ws}", "-v"], "taint_command"),
    (["taint-confirm", "{ws}", "-f", "{f}", "-v"], "taint_confirm_command"),
])
def test_verbose_flag_reaches_the_command(argv, target, tmp_path, monkeypatch):
    from falconeye.adapters.cli import main as cli_main

    findings = tmp_path / "f.json"
    findings.write_text("[]", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(cli_main, target, lambda **kw: seen.update(kw))
    args = [a.replace("{ws}", str(tmp_path)).replace("{f}", str(findings)) for a in argv]
    result = CliRunner().invoke(cli_main.app, args)
    assert result.exit_code == 0, result.output
    assert seen["verbose"] is True
