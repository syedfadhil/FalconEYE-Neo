"""Unit tests for skills-directory resolution.

Covers:
  * New default: ``~/.falconeye/skills`` (user-level, not per-project)
  * FALCONEYE_SKILLS_DIR env var override
  * Tilde and $VAR expansion
  * Relative-config-path resolution against config file location
  * ``--skills-dir`` CLI override bypasses config gating
"""

from __future__ import annotations

import os
import textwrap
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from falconeye.infrastructure.config.config_models import SkillsConfig
from falconeye.infrastructure.di.container import _resolve_skills_directory
from falconeye.adapters.cli.commands_skill import skill_list_command


# ---------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------


@pytest.mark.unit
def test_default_directory_is_user_level():
    """Default is a user-level path, not per-project.

    Rationale: skills are a global asset shared across every project
    the operator analyzes; the ``./skills`` default would force a
    copy into every codebase.
    """
    assert SkillsConfig().directory == "~/.falconeye/skills"


# ---------------------------------------------------------------
# _resolve_skills_directory
# ---------------------------------------------------------------


@pytest.mark.unit
def test_resolve_env_var_wins_over_config(monkeypatch):
    """FALCONEYE_SKILLS_DIR takes precedence over config value."""
    monkeypatch.setenv("FALCONEYE_SKILLS_DIR", "/tmp/env-override")
    result = _resolve_skills_directory(
        config_value="/config/value/skills",
        config_path="/proj/config.yaml",
    )
    assert str(result) == "/tmp/env-override"


@pytest.mark.unit
def test_resolve_env_var_expands_tilde(monkeypatch):
    """~ in env var expands to $HOME."""
    monkeypatch.setenv("FALCONEYE_SKILLS_DIR", "~/env-tilde")
    result = _resolve_skills_directory(
        config_value="ignored",
        config_path=None,
    )
    assert result == Path.home() / "env-tilde"


@pytest.mark.unit
def test_resolve_env_var_expands_shell_var(monkeypatch):
    """$VAR expansion works too (via os.path.expandvars)."""
    monkeypatch.setenv("FALCONEYE_SKILLS_DIR", "$HOME/env-var")
    result = _resolve_skills_directory(
        config_value="ignored",
        config_path=None,
    )
    assert result == Path.home() / "env-var"


@pytest.mark.unit
def test_resolve_empty_env_var_falls_through(monkeypatch):
    """Empty string in env var is treated as unset."""
    monkeypatch.setenv("FALCONEYE_SKILLS_DIR", "")
    result = _resolve_skills_directory(
        config_value="/absolute/config",
        config_path=None,
    )
    assert str(result) == "/absolute/config"


@pytest.mark.unit
def test_resolve_whitespace_env_var_falls_through(monkeypatch):
    """Whitespace-only env var is treated as unset."""
    monkeypatch.setenv("FALCONEYE_SKILLS_DIR", "   ")
    result = _resolve_skills_directory(
        config_value="/absolute/config",
        config_path=None,
    )
    assert str(result) == "/absolute/config"


@pytest.mark.unit
def test_resolve_config_absolute_stays_absolute(monkeypatch):
    """An absolute config value is not re-parented on config_path."""
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    result = _resolve_skills_directory(
        config_value="/absolute/skills",
        config_path="/proj/config.yaml",
    )
    assert str(result) == "/absolute/skills"


@pytest.mark.unit
def test_resolve_config_relative_uses_config_parent(monkeypatch, tmp_path):
    """Relative config value resolves against the config file's parent."""
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("skills:\n  enabled: true\n")
    result = _resolve_skills_directory(
        config_value="./skills",
        config_path=str(cfg),
    )
    assert result == tmp_path / "skills"


@pytest.mark.unit
def test_resolve_config_expands_tilde(monkeypatch):
    """~ in config value expands to $HOME (typical default case)."""
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    result = _resolve_skills_directory(
        config_value="~/.falconeye/skills",
        config_path=None,
    )
    assert result == Path.home() / ".falconeye" / "skills"


@pytest.mark.unit
def test_resolve_default_case_no_env_no_config_path(monkeypatch):
    """The most common no-config-file case: the default expands to
    the user's ~/.falconeye/skills."""
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    result = _resolve_skills_directory(
        config_value=SkillsConfig().directory,
        config_path=None,
    )
    assert result == Path.home() / ".falconeye" / "skills"


# ---------------------------------------------------------------
# CLI --skills-dir override
# ---------------------------------------------------------------


def _write_skill(root: Path, name: str, content: str) -> None:
    sub = root / name
    sub.mkdir()
    (sub / "SKILL.md").write_text(content, encoding="utf-8")


@pytest.mark.unit
def test_cli_skills_dir_override_bypasses_config_gating(tmp_path, monkeypatch):
    """--skills-dir works even when config.skills.enabled=false.

    Rationale: the operator explicitly named a directory; that intent
    is unambiguous. Making them flip the config flag first would be
    friction for one-off inspection.
    """
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "test-skill", textwrap.dedent("""\
    ---
    name: test-skill
    description: a test
    ---
    """))

    # Config with skills DISABLED
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""\
    llm:
      provider: litellm
    skills:
      enabled: false
    """))

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    skill_list_command(
        config_path=cfg,
        console=console,
        show_patterns=False,
        skills_dir_override=skills_dir,
    )
    output = buf.getvalue()
    assert "test-skill" in output
    # Should NOT show the "not enabled" error
    assert "not enabled" not in output.lower()


@pytest.mark.unit
def test_cli_skills_dir_override_expands_tilde(tmp_path, monkeypatch):
    """A ~-prefixed --skills-dir gets expanded before loading."""
    monkeypatch.setenv("HOME", str(tmp_path))
    skills_dir = tmp_path / "custom-skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "hello", textwrap.dedent("""\
    ---
    name: hello
    description: greeter
    ---
    """))

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    skill_list_command(
        config_path=None,
        console=console,
        show_patterns=False,
        skills_dir_override=Path("~/custom-skills"),
    )
    assert "hello" in buf.getvalue()


@pytest.mark.unit
def test_cli_error_mentions_all_three_configuration_paths(tmp_path, monkeypatch):
    """The error message when skills are disabled points to all three
    ways to enable them: config, env var, and --skills-dir."""
    monkeypatch.delenv("FALCONEYE_SKILLS_DIR", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  provider: litellm\n")  # skills defaults to disabled

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    with pytest.raises(SystemExit):
        skill_list_command(
            config_path=cfg,
            console=console,
            show_patterns=False,
            skills_dir_override=None,
        )
    output = buf.getvalue().lower()
    assert "config.skills.enabled" in output
    assert "falconeye_skills_dir" in output
    assert "--skills-dir" in output
