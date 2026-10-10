"""Unit tests for the single runtime version string.

Coverage:
  * falconeye.__version__ equals pyproject.toml's [project].version
  * `falconeye upgrade` would parse the same value from __init__.py
  * SARIF output, --help, the banners and generated configs all show it
"""

from __future__ import annotations

import json
import tomllib
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import falconeye
from falconeye import __version__
from falconeye.adapters.cli.banner import print_banner, print_compact_banner
from falconeye.adapters.formatters.sarif_formatter import SARIFFormatter
from falconeye.infrastructure.config.config_loader import ConfigLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_version_matches_pyproject():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        assert __version__ == tomllib.load(f)["project"]["version"]


@pytest.mark.unit
def test_upgrade_parser_reads_the_same_version():
    """_get_remote_version takes the first line holding both
    '__version__' and '=' -- comments must not shadow the assignment."""
    for line in Path(falconeye.__file__).read_text(encoding="utf-8").splitlines():
        if "__version__" in line and "=" in line:
            assert line.split("=", 1)[1].strip().strip('"').strip("'") == __version__
            break
    else:
        pytest.fail("no __version__ assignment found")


@pytest.mark.unit
def test_sarif_reports_the_runtime_version(empty_review):
    driver = json.loads(SARIFFormatter().format_review(empty_review))["runs"][0]["tool"]["driver"]
    assert driver["version"] == driver["semanticVersion"] == __version__


@pytest.mark.unit
def test_help_and_banners_show_the_version():
    from falconeye.adapters.cli import main as cli_main

    assert f"v{__version__}" in CliRunner().invoke(cli_main.app, ["--help"]).output
    for printer in (print_banner, print_compact_banner):
        buf = StringIO()
        printer(Console(file=buf, force_terminal=False, width=200))
        assert f"v{__version__}" in buf.getvalue()


@pytest.mark.unit
def test_compact_banner_title_keeps_box_width():
    buf = StringIO()
    print_compact_banner(Console(file=buf, force_terminal=False, width=200))
    title = next(l for l in buf.getvalue().splitlines() if "Security Code Review" in l)
    assert len(title.strip().strip("║")) == 75


@pytest.mark.unit
def test_generated_config_header_shows_the_version(tmp_path):
    path = ConfigLoader.create_default_config(str(tmp_path / "config.yaml"))
    assert path.read_text().splitlines()[0] == f"# FalconEYE v{__version__} Configuration"
