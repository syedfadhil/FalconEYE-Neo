"""Unit tests for file_discovery exclusions (index and review).

Coverage:
  * is_excluded: directory patterns match Windows (backslash) paths and
    top-level directories; suffix patterns; patterns written with
    backslashes; wildcard-only patterns match nothing
  * Indexer discovery (code and documents): a repo living under a
    directory named like an excluded one (build/) is still indexed,
    while excluded directories inside it are skipped
  * Review file collection: exclusions applied, excluded count,
    sorted order
  * CLI: -e adds to the configured defaults; review/scan accept -e
"""

from __future__ import annotations

import logging
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from falconeye.adapters.cli.commands import _collect_review_files, _effective_exclusions
from falconeye.application.commands.index_codebase import IndexCodebaseHandler
from falconeye.domain.services.language_detector import LanguageDetector
from falconeye.infrastructure.config.config_models import FileDiscoveryConfig
from falconeye.infrastructure.workspace_files import is_excluded


DEFAULTS = FileDiscoveryConfig().default_exclusions


def _touch(path: Path, content: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------
# is_excluded
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("rel, expected", [
    (PureWindowsPath(r"apps\web\node_modules\rxjs\index.js"), True),  # Windows separators
    (PureWindowsPath(r"node_modules\rxjs\index.js"), True),            # top-level dir
    ("src/dist/bundle.js", True),
    ("vendor.min.js", True),
    ("src/app/main.ts", False),
    ("src/distribution/main.ts", False),  # "/dist/" is a whole segment
])
def test_default_patterns(rel, expected):
    assert is_excluded(rel, DEFAULTS) is expected


@pytest.mark.unit
def test_suffix_and_backslash_patterns():
    assert is_excluded(r"libs\topology-stub\models\MX7000.stub.ts", ["*.stub.ts"])
    assert is_excluded("src/generated/api.ts", [r"*\generated\*"])
    assert not is_excluded("src/app.ts", [r"*\generated\*"])


@pytest.mark.unit
@pytest.mark.parametrize("pattern", ["*", "**", "***"])
def test_wildcard_only_pattern_matches_nothing(pattern):
    """It used to strip to '' -- a substring of everything."""
    assert not is_excluded("src/app.py", [pattern])


# ---------------------------------------------------------------
# Indexer discovery
# ---------------------------------------------------------------


def _discovery_handler():
    handler = IndexCodebaseHandler.__new__(IndexCodebaseHandler)
    handler.language_detector = LanguageDetector()
    handler.logger = logging.getLogger("test.exclusions")
    return handler


@pytest.mark.unit
def test_index_discovery_under_build_dir_with_default_patterns(tmp_path):
    """Regression: the old matcher also searched the ABSOLUTE path, so
    '*/build/*' excluded every file of a repo under C:\\...\\build\\."""
    root = tmp_path / "build" / "Portal"
    _touch(root / "src" / "app.py")
    _touch(root / "node_modules" / "pkg" / "dep.py")
    found = _discovery_handler()._discover_files(root, "python", DEFAULTS)
    assert [p.relative_to(root).as_posix() for p in found] == ["src/app.py"]


@pytest.mark.unit
def test_document_discovery_under_build_dir(tmp_path):
    root = tmp_path / "build" / "Portal"
    _touch(root / "README.md", "# readme\n")
    _touch(root / "node_modules" / "pkg" / "README.md", "# dep\n")
    found = _discovery_handler()._discover_documents(root, DEFAULTS)
    assert [p.relative_to(root).as_posix() for p in found] == ["README.md"]


# ---------------------------------------------------------------
# Review file collection
# ---------------------------------------------------------------


@pytest.mark.unit
def test_review_collection_applies_exclusions(tmp_path):
    _touch(tmp_path / "src" / "b.ts")
    _touch(tmp_path / "src" / "a.ts")
    _touch(tmp_path / "libs" / "models" / "MX7000.stub.ts")
    _touch(tmp_path / "node_modules" / "rxjs" / "index.ts")
    files, excluded = _collect_review_files(
        tmp_path, ["typescript"], LanguageDetector.LANGUAGE_EXTENSIONS,
        DEFAULTS + ["*.stub.ts"],
    )
    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["src/a.ts", "src/b.ts"]
    assert excluded == 2


@pytest.mark.unit
def test_review_collection_without_exclusions_keeps_everything(tmp_path):
    _touch(tmp_path / "x.stub.ts")
    files, excluded = _collect_review_files(
        tmp_path, ["typescript"], LanguageDetector.LANGUAGE_EXTENSIONS, [],
    )
    assert len(files) == 1 and excluded == 0


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


@pytest.mark.unit
def test_cli_patterns_add_to_config_defaults():
    container = SimpleNamespace(config=SimpleNamespace(
        file_discovery=SimpleNamespace(default_exclusions=["*/node_modules/*"]),
    ))
    assert _effective_exclusions(container, None) == ["*/node_modules/*"]
    assert _effective_exclusions(container, ["*.stub.ts"]) == ["*/node_modules/*", "*.stub.ts"]


@pytest.mark.unit
@pytest.mark.parametrize("command, target", [
    ("review", "review_command"),
    ("scan", "scan_command"),
])
def test_review_and_scan_accept_exclude(command, target, tmp_path, monkeypatch):
    from falconeye.adapters.cli import main as cli_main

    seen = {}
    monkeypatch.setattr(cli_main, target, lambda **kw: seen.update(kw))
    result = CliRunner().invoke(
        cli_main.app, [command, str(tmp_path), "-e", "*.stub.ts", "--exclude", "*/e2e/*"],
    )
    assert result.exit_code == 0, result.output
    assert seen["exclude"] == ["*.stub.ts", "*/e2e/*"]
