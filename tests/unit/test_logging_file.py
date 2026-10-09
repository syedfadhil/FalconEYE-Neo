"""Unit tests for applying config.logging to the shared FalconEYE logger.

Coverage:
  * The log file is written even though the singleton was created
    earlier with defaults (as the LLM adapters do during container
    creation) -- the bug that meant logging.file was never written
  * Non-verbose: file only; verbose: console + file
  * Never zero handlers: no bare last-resort output on stderr
  * Reconfiguring closes the previous file handler
  * An unwritable log path warns once and doesn't raise
  * JSON lines carry extras, including non-JSON values
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from falconeye.adapters.cli.commands import _configure_logger_verbosity
from falconeye.infrastructure.logging import FalconEyeLogger


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


def _container(log_file):
    return SimpleNamespace(config=SimpleNamespace(logging=SimpleNamespace(
        level="INFO", file=str(log_file) if log_file else "", console=True,
        rotation="daily", retention_days=30,
    )))


def _flush(lg):
    for handler in lg.logger.handlers:
        handler.flush()


def _kinds(lg):
    return sorted(type(h).__name__ for h in lg.logger.handlers)


@pytest.mark.unit
def test_log_file_is_written_although_singleton_existed(shared_logger, tmp_path):
    log_file = tmp_path / "logs" / "falconeye.log"
    _configure_logger_verbosity(_container(log_file), verbose=False)
    shared_logger.warning("File processing failed", extra={"file_path": "libs/x.stub.ts"})
    _flush(shared_logger)

    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["message"] == "File processing failed"
    assert record["file_path"] == "libs/x.stub.ts"
    assert record["level"] == "WARNING"


@pytest.mark.unit
def test_non_verbose_logs_to_file_only(shared_logger, tmp_path):
    _configure_logger_verbosity(_container(tmp_path / "f.log"), verbose=False)
    assert _kinds(shared_logger) == ["TimedRotatingFileHandler"]


@pytest.mark.unit
def test_verbose_logs_to_console_and_file(shared_logger, tmp_path):
    _configure_logger_verbosity(_container(tmp_path / "f.log"), verbose=True)
    assert _kinds(shared_logger) == ["StreamHandler", "TimedRotatingFileHandler"]


@pytest.mark.unit
def test_no_handlers_means_no_last_resort_spill(shared_logger, capsys):
    """Before: non-verbose with no file left zero handlers, and Python's
    last-resort handler printed bare warnings over the progress bar."""
    _configure_logger_verbosity(_container(None), verbose=False)
    assert _kinds(shared_logger) == ["NullHandler"]
    shared_logger.warning("Truncating text for embedding")
    assert "Truncating" not in capsys.readouterr().err


@pytest.mark.unit
def test_reconfigure_closes_previous_file(shared_logger, tmp_path):
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    shared_logger.configure(log_file=first, console=False)
    shared_logger.info("one")
    shared_logger.configure(log_file=second, console=False)
    os.remove(first)  # fails on Windows if the handle were still open
    assert _kinds(shared_logger) == ["TimedRotatingFileHandler"]


@pytest.mark.unit
def test_unwritable_log_path_warns_and_continues(shared_logger, tmp_path, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    shared_logger.configure(log_file=blocker / "falconeye.log", console=False)
    assert "cannot write log file" in capsys.readouterr().err
    assert _kinds(shared_logger) == ["NullHandler"]
    shared_logger.warning("still fine")


@pytest.mark.unit
def test_json_lines_tolerate_non_json_extras(shared_logger, tmp_path, capsys):
    log_file = tmp_path / "f.log"
    shared_logger.configure(log_file=log_file, console=False, rotation="none")
    shared_logger.info("indexed", extra={"root": Path("C:/src/app"), "count": 3})
    _flush(shared_logger)
    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["root"] in ("C:/src/app", "C:\\src\\app")
    assert record["count"] == 3
    assert "Logging error" not in capsys.readouterr().err
