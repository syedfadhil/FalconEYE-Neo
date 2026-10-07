"""Additional edge-case coverage for previously untested branches."""

import subprocess
import time
from pathlib import Path

import pytest

from falconeye.domain.services.checksum_service import ChecksumService
from falconeye.domain.services.language_detector import LanguageDetector
from falconeye.domain.services.project_identifier import ProjectIdentifier
from falconeye.domain.value_objects.project_metadata import ProjectType
from falconeye.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerState,
)
from falconeye.infrastructure.resilience.retry import RetryConfig, retry_with_backoff, retry_with_backoff_sync


def test_checksum_service_no_cache_batch_new_deleted_and_unreadable(tmp_path, monkeypatch):
    service = ChecksumService()
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    missing = tmp_path / "missing.py"
    a.write_text("alpha")
    b.write_text("beta")

    assert service.has_file_changed_quick(a, None) is True
    assert service.has_file_changed_checksum(a, None) is True
    assert service.identify_new_files({a, b}, {a}) == {b}
    assert service.identify_deleted_files({a}, {a, b}) == {b}

    checksums = service.batch_calculate_checksums([a, b, missing], max_workers=2)
    assert set(checksums) == {a, b}
    assert checksums[a].startswith("sha256:")

    class UnreadablePath:
        def stat(self):
            raise PermissionError("denied")

    assert service.has_file_changed_quick(UnreadablePath(), object()) is True


def test_language_detector_supported_languages_listing_metadata_and_skips(tmp_path):
    detector = LanguageDetector()
    supported = detector.get_supported_languages()
    assert {"python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "php", "ruby", "csharp", "dart"}.issubset(supported)

    assert detector._is_valid_language("python") is True
    assert detector._is_valid_language("brainfuck") is False
    assert ".py" in detector.LANGUAGE_EXTENSIONS["python"]
    one = tmp_path / "one.py"
    one.write_text("print('x')")
    assert detector.detect_all_languages(one) == ["python"]

    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    (skipped / "ignored.js").write_text("console.log(1)")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception):
        detector.detect_language(empty)


def test_project_identifier_local_git_without_remote_and_helper_methods(tmp_path, monkeypatch):
    service = ProjectIdentifier()
    repo = tmp_path / "Repo 123"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    assert service._find_git_root(nested) == repo
    assert service._sanitize_project_id(" 123 weird!!name ") == "p123_weird_name"
    assert service._sanitize_project_id("!!!") == "project"
    assert service._normalize_git_url("https://github.com/u/r.git") == "github.com/u/r"
    assert service._normalize_git_url("http://github.com/u/r") == "github.com/u/r"
    assert service._normalize_git_url("git@github.com:u/r.git") == "github.com/u/r"

    def no_remote(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", no_remote)
    project_id, name, project_type, remote = service.identify_project(nested)
    assert project_id == "repo_123"
    assert name == "Repo 123"
    assert project_type is ProjectType.GIT
    assert remote is None


def test_project_identifier_git_command_success_failure_and_changed_files(tmp_path, monkeypatch):
    service = ProjectIdentifier()
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "rev-parse HEAD" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n")
        if "status --porcelain" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M a.py\n")
        if "diff --name-only old HEAD" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="a.py\nb.py\n")
        if "diff --name-only HEAD" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="c.py\n")
        if "ls-files --others" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="new.py\n")
        if "config --get remote.origin.url" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="git@github.com:u/r.git\n")
        return subprocess.CompletedProcess(cmd, 2, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert service._get_git_remote_url(repo) == "github.com/u/r"
    assert service.get_current_git_commit(repo) == "abc123"
    assert service.has_uncommitted_changes(repo) is True
    assert service.get_git_changed_files(repo, "old") == [repo / "a.py", repo / "b.py"]
    assert service.get_git_changed_files(repo) == [repo / "c.py"]
    assert service.get_git_untracked_files(repo) == [repo / "new.py"]

    def raises_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr(subprocess, "run", raises_timeout)
    assert service._get_git_remote_url(repo) is None
    assert service.get_current_git_commit(repo) is None
    assert service.has_uncommitted_changes(repo) is True
    assert service.get_git_changed_files(repo) == []
    assert service.get_git_untracked_files(repo) == []


@pytest.mark.asyncio
async def test_retry_default_config_and_jitter_branch(monkeypatch):
    sleeps = []
    monkeypatch.setattr("random.random", lambda: 0.5)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    attempts = 0

    @retry_with_backoff(RetryConfig(max_retries=1, initial_delay=2, jitter=0.5, retryable_exceptions=(OSError,)))
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert sleeps == [2.5]
    assert retry_with_backoff()(lambda: None) is not None


def test_retry_sync_success_exhaustion_non_retryable_and_jitter(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr("random.random", lambda: 1.0)
    attempts = 0

    @retry_with_backoff_sync(RetryConfig(max_retries=2, initial_delay=1, exponential_base=2, jitter=0.25, retryable_exceptions=(ConnectionError,)))
    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert attempts == 3
    assert sleeps == [1.25, 2.5]

    @retry_with_backoff_sync(RetryConfig(max_retries=1, initial_delay=0, jitter=0, retryable_exceptions=(TimeoutError,)))
    def exhausts():
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        exhausts()

    @retry_with_backoff_sync(RetryConfig(max_retries=3, retryable_exceptions=(TimeoutError,)))
    def non_retryable():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        non_retryable()
    assert retry_with_backoff_sync()(lambda: "x")() == "x"


def test_circuit_breaker_sync_paths_and_manual_reset(monkeypatch):
    breaker = CircuitBreaker("sync", CircuitBreakerConfig(failure_threshold=1, success_threshold=1, timeout=10, exclude_exceptions=()))

    @breaker.protect_sync
    def fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        fail()
    assert breaker.state is CircuitBreakerState.OPEN
    with pytest.raises(CircuitBreakerError):
        fail()

    breaker.reset()
    assert breaker.state is CircuitBreakerState.CLOSED

    @breaker.protect_sync
    def ok(value):
        return value

    assert ok("healthy") == "healthy"

    now = 10.0
    monkeypatch.setattr("falconeye.infrastructure.resilience.circuit_breaker.time.time", lambda: now)
    with pytest.raises(RuntimeError):
        fail()
    now = 21.0
    assert breaker.state is CircuitBreakerState.HALF_OPEN
    assert ok("recovered") == "recovered"
    assert breaker.state is CircuitBreakerState.CLOSED
