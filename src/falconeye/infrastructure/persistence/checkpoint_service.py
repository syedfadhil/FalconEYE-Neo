"""
Checkpoint service for resumable security reviews.

Persists incremental progress during a multi-file scan so that a run
interrupted by Ctrl+C, LLM disconnection, or crash can be resumed
without re-analyzing files that already completed.

Design:
- Session is keyed by a stable hash of (codebase root, sorted file paths,
  system_prompt hash, top_k, validate, chat model). Same inputs -> same
  session ID -> auto-resume; changing any of them starts a fresh session.
- Checkpoint file is written atomically after every completed file via
  temp-file + os.replace, so a crash mid-write cannot corrupt state.
- Concurrent writers (files completing in parallel) are serialized with
  an asyncio.Lock.
- On successful completion of the entire scan, the checkpoint is
  removed so it doesn't clutter the checkpoint directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set
from uuid import UUID

from ...domain.models.security import SecurityFinding, Severity, FindingConfidence


@dataclass
class CheckpointState:
    """In-memory representation of a checkpoint."""
    session_id: str
    codebase_path: str
    total_files: int
    completed_files: Set[str] = field(default_factory=set)
    failed_files: List[dict] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "session_id": self.session_id,
            "codebase_path": self.codebase_path,
            "total_files": self.total_files,
            "completed_files": sorted(self.completed_files),
            "failed_files": self.failed_files,
            "findings": self.findings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        return cls(
            session_id=data["session_id"],
            codebase_path=data["codebase_path"],
            total_files=data.get("total_files", 0),
            completed_files=set(data.get("completed_files", [])),
            failed_files=list(data.get("failed_files", [])),
            findings=list(data.get("findings", [])),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class CheckpointService:
    """Manages checkpoint files for resumable scans."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._state: Optional[CheckpointState] = None
        self._path: Optional[Path] = None

    # ---------- session identity ----------

    @staticmethod
    def compute_session_id(
        codebase_path: Path,
        file_paths: Iterable[Path],
        *,
        system_prompt_signature: str = "",
        top_k: int = 5,
        validate: bool = False,
        chat_model: str = "",
    ) -> str:
        """
        Build a stable session ID from the parameters that affect results.

        Adding/removing files or changing analysis parameters yields a
        different session, which is intentional -- resuming with a
        different file set would produce incoherent output.
        """
        h = hashlib.sha256()
        h.update(str(Path(codebase_path).resolve()).encode())
        h.update(b"|")
        h.update(str(top_k).encode())
        h.update(b"|")
        h.update(str(validate).encode())
        h.update(b"|")
        h.update(chat_model.encode())
        h.update(b"|")
        h.update(system_prompt_signature.encode())
        h.update(b"|")
        # Sort so ordering differences (e.g., rglob iteration order) don't
        # change the session ID.
        for p in sorted(str(Path(fp).resolve()) for fp in file_paths):
            h.update(p.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    def _path_for(self, session_id: str) -> Path:
        return self.checkpoint_dir / f"{session_id}.json"

    # ---------- lifecycle ----------

    def find_existing(self, session_id: str) -> Optional[CheckpointState]:
        """Return the checkpoint for this session if present, else None."""
        p = self._path_for(session_id)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except (OSError, ValueError, KeyError):
            # Corrupt or unreadable -- pretend it's not there. Caller
            # can decide whether to warn.
            return None

    async def start(
        self,
        session_id: str,
        codebase_path: Path,
        total_files: int,
        resume_from: Optional[CheckpointState] = None,
    ) -> CheckpointState:
        """
        Begin (or resume) a session. Returns the live state object.
        """
        async with self._lock:
            if resume_from is not None:
                assert resume_from.session_id == session_id
                resume_from.total_files = total_files
                self._state = resume_from
            else:
                self._state = CheckpointState(
                    session_id=session_id,
                    codebase_path=str(codebase_path),
                    total_files=total_files,
                )
            self._path = self._path_for(session_id)
            await self._flush_locked()
            return self._state

    async def mark_completed(
        self,
        file_path: Path,
        findings: Iterable[SecurityFinding],
    ) -> None:
        """Record a successfully-analyzed file and its findings."""
        key = self._file_key(file_path)
        async with self._lock:
            if self._state is None:
                return
            self._state.completed_files.add(key)
            self._state.findings.extend(f.to_dict() for f in findings)
            self._state.updated_at = time.time()
            await self._flush_locked()

    async def mark_failed(
        self,
        file_path: Path,
        error: str,
    ) -> None:
        """Record a file that raised during analysis."""
        key = self._file_key(file_path)
        async with self._lock:
            if self._state is None:
                return
            self._state.failed_files.append({"path": key, "error": error})
            self._state.updated_at = time.time()
            await self._flush_locked()

    async def finish(self, remove: bool = True) -> None:
        """
        Mark the scan complete. When remove=True (the default), the
        checkpoint file is deleted so it does not linger.
        """
        async with self._lock:
            if self._state is None or self._path is None:
                return
            if remove:
                try:
                    self._path.unlink()
                except FileNotFoundError:
                    pass
            self._state = None
            self._path = None

    # ---------- helpers ----------

    @staticmethod
    def _file_key(file_path: Path) -> str:
        return str(Path(file_path).resolve())

    def already_completed(self, file_path: Path) -> bool:
        """Whether this file was already analyzed in the current session."""
        if self._state is None:
            return False
        return self._file_key(file_path) in self._state.completed_files

    def restored_findings(self) -> List[SecurityFinding]:
        """
        Reconstruct SecurityFinding objects from checkpoint data.

        Used at resume time to seed the aggregate review with findings
        collected in the interrupted run.
        """
        if self._state is None:
            return []
        out: List[SecurityFinding] = []
        for d in self._state.findings:
            try:
                out.append(self._finding_from_dict(d))
            except Exception:
                continue
        return out

    @staticmethod
    def _finding_from_dict(d: dict) -> SecurityFinding:
        return SecurityFinding(
            id=UUID(d["id"]),
            issue=d["issue"],
            reasoning=d["reasoning"],
            mitigation=d["mitigation"],
            severity=Severity(d["severity"]),
            confidence=FindingConfidence(d["confidence"]),
            file_path=d["file_path"],
            code_snippet=d["code_snippet"],
            line_start=d.get("line_start"),
            line_end=d.get("line_end"),
            cwe_id=d.get("cwe_id"),
            tags=list(d.get("tags", [])),
            # New reasoning/validation fields; safe defaults for
            # checkpoints written before this patch.
            evidence_lines=list(d.get("evidence_lines", [])),
            assumed_context=d.get("assumed_context", "") or "",
            why_not_false_positive=d.get("why_not_false_positive", "") or "",
            reasoning_trace=d.get("reasoning_trace", "") or "",
            validation_warnings=list(d.get("validation_warnings", [])),
            # Verifier fields; also safe for pre-verifier checkpoints.
            verifier_verdict=d.get("verifier_verdict"),
            verifier_reasoning=d.get("verifier_reasoning", "") or "",
            verifier_counterargument=d.get("verifier_counterargument", "") or "",
        )

    async def _flush_locked(self) -> None:
        """
        Atomically write the current state to disk.

        MUST be called with self._lock held.
        """
        if self._state is None or self._path is None:
            return

        data = self._state.to_dict()
        # os.replace() is atomic on POSIX and Windows for existing paths,
        # so a crash mid-write cannot leave the checkpoint half-written.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
