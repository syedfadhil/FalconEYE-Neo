"""
Domain models for security-audit skills.

A "skill" is a structured audit playbook: YAML frontmatter with
metadata (name, category, function), a markdown body with the audit
procedure, and grep-style discovery patterns that narrow the LLM's
search space before evaluation.

Skills are supplied by the operator, not shipped by FalconEYE. The
skill directory is configured in ``config.skills.directory`` and each
skill lives at ``<directory>/<name>/SKILL.md``.

This module defines the parsed representation only. Loading and
grep discovery live in :mod:`falconeye.infrastructure.skills`.
Execution (LLM invocation + finding emission) lives in the
application layer and is added in a follow-up patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class SkillMetadata:
    """Frontmatter fields from a skill markdown file.

    All fields except ``name`` and ``description`` are optional. Skills
    in the wild vary widely in which metadata they populate; the loader
    accepts any subset and preserves what's there.

    ``function`` is important for pipeline routing: only ``scan``
    skills are runnable today. ``verify`` and ``triage`` skills are
    accepted for listing but produce a clear error if invoked -- their
    integration is deferred to later phases.
    """
    name: str
    description: str
    version: Optional[str] = None
    maintainer: Optional[str] = None
    tags: Tuple[str, ...] = ()
    display_name: Optional[str] = None
    category: Optional[str] = None
    # llm-reasoning | tool-invocation | research-agent
    skill_type: Optional[str] = None
    # scan | generate | verify | triage | research
    function: Optional[str] = None
    breadth: Optional[str] = None  # broad | targeted
    status: Optional[str] = None   # ga | experimental | wip | n/a
    # FalconEYE-specific: filename globs selecting the files an
    # LLM-only skill sweeps (e.g. ("package.json", "*.csproj") for a
    # dependency audit). Empty means "source files of every enabled
    # language". Ignored when the skill has discovery patterns.
    targets: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryPattern:
    """A grep-style discovery pattern extracted from a skill's Step 1.

    Represents one ``grep -rn -E "..." --include="*.{ext}"`` command:
    a regex plus the file extensions it targets. Extensions are stored
    normalized (leading dot, lowercase).

    ``description`` is the ``# comment`` that preceded the grep in the
    skill file, if any. Populated so ``skill list`` can show what each
    pattern is looking for.
    """
    regex: str
    extensions: Tuple[str, ...]
    description: Optional[str] = None


@dataclass(frozen=True)
class DiscoveryHit:
    """A single file:line match from grep discovery.

    ``matched_span`` records where the regex matched inside the line
    so downstream code (Phase 1b) can highlight the match when
    presenting the hit to an LLM.
    """
    file_path: str
    line_number: int
    line_text: str
    matched_span: Tuple[int, int]  # (start, end) column indices in line_text
    pattern_description: Optional[str] = None


@dataclass(frozen=True)
class SkillDefinition:
    """A fully-parsed skill: metadata + body + extracted patterns.

    ``body`` is the full markdown body (everything after the closing
    ``---`` of the frontmatter). It's kept intact so the runner can
    include the audit description in the LLM system prompt without
    re-parsing.

    ``discovery_patterns`` may be empty for skills that don't ship
    grep patterns (e.g. skills that rely entirely on LLM reasoning
    over the whole workspace). The runner then sweeps: one LLM call
    per candidate file (see ``metadata.targets``), capped per run.
    """
    metadata: SkillMetadata
    source_path: Path
    body: str
    discovery_patterns: Tuple[DiscoveryPattern, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        """Canonical name -- shortcut for metadata.name."""
        return self.metadata.name

    @property
    def is_runnable_today(self) -> bool:
        """Whether Phase 1b can execute this skill.

        Only ``scan``-function skills are runnable in Phase 1. ``verify``
        maps to fp-check (Phase 3) and ``triage`` maps to the triage
        stage (Phase 4). Skills without a declared function default to
        scan for backwards compatibility with older skill files.
        """
        fn = (self.metadata.function or "scan").strip().lower()
        return fn == "scan"
