"""
Skill loader: parse a SKILL.md file into a :class:`SkillDefinition`.

Handles the frontmatter+body+grep-patterns format documented in
:mod:`falconeye.domain.models.skill`:

  ---
  name: injection-prevention
  description: ...
  ---

  # Human title

  ## Step 1 -- Discover ...

  ```bash
  # SQL injection ...
  grep -rn -E "..." --include="*.{py,java,cs}" .
  ```

The loader is deliberately tolerant: skills come from operators with
varying conventions, and refusing to load on minor formatting
differences would be a bad experience. Concrete tolerances:

- YAML frontmatter is optional; skills without frontmatter get name
  derived from the file path.
- Step 1 section is located by heading match ("Step 1"); the loader
  handles ``##`` or ``###`` and any dash / em-dash separator.
- Bash blocks that aren't grep commands are silently skipped.
- Malformed grep commands (unbalanced quotes, no -E flag) are
  skipped with a debug log but don't fail the load.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml

from ...domain.models.skill import (
    DiscoveryPattern,
    SkillDefinition,
    SkillMetadata,
)


_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Frontmatter parsing
# --------------------------------------------------------------------


# The closing fence may end the file: a frontmatter-only stub (a
# verify/triage skill with no body yet) saved without a trailing
# newline must not lose its metadata.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---[ \t]*(?:\r?\n(.*))?\Z",
    re.DOTALL,
)


def _parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Split file content into (metadata_dict, body).

    Missing frontmatter returns ``({}, content)`` so the loader can
    still produce a SkillDefinition from a bare markdown file.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    raw_yaml, body = match.group(1), match.group(2) or ""
    try:
        data = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as e:
        _logger.warning("Failed to parse frontmatter YAML: %s", e)
        return {}, content
    if not isinstance(data, dict):
        return {}, content
    return data, body


def _metadata_from_frontmatter(
    fm: dict,
    fallback_name: str,
) -> SkillMetadata:
    """Build a SkillMetadata from a frontmatter dict.

    Missing fields default to None; ``tags`` is normalized to a tuple.
    ``name`` and ``description`` fall back to the file's basename and
    an empty string so the loader can't produce an invalid metadata.
    """
    tags = fm.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    return SkillMetadata(
        name=str(fm.get("name") or fallback_name).strip(),
        description=str(fm.get("description") or "").strip(),
        version=_str_or_none(fm.get("version")),
        maintainer=_str_or_none(fm.get("maintainer")),
        tags=tuple(str(t) for t in tags),
        display_name=_str_or_none(fm.get("display_name")),
        category=_str_or_none(fm.get("category")),
        skill_type=_str_or_none(fm.get("skill_type")),
        function=_str_or_none(fm.get("function")),
        breadth=_str_or_none(fm.get("breadth")),
        status=_str_or_none(_nested_get(fm, "execution", "status")),
    )


def _str_or_none(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _nested_get(d: dict, *keys):
    """Safely walk d[key1][key2][...]; return None on any miss."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# --------------------------------------------------------------------
# Discovery pattern extraction
# --------------------------------------------------------------------


# Match ``## Step 1`` or ``### Step 1`` up to end-of-line, tolerating
# various dash characters and content following "Step 1".
_STEP1_HEADING_RE = re.compile(
    r"^(#{2,3})\s*Step\s+1\b.*$",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_step1_content(body: str) -> Optional[str]:
    """Slice out the Step 1 section body.

    Returns the text between the Step 1 heading and the next heading
    of the same or higher level (``##`` or ``###`` depending on Step 1's
    own level). Returns None when no Step 1 heading exists.
    """
    m = _STEP1_HEADING_RE.search(body)
    if m is None:
        return None
    start = m.end()
    heading_level = len(m.group(1))  # 2 or 3
    # Look for the next heading of same or higher level.
    # We mask code block contents first so a "# comment" line inside a
    # bash block doesn't get mistaken for a markdown heading.
    remaining = body[start:]
    masked = _mask_code_blocks(remaining)
    next_pattern = re.compile(
        r"^#{1," + str(heading_level) + r"}\s+\S",
        re.MULTILINE,
    )
    nxt = next_pattern.search(masked)
    if nxt is None:
        return remaining
    return remaining[: nxt.start()]


def _mask_code_blocks(text: str) -> str:
    """Replace fenced code block content with spaces (line count preserved).

    Structure of the input is preserved: line count and offsets stay
    correct so downstream regex offsets are valid, but code-block
    contents (which may contain ``#`` for shell comments) can no
    longer match heading regexes.
    """
    out = list(text)
    i = 0
    while True:
        fence_open = text.find("```", i)
        if fence_open == -1:
            break
        fence_close = text.find("```", fence_open + 3)
        if fence_close == -1:
            break
        for j in range(fence_open, fence_close + 3):
            if out[j] != "\n":
                out[j] = " "
        i = fence_close + 3
    return "".join(out)


_BASH_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)?\s*\n(.*?)\n```",
    re.DOTALL,
)


def _extract_bash_blocks(section: str) -> List[str]:
    """Return all fenced bash blocks inside a section."""
    return [m.group(1) for m in _BASH_BLOCK_RE.finditer(section)]


def _extract_patterns_from_bash_block(block: str) -> List[DiscoveryPattern]:
    """Parse a bash block and extract grep commands as patterns.

    Handles:
    - Line continuations (``\\`` at end of line).
    - Preceding ``# comment`` treated as the pattern's description.
    - Multiple grep commands in one block.
    - ``grep`` commands lacking ``-E`` or ``--include`` are skipped.
    """
    # Normalize line continuations:
    #   ``grep ... \\\n  --include=...`` -> ``grep ...   --include=...``
    joined = re.sub(r"\\\s*\n\s*", " ", block)

    patterns: List[DiscoveryPattern] = []
    pending_comment: Optional[str] = None

    for raw_line in joined.splitlines():
        line = raw_line.strip()
        if not line:
            # Blank line resets the pending comment so a stale comment
            # doesn't attach to a grep several lines later.
            pending_comment = None
            continue
        if line.startswith("#"):
            pending_comment = line.lstrip("#").strip() or None
            continue
        if not line.startswith("grep"):
            pending_comment = None
            continue

        pattern = _parse_grep_command(line, description=pending_comment)
        pending_comment = None
        if pattern is not None:
            patterns.append(pattern)

    return patterns


def _parse_grep_command(
    cmd: str,
    *,
    description: Optional[str],
) -> Optional[DiscoveryPattern]:
    """Parse a single grep command line into a DiscoveryPattern.

    Uses shlex to handle shell quoting. Extracts:
      * The ``-E`` regex argument.
      * All ``--include=<glob>`` arguments; expands brace globs
        (``*.{py,java,cs}``) to a tuple of extensions.

    Returns None if the command doesn't have both a regex and at least
    one extension (a grep with no include filter would run on every
    file in the workspace, which is almost always unintentional in a
    skill discovery step).
    """
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError as e:
        _logger.debug("shlex.split failed on grep command: %s", e)
        return None

    if not parts or parts[0] != "grep":
        return None

    regex: Optional[str] = None
    extensions: List[str] = []
    i = 1
    while i < len(parts):
        p = parts[i]
        if p == "-E" and i + 1 < len(parts):
            regex = parts[i + 1]
            i += 2
            continue
        if p.startswith("--include="):
            extensions.extend(_expand_include_glob(p[len("--include="):]))
            i += 1
            continue
        if p == "--include" and i + 1 < len(parts):
            extensions.extend(_expand_include_glob(parts[i + 1]))
            i += 2
            continue
        i += 1

    if not regex:
        return None
    if not extensions:
        _logger.debug(
            "Skipping grep without --include (would run on all files): %s",
            cmd[:80],
        )
        return None

    return DiscoveryPattern(
        regex=regex,
        extensions=tuple(extensions),
        description=description,
    )


def _expand_include_glob(glob: str) -> List[str]:
    """Expand shell-style ``*.{py,java,cs}`` to ``['.py', '.java', '.cs']``.

    Also handles plain ``*.ext`` and bare ``.ext`` forms. Unknown
    formats return an empty list so the caller can decide whether to
    treat that as "no filter" (currently: skip the pattern).
    """
    glob = glob.strip()

    m = re.match(r"^\*\.\{([^}]+)\}$", glob)
    if m:
        return [
            f".{ext.strip().lower()}"
            for ext in m.group(1).split(",")
            if ext.strip()
        ]

    m = re.match(r"^\*\.(\w+)$", glob)
    if m:
        return [f".{m.group(1).lower()}"]

    m = re.match(r"^\.(\w+)$", glob)
    if m:
        return [f".{m.group(1).lower()}"]

    return []


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


class SkillLoader:
    """Discovers and loads skills from a directory.

    Skills are expected at ``<directory>/<name>/SKILL.md``. A flat
    ``<directory>/*.md`` layout also works for operators with a
    simpler organization; the name is derived from the filename stem
    in that case.

    The loader is stateless; load_all() re-reads every skill on every
    call so operator edits are picked up without restart.
    """

    def __init__(self, *, directory: Path) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def load_all(self) -> List[SkillDefinition]:
        """Load every skill under the directory.

        Order: alphabetical by skill name. Skills that fail to load
        (invalid file, unreadable) are logged and skipped. This never
        raises -- the caller gets whatever loaded successfully.
        """
        skill_files = list(self._iter_skill_files())
        results: List[SkillDefinition] = []
        for path in skill_files:
            try:
                skill = self.load_file(path)
            except Exception as e:
                _logger.warning("Failed to load skill from %s: %s", path, e)
                continue
            if skill is not None:
                results.append(skill)
        results.sort(key=lambda s: s.name)
        return results

    def load_file(self, path: Path) -> Optional[SkillDefinition]:
        """Load a single skill file.

        Returns None if the file can't be read. Raises no exceptions
        for the format-tolerance reasons discussed at the top of this
        module.
        """
        try:
            # utf-8-sig drops a byte-order mark. Windows editors often
            # save one, and a BOM before the opening "---" used to hide
            # the whole frontmatter -- a function=verify skill then
            # loaded as a runnable scan skill named after its file.
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as e:
            _logger.warning("Cannot read skill file %s: %s", path, e)
            return None

        fallback_name = self._name_from_path(path)
        fm, body = _parse_frontmatter(content)
        metadata = _metadata_from_frontmatter(fm, fallback_name=fallback_name)

        step1 = _extract_step1_content(body)
        patterns: Tuple[DiscoveryPattern, ...] = ()
        if step1 is not None:
            all_patterns: List[DiscoveryPattern] = []
            for block in _extract_bash_blocks(step1):
                all_patterns.extend(_extract_patterns_from_bash_block(block))
            patterns = tuple(all_patterns)

        return SkillDefinition(
            metadata=metadata,
            source_path=path,
            body=body,
            discovery_patterns=patterns,
        )

    # ---- internals ----

    def _iter_skill_files(self) -> Iterable[Path]:
        """Yield every SKILL.md and *.md under the configured directory.

        Preference: ``<root>/<name>/SKILL.md`` first, then flat
        ``<root>/<name>.md``. Files starting with ``_`` (like
        ``_shared/``) are skipped so shared reference material
        doesn't get loaded as skills.
        """
        if not self._directory.exists():
            return

        # Directory-per-skill: <root>/<name>/SKILL.md
        for sub in sorted(self._directory.iterdir()):
            if sub.name.startswith("_"):
                continue
            if sub.is_dir():
                skill_md = sub / "SKILL.md"
                if skill_md.exists():
                    yield skill_md
                    continue
                # Case-insensitive fallback for Windows / operators
                # using lowercase skill.md
                for md in sorted(sub.glob("*.md")):
                    if md.name.lower() == "skill.md":
                        yield md
                        break

        # Flat layout: <root>/<name>.md
        for md in sorted(self._directory.glob("*.md")):
            if md.name.lower() == "skill.md":
                continue  # already handled above
            if md.name.startswith("_"):
                continue
            yield md

    @staticmethod
    def _name_from_path(path: Path) -> str:
        """Derive a fallback skill name from a file path.

        ``.../injection-prevention/SKILL.md`` -> ``injection-prevention``
        ``.../code-review-diff.md``            -> ``code-review-diff``
        """
        if path.name.lower() == "skill.md":
            return path.parent.name
        return path.stem
