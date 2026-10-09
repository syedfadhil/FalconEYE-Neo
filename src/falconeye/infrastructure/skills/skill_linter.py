"""
Skill linter: static diagnostics for operator-authored skill files.

The loader and grep runner are deliberately tolerant (see
:mod:`skill_loader`): a grep command the loader can't use is skipped
with a debug log, a regex Python can't compile is skipped by
GrepRunner, and malformed frontmatter falls back to defaults. That
keeps ``skill run`` going, but it also means an operator never learns
that part of a skill is dead. ``falconeye skill lint`` reports all of
it without calling an LLM.

The diagnosis reuses the loader's own parsing helpers, so lint
describes exactly what the loader does rather than a second
interpretation of the format.

Most pattern checks target the gap between what skill authors test
with (GNU ``grep -E``) and what FalconEYE executes (Python ``re``).
Each corresponds to a way a pattern silently stops matching:

* ``\\|`` between words is a literal pipe in both grep -E and Python,
  so ``decode\\|shelve`` never acts as alternation. The operator's
  deserialization-xxe skill has this.
* POSIX bracket classes (``[[:alpha:]]``) are not supported by Python.
* GNU word boundaries (``\\<`` / ``\\>``) are literal ``<`` / ``>`` in
  Python.
* A pattern that matches the empty string hits every line.
"""

from __future__ import annotations

import re
import shlex
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import yaml

from ...domain.models.skill import SkillDefinition
from .skill_loader import (
    _FRONTMATTER_RE,
    _expand_include_glob,
    _extract_bash_blocks,
    _extract_step1_content,
    _parse_frontmatter,
    _parse_grep_command,
)


LINT_ERROR = "error"
LINT_WARNING = "warning"
LINT_INFO = "info"

_KNOWN_FUNCTIONS = ("scan", "generate", "verify", "triage", "research")

# `\|` with a word, `)` or `]` before it and a word, `(` or `[` after
# it: alternation written BRE-style. A deliberate literal pipe (e.g.
# `\|\|` for a logical-or) doesn't have word characters on both sides.
_BRE_ALTERNATION = re.compile(r"(?<=[\w)\]])\\\|(?=[\w(\[])")
_POSIX_CLASS = re.compile(
    r"\[:(?:alnum|alpha|blank|cntrl|digit|graph|lower|print|punct|space"
    r"|upper|word|xdigit):\]"
)
_GNU_BOUNDARY = re.compile(r"(?<!\\)\\[<>]")
_OTHER_SEARCH_COMMAND = re.compile(r"^(?:egrep|fgrep|rg|ag|ack|git\s+grep|find)\b")
_COMBINED_E_FLAG = re.compile(r"-[A-Za-z]*E[A-Za-z]*")


@dataclass(frozen=True)
class LintIssue:
    """One diagnostic. ``code`` is a stable kebab-case identifier."""
    skill: str
    level: str
    code: str
    message: str


def lint_skills(skills: Sequence[SkillDefinition]) -> List[LintIssue]:
    """Lint every skill, plus checks that need the whole library."""
    issues: List[LintIssue] = []
    for skill in skills:
        issues.extend(lint_skill(skill))

    by_name = defaultdict(list)
    for skill in skills:
        by_name[skill.name].append(skill)
    for name, group in sorted(by_name.items()):
        if len(group) > 1:
            paths = ", ".join(str(s.source_path) for s in group)
            issues.append(LintIssue(
                name, LINT_ERROR, "duplicate-name",
                f"{len(group)} skills share this name ({paths}); "
                f"`skill run {name}` only ever uses the first.",
            ))
    return issues


def lint_skill(skill: SkillDefinition) -> List[LintIssue]:
    """Lint a single loaded skill (re-reading its source for context)."""
    try:
        content = skill.source_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        return [LintIssue(
            skill.name, LINT_ERROR, "unreadable",
            f"cannot read {skill.source_path}: {e}",
        )]
    frontmatter, _ = _parse_frontmatter(content)
    return (
        _check_frontmatter(skill, content)
        + _check_metadata(skill, frontmatter)
        + _check_patterns(skill)
        + _check_commands(skill)
        + _check_targets(skill, frontmatter)
    )


# --------------------------------------------------------------------
# Frontmatter and metadata
# --------------------------------------------------------------------


_METADATA_LOST = (
    "all metadata is ignored, so the skill loads as function=scan "
    "and is named after its file"
)


def _check_frontmatter(skill: SkillDefinition, content: str) -> List[LintIssue]:
    if not content.startswith("---"):
        return [LintIssue(
            skill.name, LINT_INFO, "no-frontmatter",
            "no YAML frontmatter: the name comes from the file name and "
            "function defaults to scan.",
        )]
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return [LintIssue(
            skill.name, LINT_ERROR, "frontmatter-unterminated",
            f"frontmatter opens with '---' but never closes; {_METADATA_LOST}.",
        )]
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        first_line = str(e).strip().splitlines()[0]
        return [LintIssue(
            skill.name, LINT_ERROR, "frontmatter-invalid",
            f"frontmatter is not valid YAML ({first_line}); {_METADATA_LOST}.",
        )]
    if data is not None and not isinstance(data, dict):
        return [LintIssue(
            skill.name, LINT_ERROR, "frontmatter-invalid",
            f"frontmatter is not a YAML mapping; {_METADATA_LOST}.",
        )]
    return []


def _check_metadata(skill: SkillDefinition, frontmatter: dict) -> List[LintIssue]:
    issues: List[LintIssue] = []
    fn = (skill.metadata.function or "scan").strip().lower()
    if fn not in _KNOWN_FUNCTIONS:
        issues.append(LintIssue(
            skill.name, LINT_WARNING, "function-unknown",
            f"function={skill.metadata.function!r} is not one of "
            f"{', '.join(_KNOWN_FUNCTIONS)}; the skill is listed but never run.",
        ))
        return issues
    if fn in ("verify", "triage"):
        stage = "fp-check" if fn == "verify" else "triage"
        issues.append(LintIssue(
            skill.name, LINT_INFO, "deferred",
            f"function={fn} routes to the {stage} stage, which is not built "
            "yet; skill run and run-all skip it.",
        ))
        return issues
    if fn != "scan":
        issues.append(LintIssue(
            skill.name, LINT_INFO, "not-executable",
            f"function={fn} is not executed by FalconEYE.",
        ))
        return issues

    skill_type = skill.metadata.skill_type
    if skill_type and skill_type.strip().lower() != "llm-reasoning":
        issues.append(LintIssue(
            skill.name, LINT_WARNING, "skill-type",
            f"skill_type={skill_type}: FalconEYE runs scan skills as LLM "
            "reasoning over the discovered code; it will not invoke the tools "
            "or agents this skill expects.",
        ))
    execution = frontmatter.get("execution")
    mode = execution.get("mode") if isinstance(execution, dict) else None
    if mode and str(mode).strip().lower() not in ("local-llm", "n/a"):
        issues.append(LintIssue(
            skill.name, LINT_WARNING, "execution-mode",
            f"execution.mode={mode}: FalconEYE executes skills with its "
            "configured LLM only; sandbox execution is not supported.",
        ))
    return issues


# --------------------------------------------------------------------
# Discovery patterns
# --------------------------------------------------------------------


def _check_patterns(skill: SkillDefinition) -> List[LintIssue]:
    issues: List[LintIssue] = []
    for i, p in enumerate(skill.discovery_patterns):
        label = f"pattern [{i}]" + (f" ({p.description})" if p.description else "")

        def warn(code: str, message: str) -> None:
            issues.append(LintIssue(skill.name, LINT_WARNING, code, f"{label}: {message}"))

        try:
            with warnings.catch_warnings():
                # Python warns "Possible nested set" on POSIX classes;
                # the posix-class check below reports it properly.
                warnings.simplefilter("ignore", FutureWarning)
                compiled = re.compile(p.regex)
        except re.error as e:
            hint = ""
            if "\\" in p.regex and "character set" in str(e):
                hint = (
                    " Inside [...] Python treats '\\' as an escape while grep -E "
                    "treats it literally; write '\\\\' for a literal backslash."
                )
            issues.append(LintIssue(
                skill.name, LINT_ERROR, "regex-invalid",
                f"{label}: not a valid Python regex ({e}); the pattern is "
                f"skipped at run time.{hint}",
            ))
            continue

        m = _BRE_ALTERNATION.search(p.regex)
        if m is not None:
            snippet = p.regex[max(0, m.start() - 20): m.end() + 20]
            warn(
                "bre-alternation",
                f"'\\|' is a literal '|' in grep -E and in Python, so "
                f"'...{snippet}...' never acts as alternation. Use a bare '|'.",
            )
        if _POSIX_CLASS.search(p.regex):
            warn(
                "posix-class",
                "POSIX classes like [[:alpha:]] are not supported by Python's "
                "re (it reads them as a set of literal characters); use \\w, "
                "\\d, \\s or an explicit range.",
            )
        if _GNU_BOUNDARY.search(p.regex):
            warn(
                "gnu-word-boundary",
                "GNU grep reads '\\<' / '\\>' as word boundaries, Python as a "
                "literal '<' / '>'; use \\b.",
            )
        if compiled.search("") is not None:
            warn(
                "matches-everything",
                "matches the empty string, so every line of every scanned "
                "file is a hit.",
            )
    return issues


# --------------------------------------------------------------------
# Step 1 commands
# --------------------------------------------------------------------


def _command_lines(block: str) -> Iterator[str]:
    """Non-comment command lines, with continuations joined like the loader."""
    joined = re.sub(r"\\\s*\n\s*", " ", block)
    for raw in joined.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def _why_grep_ignored(cmd: str) -> str:
    """Explain why the loader's _parse_grep_command rejected ``cmd``."""
    try:
        parts = shlex.split(cmd, posix=True)
    except ValueError as e:
        return f"cannot be parsed ({e})"
    if not parts or parts[0] != "grep":
        return "not a plain `grep` command"
    if "-E" not in parts:
        if any(_COMBINED_E_FLAG.fullmatch(p) for p in parts[1:]):
            return (
                "combined flags such as -rnE: the loader needs -E as its own "
                "argument (`grep -rn -E ...`)"
            )
        return "no -E flag: only extended-regex (-E) greps become patterns"
    if parts.index("-E") + 1 >= len(parts):
        return "-E is not followed by a pattern"
    includes = [p[len("--include="):] for p in parts if p.startswith("--include=")]
    includes += [parts[i + 1] for i, p in enumerate(parts[:-1]) if p == "--include"]
    if not includes:
        return (
            "no --include filter: unfiltered greps are skipped rather than "
            "run on every file"
        )
    if not any(_expand_include_glob(g) for g in includes):
        return (
            f"--include glob {', '.join(includes)} is not understood "
            "(use *.ext or *.{a,b})"
        )
    return "not usable by the loader"


def _check_commands(skill: SkillDefinition) -> List[LintIssue]:
    issues: List[LintIssue] = []
    step1 = _extract_step1_content(skill.body)
    step1_blocks = _extract_bash_blocks(step1) if step1 is not None else []

    step1_greps = 0
    for block in step1_blocks:
        for line in _command_lines(block):
            if line.startswith("grep"):
                step1_greps += 1
                if _parse_grep_command(line, description=None) is None:
                    issues.append(LintIssue(
                        skill.name, LINT_WARNING, "grep-ignored",
                        f"Step 1 command ignored by the loader, "
                        f"{_why_grep_ignored(line)}: {line[:80]}",
                    ))
            elif _OTHER_SEARCH_COMMAND.match(line):
                issues.append(LintIssue(
                    skill.name, LINT_WARNING, "unsupported-command",
                    f"Step 1 command ignored, only `grep -E` commands become "
                    f"discovery patterns: {line[:80]}",
                ))

    all_greps = sum(
        1
        for block in _extract_bash_blocks(skill.body)
        for line in _command_lines(block)
        if line.startswith("grep")
    )
    outside = all_greps - step1_greps
    if outside > 0:
        issues.append(LintIssue(
            skill.name, LINT_WARNING, "grep-outside-step1",
            f"{outside} grep command{'s' if outside != 1 else ''} outside a "
            "'Step 1' section; the loader only reads Step 1.",
        ))

    if skill.is_runnable_today and not skill.discovery_patterns:
        reason = (
            "no 'Step 1' heading" if step1 is None
            else "Step 1 has no ```bash``` grep command the loader can use"
        )
        scope = (
            f"the files matching targets ({', '.join(skill.metadata.targets)})"
            if skill.metadata.targets
            else "every enabled-language source file"
        )
        issues.append(LintIssue(
            skill.name, LINT_INFO, "llm-only",
            f"no discovery patterns ({reason}): skill run sweeps {scope}, "
            "one LLM call each up to --max-files; run-all includes it only "
            "with --include-llm-only.",
        ))
    return issues


# --------------------------------------------------------------------
# Sweep targets
# --------------------------------------------------------------------


def _check_targets(skill: SkillDefinition, frontmatter: dict) -> List[LintIssue]:
    if "targets" not in frontmatter:
        return []
    raw = frontmatter["targets"]
    entries = [raw] if isinstance(raw, str) else raw
    if not isinstance(entries, list) or not all(
        isinstance(t, str) and t.strip() for t in entries
    ):
        return [LintIssue(
            skill.name, LINT_WARNING, "targets-invalid",
            "targets must be a filename glob or a list of them (e.g. "
            "[package.json, '*.csproj']); non-string or empty entries are "
            f"ignored, leaving {list(skill.metadata.targets) or 'none'}.",
        )]
    if skill.discovery_patterns:
        return [LintIssue(
            skill.name, LINT_INFO, "targets-ignored",
            "targets only selects files for a sweep; this skill has "
            "discovery patterns, so grep hits pick its files instead.",
        )]
    return []
