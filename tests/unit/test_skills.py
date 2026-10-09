"""Unit tests for the skill loading + discovery infrastructure (Phase 1a)."""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from falconeye.domain.models.skill import (
    DiscoveryHit,
    DiscoveryPattern,
    SkillDefinition,
    SkillMetadata,
)
from falconeye.infrastructure.skills.skill_loader import (
    SkillLoader,
    _expand_include_glob,
    _extract_bash_blocks,
    _extract_patterns_from_bash_block,
    _extract_step1_content,
    _parse_frontmatter,
    _parse_grep_command,
)
from falconeye.infrastructure.skills.grep_runner import GrepRunner


# =====================================================================
# _parse_frontmatter
# =====================================================================


@pytest.mark.unit
def test_parse_frontmatter_valid():
    content = textwrap.dedent("""\
    ---
    name: test-skill
    description: A test skill
    tags:
    - foo
    - bar
    ---
    # Body starts here
    """)
    fm, body = _parse_frontmatter(content)
    assert fm["name"] == "test-skill"
    assert fm["description"] == "A test skill"
    assert fm["tags"] == ["foo", "bar"]
    assert body.startswith("# Body starts here")


@pytest.mark.unit
def test_parse_frontmatter_missing_returns_empty_dict():
    """A skill file with no frontmatter shouldn't fail the loader."""
    content = "# Just a title, no frontmatter"
    fm, body = _parse_frontmatter(content)
    assert fm == {}
    assert body == content


@pytest.mark.unit
def test_parse_frontmatter_malformed_yaml_returns_empty_dict():
    """Broken YAML degrades gracefully to empty metadata."""
    content = "---\nname: :::: not yaml :\n---\n\nbody"
    fm, body = _parse_frontmatter(content)
    # Body is preserved because we couldn't parse frontmatter
    assert fm == {} or fm.get("name") is None


# =====================================================================
# _extract_step1_content
# =====================================================================


@pytest.mark.unit
def test_extract_step1_finds_h2_section():
    body = textwrap.dedent("""\
    ## Step 1 -- Discover code

    ```bash
    grep -rn -E "foo" --include="*.py" .
    ```

    ## Step 2 -- Evaluate
    """)
    section = _extract_step1_content(body)
    assert section is not None
    assert "grep -rn -E" in section
    assert "Step 2" not in section


@pytest.mark.unit
def test_extract_step1_finds_h3_section():
    """### Step 1 (used by some skills) also works."""
    body = textwrap.dedent("""\
    ## Something

    ### Step 1 -- Discover

    ```bash
    grep -rn -E "foo" --include="*.py" .
    ```

    ### Step 2
    """)
    section = _extract_step1_content(body)
    assert section is not None
    assert "grep" in section
    assert "Step 2" not in section


@pytest.mark.unit
def test_extract_step1_absent_returns_none():
    body = "# Just a document with no Step 1 heading"
    assert _extract_step1_content(body) is None


@pytest.mark.unit
def test_extract_step1_stops_at_higher_level_heading():
    """A ### Step 1 shouldn't consume text past the next ## heading."""
    body = textwrap.dedent("""\
    ## First
    ### Step 1
    inside step 1
    ## Second
    outside step 1
    """)
    section = _extract_step1_content(body)
    assert section is not None
    assert "inside step 1" in section
    assert "outside step 1" not in section


@pytest.mark.unit
def test_extract_step1_masks_shell_comments_inside_code_blocks():
    """A '# comment' line INSIDE a bash block must not be treated as a
    markdown heading that ends the Step 1 section."""
    body = textwrap.dedent("""\
    ## Step 1
    ```bash
    # This looks like a heading but is a shell comment
    grep -rn -E "foo" --include="*.py" .
    ```
    ## Step 2
    """)
    section = _extract_step1_content(body)
    assert section is not None
    assert "grep -rn" in section


# =====================================================================
# _extract_bash_blocks & _extract_patterns_from_bash_block
# =====================================================================


@pytest.mark.unit
def test_extract_bash_blocks_finds_multiple():
    section = textwrap.dedent("""\
    Some text.

    ```bash
    grep foo
    ```

    Other text.

    ```
    grep bar
    ```
    """)
    blocks = _extract_bash_blocks(section)
    assert len(blocks) == 2
    assert "grep foo" in blocks[0]
    assert "grep bar" in blocks[1]


@pytest.mark.unit
def test_pattern_extraction_with_line_continuations():
    """Real skills wrap long grep commands with backslashes."""
    block = textwrap.dedent("""\
    # SQL injection detection
    grep -rn -E "(cursor\\.execute|\\.query\\()" \\
      --include="*.{py,java,cs}" .
    """)
    patterns = _extract_patterns_from_bash_block(block)
    assert len(patterns) == 1
    p = patterns[0]
    assert p.description == "SQL injection detection"
    assert set(p.extensions) == {".py", ".java", ".cs"}
    assert "cursor" in p.regex


@pytest.mark.unit
def test_pattern_extraction_multiple_greps_with_comments():
    """Each grep gets its own preceding # comment as description."""
    block = textwrap.dedent("""\
    # SQL injection
    grep -rn -E "sql_pattern" --include="*.py" .

    # Command injection
    grep -rn -E "cmd_pattern" --include="*.py" .
    """)
    patterns = _extract_patterns_from_bash_block(block)
    assert len(patterns) == 2
    assert patterns[0].description == "SQL injection"
    assert patterns[1].description == "Command injection"


@pytest.mark.unit
def test_pattern_extraction_blank_line_resets_comment():
    """A comment that has a blank line before the grep shouldn't attach."""
    block = textwrap.dedent("""\
    # unrelated comment

    grep -rn -E "foo" --include="*.py" .
    """)
    patterns = _extract_patterns_from_bash_block(block)
    assert len(patterns) == 1
    assert patterns[0].description is None


@pytest.mark.unit
def test_pattern_extraction_non_grep_lines_ignored():
    block = textwrap.dedent("""\
    echo hello
    ls -la
    # This is a comment
    grep -rn -E "foo" --include="*.py" .
    """)
    patterns = _extract_patterns_from_bash_block(block)
    assert len(patterns) == 1


@pytest.mark.unit
def test_grep_without_include_skipped():
    """A grep with no --include would scan every file; skip it."""
    p = _parse_grep_command(
        'grep -rn -E "foo" .',
        description=None,
    )
    assert p is None


@pytest.mark.unit
def test_grep_without_E_flag_skipped():
    """No -E means no regex we can extract."""
    p = _parse_grep_command(
        'grep -rn "foo" --include="*.py" .',
        description=None,
    )
    assert p is None


# =====================================================================
# _expand_include_glob
# =====================================================================


@pytest.mark.unit
def test_expand_brace_glob():
    assert set(_expand_include_glob("*.{py,java,cs}")) == {".py", ".java", ".cs"}


@pytest.mark.unit
def test_expand_single_extension_glob():
    assert _expand_include_glob("*.py") == [".py"]


@pytest.mark.unit
def test_expand_bare_extension():
    assert _expand_include_glob(".py") == [".py"]


@pytest.mark.unit
def test_expand_unknown_glob_form():
    assert _expand_include_glob("weird_thing") == []


@pytest.mark.unit
def test_expand_glob_normalizes_case():
    assert set(_expand_include_glob("*.{PY,JAVA}")) == {".py", ".java"}


@pytest.mark.unit
def test_expand_glob_filters_empty_entries():
    """Trailing commas in *.{py,} produce an empty entry that we skip."""
    assert set(_expand_include_glob("*.{py, , java}")) == {".py", ".java"}


# =====================================================================
# SkillLoader end-to-end
# =====================================================================


def _write_skill(root: Path, name: str, content: str) -> Path:
    sub = root / name
    sub.mkdir(exist_ok=True)
    p = sub / "SKILL.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.mark.unit
def test_loader_loads_directory_layout(tmp_path):
    """<root>/<name>/SKILL.md layout works."""
    _write_skill(tmp_path, "alpha", textwrap.dedent("""\
    ---
    name: alpha
    description: alpha skill
    ---
    body
    """))
    _write_skill(tmp_path, "beta", textwrap.dedent("""\
    ---
    name: beta
    description: beta skill
    ---
    body
    """))
    skills = SkillLoader(directory=tmp_path).load_all()
    assert [s.name for s in skills] == ["alpha", "beta"]


@pytest.mark.unit
def test_loader_flat_layout(tmp_path):
    """<root>/<name>.md layout also works."""
    (tmp_path / "alpha.md").write_text(textwrap.dedent("""\
    ---
    name: alpha
    description: flat layout
    ---
    body
    """))
    skills = SkillLoader(directory=tmp_path).load_all()
    assert len(skills) == 1
    assert skills[0].name == "alpha"


@pytest.mark.unit
def test_loader_skips_underscore_prefixed(tmp_path):
    """_shared/, _internal/, etc. are reserved for non-skill content."""
    _write_skill(tmp_path, "real", textwrap.dedent("""\
    ---
    name: real
    description: x
    ---
    """))
    (tmp_path / "_shared").mkdir()
    (tmp_path / "_shared" / "template.md").write_text("shared content")
    (tmp_path / "_notes.md").write_text("---\nname: notes\n---\n")
    skills = SkillLoader(directory=tmp_path).load_all()
    assert [s.name for s in skills] == ["real"]


@pytest.mark.unit
def test_loader_missing_directory_returns_empty(tmp_path):
    """Nonexistent directory doesn't raise; caller gets []."""
    loader = SkillLoader(directory=tmp_path / "does-not-exist")
    assert loader.load_all() == []


@pytest.mark.unit
def test_loader_fallback_name_from_path(tmp_path):
    """Skill with no name in frontmatter takes name from directory."""
    _write_skill(tmp_path, "path-derived", textwrap.dedent("""\
    ---
    description: no name field
    ---
    body
    """))
    skills = SkillLoader(directory=tmp_path).load_all()
    assert skills[0].name == "path-derived"


@pytest.mark.unit
def test_loader_extracts_patterns_from_step1(tmp_path):
    _write_skill(tmp_path, "grep-skill", textwrap.dedent("""\
    ---
    name: grep-skill
    description: has patterns
    ---

    # A skill

    ## Step 1 -- Discover

    ```bash
    # find sql
    grep -rn -E "SELECT.*\\+" --include="*.{py,java}" .

    # find cmd
    grep -rn -E "shell=True" --include="*.py" .
    ```

    ## Step 2 -- Evaluate
    """))
    skills = SkillLoader(directory=tmp_path).load_all()
    assert len(skills) == 1
    patterns = skills[0].discovery_patterns
    assert len(patterns) == 2
    assert patterns[0].description == "find sql"
    assert set(patterns[0].extensions) == {".py", ".java"}
    assert "SELECT" in patterns[0].regex
    assert patterns[1].description == "find cmd"


@pytest.mark.unit
def test_loader_extracts_execution_status(tmp_path):
    """execution.status from frontmatter surfaces on SkillMetadata."""
    _write_skill(tmp_path, "s", textwrap.dedent("""\
    ---
    name: s
    description: x
    execution:
      status: experimental
    ---
    """))
    skills = SkillLoader(directory=tmp_path).load_all()
    assert skills[0].metadata.status == "experimental"


@pytest.mark.unit
def test_is_runnable_today_only_scan_function(tmp_path):
    """scan -> runnable. verify/triage -> deferred. Missing -> default scan."""
    for fn, expected in [
        ("scan", True),
        ("verify", False),
        ("triage", False),
        ("research", False),
        (None, True),  # missing function defaults to scan
    ]:
        _write_skill(tmp_path, f"skill-{fn or 'default'}", textwrap.dedent(f"""\
        ---
        name: skill-{fn or 'default'}
        description: x
        {f'function: {fn}' if fn else ''}
        ---
        """))
    skills = {s.name: s for s in SkillLoader(directory=tmp_path).load_all()}
    assert skills["skill-scan"].is_runnable_today is True
    assert skills["skill-verify"].is_runnable_today is False
    assert skills["skill-triage"].is_runnable_today is False
    assert skills["skill-research"].is_runnable_today is False
    assert skills["skill-default"].is_runnable_today is True


# =====================================================================
# GrepRunner
# =====================================================================


@pytest.mark.unit
def test_grep_runner_discovers_matches(tmp_path):
    (tmp_path / "vuln.py").write_text(
        "def q(uid): cursor.execute('SELECT * FROM u WHERE id = ' + uid)"
    )
    pattern = DiscoveryPattern(
        regex=r"cursor\.execute",
        extensions=(".py",),
        description="sql exec",
    )
    hits = GrepRunner(workspace_root=tmp_path).discover([pattern])
    assert len(hits) == 1
    h = hits[0]
    assert h.line_number == 1
    assert h.pattern_description == "sql exec"
    assert "cursor.execute" in h.line_text


@pytest.mark.unit
def test_grep_runner_filters_by_extension(tmp_path):
    """Only files whose suffix is in pattern.extensions are scanned."""
    (tmp_path / "match.py").write_text("dangerous_thing")
    (tmp_path / "skip.js").write_text("dangerous_thing")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="dangerous_thing", extensions=(".py",)),
    ])
    assert len(hits) == 1
    assert hits[0].file_path.endswith("match.py")


@pytest.mark.unit
def test_grep_runner_skips_default_excluded_dirs(tmp_path):
    """bin/, node_modules/, etc. are always skipped."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("dangerous_thing")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "bad.py").write_text("dangerous_thing")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "also_bad.py").write_text("dangerous_thing")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="dangerous_thing", extensions=(".py",)),
    ])
    assert len(hits) == 1
    assert "src" in hits[0].file_path


@pytest.mark.unit
def test_grep_runner_caps_hits_per_pattern(tmp_path):
    """When one pattern matches many times, cap kicks in."""
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text("hit line")
    hits = GrepRunner(
        workspace_root=tmp_path,
        max_hits_per_pattern=10,
    ).discover([
        DiscoveryPattern(regex="hit line", extensions=(".py",)),
    ])
    assert len(hits) == 10


@pytest.mark.unit
def test_grep_runner_caps_total_hits(tmp_path):
    """Total cap stops discovery across patterns even if per-pattern isn't hit."""
    for i in range(3):
        (tmp_path / f"f{i}.py").write_text("alpha\nbeta\ngamma")
    patterns = [
        DiscoveryPattern(regex="alpha", extensions=(".py",)),
        DiscoveryPattern(regex="beta", extensions=(".py",)),
        DiscoveryPattern(regex="gamma", extensions=(".py",)),
    ]
    hits = GrepRunner(
        workspace_root=tmp_path,
        max_hits_per_pattern=10,
        max_total_hits=5,
    ).discover(patterns)
    assert len(hits) == 5


@pytest.mark.unit
def test_grep_runner_invalid_regex_skipped(tmp_path):
    """A Python-invalid regex from a skill is skipped, others still run."""
    (tmp_path / "f.py").write_text("good_line")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="[invalid", extensions=(".py",)),
        DiscoveryPattern(regex="good_line", extensions=(".py",)),
    ])
    assert len(hits) == 1
    assert "good_line" in hits[0].line_text


@pytest.mark.unit
def test_grep_runner_pattern_without_extensions_skipped(tmp_path):
    """A pattern with empty extensions produces no hits."""
    (tmp_path / "f.py").write_text("x")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="x", extensions=()),
    ])
    assert hits == []


@pytest.mark.unit
def test_grep_runner_case_insensitive_extension_matching(tmp_path):
    """File.CS on Windows should match extensions=('.cs',)."""
    (tmp_path / "MyFile.CS").write_text("target")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="target", extensions=(".cs",)),
    ])
    assert len(hits) == 1


@pytest.mark.unit
def test_grep_runner_matched_span_correct(tmp_path):
    """matched_span records the column range where the regex matched."""
    (tmp_path / "f.py").write_text("prefix TARGET suffix")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="TARGET", extensions=(".py",)),
    ])
    assert len(hits) == 1
    start, end = hits[0].matched_span
    assert hits[0].line_text[start:end] == "TARGET"


@pytest.mark.unit
def test_grep_runner_walks_subdirectories(tmp_path):
    """Recursive walk finds files in nested directories."""
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "deep.py").write_text("found")
    hits = GrepRunner(workspace_root=tmp_path).discover([
        DiscoveryPattern(regex="found", extensions=(".py",)),
    ])
    assert len(hits) == 1


# =====================================================================
# CLI: skill_list_command
# =====================================================================


def _make_config(root: Path, enabled: bool, skills_relpath: str = "skills") -> Path:
    """Write a minimal config.yaml that either enables or disables skills."""
    cfg = root / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
    llm:
      provider: litellm
    skills:
      enabled: {'true' if enabled else 'false'}
      directory: ./{skills_relpath}
    """))
    return cfg


@pytest.mark.unit
def test_cli_list_disabled_exits_2(tmp_path):
    from falconeye.adapters.cli.commands_skill import skill_list_command
    cfg = _make_config(tmp_path, enabled=False)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    with pytest.raises(SystemExit) as exc:
        skill_list_command(config_path=cfg, console=console, show_patterns=False)
    assert exc.value.code == 2
    assert "not enabled" in buf.getvalue().lower()


@pytest.mark.unit
def test_cli_list_missing_directory_exits_2(tmp_path):
    from falconeye.adapters.cli.commands_skill import skill_list_command
    cfg = _make_config(tmp_path, enabled=True)  # skills/ doesn't exist
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    with pytest.raises(SystemExit) as exc:
        skill_list_command(config_path=cfg, console=console, show_patterns=False)
    assert exc.value.code == 2
    assert "does not exist" in buf.getvalue().lower()


@pytest.mark.unit
def test_cli_list_empty_directory_returns_normally(tmp_path):
    """Empty skills directory prints a helpful message, doesn't crash."""
    from falconeye.adapters.cli.commands_skill import skill_list_command
    (tmp_path / "skills").mkdir()
    cfg = _make_config(tmp_path, enabled=True)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    skill_list_command(config_path=cfg, console=console, show_patterns=False)
    assert "no skills found" in buf.getvalue().lower()


@pytest.mark.unit
def test_cli_list_shows_all_skills(tmp_path):
    from falconeye.adapters.cli.commands_skill import skill_list_command
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "aud1", textwrap.dedent("""\
    ---
    name: aud1
    description: first
    category: web-api-security
    function: scan
    ---

    ## Step 1

    ```bash
    grep -rn -E "danger" --include="*.py" .
    ```
    """))
    _write_skill(skills_dir, "llm1", textwrap.dedent("""\
    ---
    name: llm1
    description: scan skill without Step 1 patterns
    function: scan
    ---
    """))
    _write_skill(skills_dir, "vfy1", textwrap.dedent("""\
    ---
    name: vfy1
    description: verifier
    category: triage-verification
    function: verify
    ---
    """))
    cfg = _make_config(tmp_path, enabled=True)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    skill_list_command(config_path=cfg, console=console, show_patterns=False)
    out = buf.getvalue()
    assert "aud1" in out
    assert "vfy1" in out
    # scan skill with patterns is runnable
    assert "yes" in out
    # scan skill without patterns is rejected by `skill run`, so it
    # must not claim "yes"
    assert "needs patterns" in out
    # verify skill is deferred
    assert "deferred" in out


@pytest.mark.unit
def test_cli_list_show_patterns_expands_details(tmp_path):
    from falconeye.adapters.cli.commands_skill import skill_list_command
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "s1", textwrap.dedent("""\
    ---
    name: s1
    description: with patterns
    ---

    ## Step 1

    ```bash
    # find dangerous stuff
    grep -rn -E "danger" --include="*.py" .
    ```
    """))
    cfg = _make_config(tmp_path, enabled=True)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    skill_list_command(config_path=cfg, console=console, show_patterns=True)
    out = buf.getvalue()
    assert "find dangerous stuff" in out
    assert "danger" in out
    assert ".py" in out


# =====================================================================
# Regression check against operator-uploaded skills
# =====================================================================


@pytest.mark.unit
def test_operator_uploaded_injection_prevention_extracts_patterns(tmp_path, operator_skill):
    """Real skill from operator uploads: injection-prevention.md.

    Verifies the loader handles em-dashes in headings, complex regexes
    with quotes-in-quotes, and multi-language --include globs.
    """
    upload = operator_skill("injection-prevention")
    (tmp_path / "injection-prevention").mkdir()
    (tmp_path / "injection-prevention" / "SKILL.md").write_text(
        upload.read_text(encoding="utf-8"), encoding="utf-8",
    )
    skills = SkillLoader(directory=tmp_path).load_all()
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "injection-prevention"
    assert s.metadata.category == "web-api-security"
    assert s.metadata.function == "scan"
    assert s.is_runnable_today is True
    # The skill has 9 grep commands in Step 1.
    assert len(s.discovery_patterns) == 9
    # Descriptions were extracted from the # comments preceding each grep.
    descriptions = [p.description for p in s.discovery_patterns]
    assert any("SQL injection" in (d or "") for d in descriptions)
    assert any("Command injection" in (d or "") for d in descriptions)


@pytest.mark.unit
def test_operator_uploaded_input_validation_zero_patterns(tmp_path, operator_skill):
    """input-validation.md has no grep commands in Step 1 (LLM-only skill).

    Confirms the loader accepts this shape and produces a runnable skill
    with zero patterns rather than failing to load.
    """
    upload = operator_skill("input-validation")
    (tmp_path / "input-validation").mkdir()
    (tmp_path / "input-validation" / "SKILL.md").write_text(
        upload.read_text(encoding="utf-8"), encoding="utf-8",
    )
    skills = SkillLoader(directory=tmp_path).load_all()
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "input-validation"
    assert s.is_runnable_today is True
    # No grep commands in Step 1 -> zero patterns.
    assert len(s.discovery_patterns) == 0


@pytest.mark.unit
def test_operator_uploaded_fp_check_marked_deferred(tmp_path, operator_skill):
    """fp-check has function=verify -> deferred to Phase 3, not runnable now."""
    upload = operator_skill("fp-check")
    (tmp_path / "fp-check").mkdir()
    (tmp_path / "fp-check" / "SKILL.md").write_text(
        upload.read_text(encoding="utf-8"), encoding="utf-8",
    )
    s = SkillLoader(directory=tmp_path).load_all()[0]
    assert s.metadata.function == "verify"
    assert s.is_runnable_today is False


@pytest.mark.unit
def test_operator_uploaded_triage_marked_deferred(tmp_path, operator_skill):
    upload = operator_skill("triage")
    (tmp_path / "triage").mkdir()
    (tmp_path / "triage" / "SKILL.md").write_text(
        upload.read_text(encoding="utf-8"), encoding="utf-8",
    )
    s = SkillLoader(directory=tmp_path).load_all()[0]
    assert s.metadata.function == "triage"
    assert s.is_runnable_today is False
