"""
`falconeye skill ...` CLI commands: list, lint, run, run-all.

This module implements the plumbing invoked by the thin wrappers in
main.py. Following the pattern established by the other commands_*.py
modules: main.py owns the typer surface, this module owns the logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...infrastructure.di.container import DIContainer


def _resolve_skill_loader(
    *,
    config_path: Optional[Path],
    skills_dir_override: Optional[Path],
    console: Console,
):
    """Return the SkillLoader every ``skill`` subcommand works from.

    ``--skills-dir`` on the command line bypasses config gating: the
    operator explicitly pointed at a directory, so honor it even if
    config.skills.enabled=false. Otherwise the container decides, and
    a disabled config exits 2 with a message naming all three ways to
    enable skills.
    """
    # Import here (not at module top) because SkillLoader lives in the
    # infrastructure layer and the CLI adapter should stay import-light.
    from ...infrastructure.skills.skill_loader import SkillLoader

    if skills_dir_override is not None:
        return SkillLoader(directory=skills_dir_override.expanduser())
    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    if container.skill_loader is None:
        console.print("[red]Skills are not enabled in the config.[/red]")
        console.print(
            "Set [bold]config.skills.enabled: true[/bold] and configure "
            "[bold]config.skills.directory[/bold] to point at your skills folder, "
            "set the [bold]FALCONEYE_SKILLS_DIR[/bold] environment variable, or "
            "pass [bold]--skills-dir[/bold] on this command."
        )
        sys.exit(2)
    return container.skill_loader


def skill_list_command(
    *,
    config_path: Optional[Path],
    console: Console,
    show_patterns: bool = False,
    skills_dir_override: Optional[Path] = None,
) -> None:
    """List every skill discoverable in the configured skills directory.

    Prints a table with name, category, function, breadth, discovery
    pattern count, and runnability status. With ``--show-patterns``,
    also expands each skill to show its Step 1 discovery patterns
    (regex and target extensions) -- useful for verifying that skill
    parsing is picking up what the operator intended.

    Requires ``config.skills.enabled: true``. Exits non-zero with an
    explanatory error otherwise.
    """
    skill_loader = _resolve_skill_loader(
        config_path=config_path,
        skills_dir_override=skills_dir_override,
        console=console,
    )

    skill_dir = skill_loader.directory
    if not skill_dir.exists():
        console.print(
            f"[red]Skills directory does not exist:[/red] {skill_dir}"
        )
        console.print(
            "Create the directory and place skill files as "
            "[bold]<directory>/<name>/SKILL.md[/bold]."
        )
        sys.exit(2)

    skills = skill_loader.load_all()
    if not skills:
        console.print(
            f"[yellow]No skills found in[/yellow] {skill_dir}"
        )
        console.print(
            "Expected either [bold]<dir>/<name>/SKILL.md[/bold] "
            "or [bold]<dir>/<name>.md[/bold]. Files starting with "
            "[bold]_[/bold] (like [bold]_shared/[/bold]) are skipped."
        )
        return

    console.print(
        f"[cyan]Skills directory:[/cyan] {skill_dir}  "
        f"[cyan]({len(skills)} skill{'s' if len(skills) != 1 else ''})[/cyan]"
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Function")
    table.add_column("Patterns", justify="right")
    table.add_column("Runnable")

    for s in skills:
        # A scan skill without Step 1 patterns runs as a per-file sweep
        # (one LLM call per file), which costs differently from "yes".
        if not s.is_runnable_today:
            runnable_marker = "[yellow]deferred[/yellow]"
        elif not s.discovery_patterns:
            runnable_marker = "[yellow]sweep[/yellow]"
        else:
            runnable_marker = "[green]yes[/green]"
        # Text cells: frontmatter values are operator-authored and must
        # not be parsed as Rich markup.
        table.add_row(
            Text(s.name),
            Text(s.metadata.category or "-"),
            Text(s.metadata.function or "scan"),
            str(len(s.discovery_patterns)),
            runnable_marker,
        )
    console.print(table)

    # Explain the "deferred" state so the operator knows why fp-check
    # and triage skills show up but aren't runnable via `skill run`.
    deferred = [s for s in skills if not s.is_runnable_today]
    if deferred:
        console.print(
            "\n[dim]Skills marked [/dim][yellow]deferred[/yellow][dim] "
            "have function=verify or function=triage; they will run in "
            "later pipeline phases, not via `skill run`.[/dim]"
        )
    if any(s.is_runnable_today and not s.discovery_patterns for s in skills):
        console.print(
            "[dim]Skills marked [/dim][yellow]sweep[/yellow][dim] have no "
            "Step 1 grep patterns: `skill run` evaluates each candidate file "
            "(one LLM call per file, capped by --max-files), and `skill "
            "run-all` includes them only with --include-llm-only.[/dim]"
        )

    if show_patterns:
        _print_pattern_details(skills, console)


def _print_pattern_details(skills, console: Console) -> None:
    """Expand each skill's discovery patterns for inspection.

    Format is compact: one line per pattern, showing the description
    (from the # comment above the grep), file extensions, and a
    truncated regex. Full regexes are often long enough to wrap
    unhelpfully in a terminal.

    Descriptions and regexes are printed as Text, never as markup: a
    regex is exactly the kind of string Rich mangles (``\\[`` loses
    its backslash, a ``[/]`` class raises MarkupError).
    """
    for s in skills:
        n = len(s.discovery_patterns)
        console.print(Text.assemble(
            "\n", (s.name, "bold"), " ",
            (f"({n} pattern{'s' if n != 1 else ''})", "dim"),
        ))
        if not s.discovery_patterns:
            console.print("  [dim](no discovery patterns; LLM-only skill)[/dim]")
            continue
        for i, p in enumerate(s.discovery_patterns):
            desc = p.description or "(no description)"
            ext_str = ",".join(p.extensions)
            regex_display = p.regex if len(p.regex) <= 80 else p.regex[:77] + "..."
            console.print(Text(f"  [{i}] {desc}"))
            console.print(Text.assemble("      ", ("ext:", "dim"), " ", ext_str))
            console.print(Text.assemble("      ", ("regex:", "dim"), " ", regex_display))


def skill_lint_command(
    *,
    config_path: Optional[Path],
    console: Console,
    skills_dir_override: Optional[Path] = None,
) -> None:
    """Report problems in skill files without running them.

    Lists, per skill, everything the tolerant loader and grep runner
    would otherwise skip silently: regexes Python can't compile or
    reads differently from grep -E, grep commands the loader ignores,
    broken frontmatter, duplicate names, and skills that can't run
    yet. Exits 1 when any error is found, so it can gate a skills
    repository in CI.
    """
    from ...infrastructure.skills.skill_linter import (
        LINT_ERROR,
        LINT_INFO,
        LINT_WARNING,
        lint_skills,
    )

    skill_loader = _resolve_skill_loader(
        config_path=config_path,
        skills_dir_override=skills_dir_override,
        console=console,
    )
    skill_dir = skill_loader.directory
    if not skill_dir.exists():
        console.print(Text.assemble(
            ("Skills directory does not exist:", "red"), f" {skill_dir}",
        ))
        sys.exit(2)
    skills = skill_loader.load_all()
    if not skills:
        console.print(Text.assemble(("No skills found in", "yellow"), f" {skill_dir}"))
        return

    issues = lint_skills(skills)
    by_skill: dict = {}
    for issue in issues:
        by_skill.setdefault(issue.skill, []).append(issue)

    styles = {LINT_ERROR: "red", LINT_WARNING: "yellow", LINT_INFO: "dim"}
    console.print(Text.assemble(
        (f"Linting {len(skills)} skill{'s' if len(skills) != 1 else ''} in", "cyan"),
        f" {skill_dir}",
    ))
    for name in sorted({s.name for s in skills}):
        console.print(Text(f"\n{name}", style="bold"))
        found = by_skill.get(name)
        if not found:
            console.print(Text("  ok", style="green"))
            continue
        for issue in found:
            console.print(Text.assemble(
                "  ", (f"{issue.level.upper():7}", styles[issue.level]),
                " ", (issue.code, "cyan"), "  ", issue.message,
            ))

    counts = {level: sum(1 for i in issues if i.level == level) for level in styles}
    console.print(Text(
        f"\n{len(skills)} skills: {counts[LINT_ERROR]} error(s), "
        f"{counts[LINT_WARNING]} warning(s), {counts[LINT_INFO]} info"
    ))
    if counts[LINT_ERROR]:
        sys.exit(1)



def skill_run_command(
    *,
    skill_name: str,
    workspace: Path,
    output_format: str,
    output_file: Optional[Path],
    config_path: Optional[Path],
    skills_dir_override: Optional[Path],
    console: Console,
    confirm_taint: bool = False,
    max_files: Optional[int] = None,
) -> None:
    """Execute a single skill against a workspace.

    Loads the skill by name (matching ``SkillDefinition.name``), runs
    the Step 1 grep discovery, invokes the LLM once per file with
    discovery hits, and emits a SecurityReview with the LLM's
    findings converted to SecurityFinding shape. A skill without
    discovery patterns is swept instead: one LLM call per candidate
    file, up to ``max_files`` (default ``config.skills.max_llm_only_files``),
    with the file count printed before any call is made.

    Same output modes as ``skill list`` -- console table or JSON. When
    ``--output-file`` is set, JSON is written to disk; otherwise
    console output goes to stdout.

    With ``confirm_taint``, findings are enriched with call-graph
    reachability before output -- the same enrichment ``falconeye
    review --confirm-taint`` performs.
    """
    import asyncio
    import json as _json
    import sys as _sys

    # Import here to keep the module import-light (skills lives in
    # infrastructure, and the CLI adapter should not pull the whole
    # tree in at import time).
    from ...application.services.skill_runner_service import SkillRunnerService

    skill_loader = _resolve_skill_loader(
        config_path=config_path,
        skills_dir_override=skills_dir_override,
        console=console,
    )

    # -- Load the skill by name --
    all_skills = skill_loader.load_all()
    matches = [s for s in all_skills if s.name == skill_name]
    if not matches:
        available = ", ".join(sorted(s.name for s in all_skills)) or "(none)"
        console.print(Text(f"Skill '{skill_name}' not found.", style="red"))
        console.print(Text.assemble(("Available skills:", "dim"), " ", available))
        _sys.exit(2)
    skill = matches[0]

    # -- Gate on runnability early with an actionable message --
    if not skill.is_runnable_today:
        fn = skill.metadata.function or "unknown"
        console.print(Text(
            f"Skill '{skill_name}' has function={fn}, which is not "
            "executable via skill run.",
            style="yellow",
        ))
        console.print(
            "  verify skills route through fp-check (Phase 3, pending).\n"
            "  triage skills route through the triage stage (Phase 4, pending)."
        )
        _sys.exit(2)

    if not workspace.exists():
        console.print(f"[red]Workspace path does not exist:[/red] {workspace}")
        _sys.exit(2)

    # -- Get the LLM service from the DI container --
    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    llm_service = getattr(container, "llm_service", None)
    if llm_service is None:
        console.print(
            "[red]No LLM service configured.[/red] Check the "
            "[bold]llm[/bold] section of your config."
        )
        _sys.exit(2)

    runner = SkillRunnerService(llm_service=llm_service)
    sweep_exts, sweep_cap = _sweep_settings(container, max_files)

    # -- Run --
    console.print(Text.assemble(
        ("Running skill", "cyan"), " ", (skill.name, "bold"), " against ",
        (str(workspace), "cyan"),
    ))
    if skill.discovery_patterns:
        n = len(skill.discovery_patterns)
        console.print(Text(
            f"{n} discovery pattern{'s' if n != 1 else ''}, "
            f"category={skill.metadata.category or 'unknown'}",
            style="dim",
        ))
    else:
        # A sweep is one LLM call per file: say how many before any
        # call is made, so an oversized workspace can be aborted.
        candidates = runner.sweep_candidates(
            skill=skill, workspace=workspace, extensions=sweep_exts,
        )
        selector = (
            f"targets {', '.join(skill.metadata.targets)}"
            if skill.metadata.targets else "enabled-language source files"
        )
        console.print(Text(
            f"LLM-only skill: evaluating {min(len(candidates), sweep_cap)} of "
            f"{len(candidates)} files ({selector}; one LLM call each, "
            f"--max-files {sweep_cap})",
            style="yellow",
        ))

    try:
        review, summary = asyncio.run(runner.run(
            skill=skill,
            workspace=workspace,
            max_total_hits=_discovery_hit_cap(container),
            sweep_extensions=sweep_exts,
            max_sweep_files=sweep_cap,
        ))
    except ValueError as e:
        # Runner rejects unrunnable skills with ValueError -- we
        # pre-checked above but the runner is defensive too.
        console.print(Text(str(e), style="red"))
        _sys.exit(2)

    # -- Summary line --
    _print_run_summary(console, summary)

    _confirm_findings(
        confirm_taint=confirm_taint,
        container=container,
        review=review,
        workspace=workspace,
        console=console,
    )

    # -- Output --
    if output_format == "json":
        _write_findings_json(review, output_file, console)
    else:
        _print_findings_console(console, review)


def _confirm_findings(
    *,
    confirm_taint: bool,
    container,
    review,
    workspace: Path,
    console: Console,
) -> None:
    """Apply ``--confirm-taint`` reachability enrichment to skill findings.

    Mirrors ``review_command``'s handling and delegates to the same
    ``_run_taint_confirmation`` helper, so both entry points share one
    implementation of the language gating, the non-fatal failure
    handling, and the one-line diagnostic. Confirmation is additive:
    findings are never dropped, whatever their reachability status.
    """
    if not confirm_taint or not review.findings:
        return
    confirmation_service = getattr(container, "review_confirmation_service", None)
    if confirmation_service is None:
        console.print(
            "[yellow]--confirm-taint requested but no taint-enabled "
            "language is configured. Skipping confirmation.[/yellow]"
        )
        return
    # Imported lazily: commands.py pulls in the whole review pipeline,
    # and this module stays import-light.
    from .commands import _run_taint_confirmation
    _run_taint_confirmation(
        review=review,
        repo_path=workspace if workspace.is_dir() else workspace.parent,
        confirmation_service=confirmation_service,
        config=container.config,
        console=console,
    )


def _discovery_hit_cap(container) -> int:
    """config.skills.max_discovery_hits, or the runner's default (200).

    The setting existed (and config.yaml documented it) but nothing
    read it: every run used GrepRunner.DEFAULT_MAX_TOTAL_HITS.
    """
    from ...infrastructure.skills.grep_runner import GrepRunner

    skills = getattr(getattr(container, "config", None), "skills", None)
    return getattr(skills, "max_discovery_hits", None) or GrepRunner.DEFAULT_MAX_TOTAL_HITS


def _sweep_settings(container, max_files: Optional[int]):
    """Return ``(extensions, file_cap)`` for sweeping LLM-only skills.

    Extensions cover the source files of ``config.languages.enabled``
    (every known language if the config doesn't say). The cap is
    ``--max-files`` if given, else ``config.skills.max_llm_only_files``.
    """
    from ...application.services.skill_runner_service import (
        DEFAULT_MAX_SWEEP_FILES,
        sweep_extensions_for,
    )
    from ...domain.services.language_detector import LanguageDetector

    config = getattr(container, "config", None)
    enabled = getattr(getattr(config, "languages", None), "enabled", None)
    if enabled is None:
        enabled = list(LanguageDetector.LANGUAGE_EXTENSIONS)
    if max_files is None:
        max_files = getattr(
            getattr(config, "skills", None), "max_llm_only_files",
            DEFAULT_MAX_SWEEP_FILES,
        )
    return sweep_extensions_for(enabled), max_files


def _print_run_summary(console: Console, summary) -> None:
    """One-line-per-file diagnostic + total, then blank line."""
    if summary.mode == "sweep":
        capped = (
            f", {summary.files_not_evaluated} over the --max-files cap"
            if summary.files_not_evaluated else ""
        )
        done = summary.files_evaluated
        scope = f"sweep: {done} file{'s' if done != 1 else ''} evaluated{capped}"
    else:
        scope = (
            f"{summary.total_discovery_hits} discovery hits, "
            f"{summary.files_evaluated} files evaluated"
        )
    console.print(
        f"\n[bold]Skill run summary[/bold]  "
        f"[dim]({scope}, "
        f"{summary.files_with_llm_errors} with LLM errors, "
        f"{summary.total_findings} findings)[/dim]"
    )
    if summary.files_not_evaluated:
        console.print(
            f"[yellow]{summary.files_not_evaluated} file"
            f"{'s' if summary.files_not_evaluated != 1 else ''} not evaluated "
            "(over the --max-files cap); raise --max-files or point the run "
            "at a subdirectory.[/yellow]"
        )
    if summary.files_with_llm_errors:
        console.print("[yellow]Files with errors:[/yellow]")
        for o in summary.outcomes:
            if o.error:
                console.print(Text.assemble(f"  {o.file_path}: ", (o.error, "red")))
    console.print()


def _print_findings_console(console: Console, review) -> None:
    """Render findings compactly for the console.

    Reuses the ConsoleFormatter so output is consistent with what
    `falconeye review` produces. This keeps the p16 taint-confirm
    reporter enhancements (Reachability line, etc.) available when
    a later patch adds --confirm-taint to `skill run`.
    """
    from ..formatters.console_formatter import ConsoleFormatter
    formatter = ConsoleFormatter(use_color=True, verbose=False)
    # Verbatim: the report carries LLM-written text and code snippets,
    # which Rich's markup parser would mangle (``args[name]`` reads as
    # a style tag and vanishes) or crash on (``a[/b]``).
    console.out(formatter.format_review(review), highlight=False)


def _write_findings_json(review, output_file, console: Console) -> None:
    """Emit findings as SecurityFinding.to_dict() JSON.

    Uses the FLAT to_dict() shape (matching the taint-confirm output
    contract from the loader-shapes patch) so downstream tools can
    consume the output directly with the same shape-tolerant loader.
    """
    import json as _json
    payload = {
        "codebase_path": review.codebase_path,
        "language": review.language,
        "findings": [f.to_dict() for f in review.findings],
    }
    text = _json.dumps(payload, indent=2, default=str)
    if output_file is None:
        # Verbatim: console.print would re-wrap long lines (invalid
        # JSON when piped) and parse LLM text as markup.
        console.out(text, highlight=False)
    else:
        output_file.write_text(text, encoding="utf-8")
        console.print(
            f"[green]Wrote {len(review.findings)} findings to[/green] {output_file}"
        )



def skill_run_all_command(
    *,
    workspace: Path,
    output_format: str,
    output_file: Optional[Path],
    config_path: Optional[Path],
    skills_dir_override: Optional[Path],
    dedupe: bool,
    console: Console,
    confirm_taint: bool = False,
    include_llm_only: bool = False,
    max_files: Optional[int] = None,
) -> None:
    """Execute every runnable skill against a workspace.

    Loads all skills from the configured skills directory (respects
    ``--skills-dir``), filters out non-runnable ones (function=verify /
    triage, and -- unless ``include_llm_only`` -- skills without Step 1
    discovery patterns), invokes each remaining skill via
    ``SkillRunnerService.run_all``, aggregates the findings into a
    single ``SecurityReview``, and optionally deduplicates. Included
    LLM-only skills are swept, up to ``max_files`` files each.

    Skipped skills are surfaced in the summary with the reason so the
    operator can see why (and fix the skill file if the skip was
    unintentional).

    With ``confirm_taint``, reachability enrichment runs after dedup,
    so each merged finding is confirmed once.
    """
    import asyncio
    import json as _json
    import sys as _sys

    from ...application.services.skill_runner_service import (
        BATCH_STATUS_ERROR,
        BATCH_STATUS_RAN,
        BATCH_STATUS_SKIPPED_NO_PATTERNS,
        BATCH_STATUS_SKIPPED_NOT_SCAN,
        SkillRunnerService,
    )

    skill_loader = _resolve_skill_loader(
        config_path=config_path,
        skills_dir_override=skills_dir_override,
        console=console,
    )

    all_skills = skill_loader.load_all()
    if not all_skills:
        console.print(
            f"[yellow]No skills found in[/yellow] {skill_loader.directory}"
        )
        console.print("Nothing to run.")
        return

    if not workspace.exists():
        console.print(f"[red]Workspace path does not exist:[/red] {workspace}")
        _sys.exit(2)

    # Get the LLM service
    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    llm_service = getattr(container, "llm_service", None)
    if llm_service is None:
        console.print(
            "[red]No LLM service configured.[/red] Check the "
            "[bold]llm[/bold] section of your config."
        )
        _sys.exit(2)

    # Preview: what will run, what will skip
    scan_skills = [s for s in all_skills if s.is_runnable_today]
    llm_only = [s for s in scan_skills if not s.discovery_patterns]
    runnable = len(scan_skills) - (0 if include_llm_only else len(llm_only))
    skipped_preview = len(all_skills) - runnable
    console.print(Text.assemble(
        (f"Running {runnable} of {len(all_skills)} skills", "cyan"),
        " against ", (str(workspace), "cyan"),
        (f"  ({skipped_preview} skipped)", "dim") if skipped_preview else "",
    ))
    sweep_exts, sweep_cap = _sweep_settings(container, max_files)
    if llm_only and include_llm_only:
        console.print(Text(
            f"{len(llm_only)} LLM-only skill{'s' if len(llm_only) != 1 else ''} "
            f"swept file by file: up to {sweep_cap} LLM calls each "
            "(--max-files).",
            style="yellow",
        ))
    elif llm_only:
        console.print(Text(
            f"{len(llm_only)} LLM-only skill{'s' if len(llm_only) != 1 else ''} "
            "skipped; pass --include-llm-only to sweep them file by file.",
            style="dim",
        ))

    runner = SkillRunnerService(llm_service=llm_service)
    review, batch = asyncio.run(runner.run_all(
        skills=all_skills,
        workspace=workspace,
        dedupe=dedupe,
        include_llm_only=include_llm_only,
        sweep_extensions=sweep_exts,
        max_sweep_files=sweep_cap,
        max_total_hits=_discovery_hit_cap(container),
    ))

    # Per-skill summary table
    _print_batch_summary(console, batch, dedupe=dedupe)

    _confirm_findings(
        confirm_taint=confirm_taint,
        container=container,
        review=review,
        workspace=workspace,
        console=console,
    )

    # Findings output
    if output_format == "json":
        _write_findings_json(review, output_file, console)
    else:
        _print_findings_console(console, review)


def _print_batch_summary(console: Console, batch, *, dedupe: bool) -> None:
    """One-line-per-skill table plus totals.

    Skills that ran show findings count and error count. Skipped
    skills show the reason so operators know why (bad function tag,
    missing Step 1 patterns, etc.). Errored skills show the exception.
    """
    from ...application.services.skill_runner_service import (
        BATCH_STATUS_ERROR,
        BATCH_STATUS_RAN,
        BATCH_STATUS_SKIPPED_NO_PATTERNS,
        BATCH_STATUS_SKIPPED_NOT_SCAN,
    )

    console.print(f"\n[bold]Batch summary[/bold]")
    for o in batch.outcomes:
        # Names come from skill frontmatter and reasons can carry
        # exception text, so both are Text, never markup.
        name = (o.skill_name, "bold")
        if o.status == BATCH_STATUS_RAN and o.summary is not None:
            n = o.summary.total_findings
            errs = o.summary.files_with_llm_errors
            err_str = (f"  {errs} error{'s' if errs != 1 else ''}", "red") if errs else ""
            sweep_str = ""
            if o.summary.mode == "sweep":
                over = o.summary.files_not_evaluated
                done = o.summary.files_evaluated
                sweep_str = (
                    f"  (sweep: {done} file{'s' if done != 1 else ''}"
                    + (f", {over} over cap" if over else "") + ")",
                    "dim",
                )
            console.print(Text.assemble(
                "  ", ("RAN", "green"), "      ", name, ": ",
                f"{n} finding{'s' if n != 1 else ''}", err_str, sweep_str,
            ))
        elif o.status == BATCH_STATUS_SKIPPED_NOT_SCAN:
            console.print(Text.assemble(
                "  ", ("SKIPPED", "yellow"), "  ", name, ": ", (o.reason, "dim"),
            ))
        elif o.status == BATCH_STATUS_SKIPPED_NO_PATTERNS:
            console.print(Text.assemble(
                "  ", ("SKIPPED", "yellow"), "  ", name, ": ", (o.reason, "dim"),
            ))
        elif o.status == BATCH_STATUS_ERROR:
            console.print(Text.assemble(
                "  ", ("ERROR", "red"), "    ", name, ": ", o.reason,
            ))

    # Totals line
    totals_parts = [
        f"[bold]{batch.skills_ran}[/bold] ran",
        f"[bold]{batch.skills_skipped}[/bold] skipped",
    ]
    if batch.skills_errored:
        totals_parts.append(f"[red]{batch.skills_errored}[/red] errored")
    findings_str = f"[bold]{batch.findings_kept}[/bold] findings"
    if dedupe and batch.duplicates_dropped:
        findings_str += (
            f" [dim](deduped from {batch.total_findings}; "
            f"{batch.duplicates_dropped} duplicate"
            f"{'s' if batch.duplicates_dropped != 1 else ''} dropped)[/dim]"
        )
    console.print(
        f"[dim]Total:[/dim] {', '.join(totals_parts)}, {findings_str}"
    )
    console.print()
