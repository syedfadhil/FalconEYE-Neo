"""
`falconeye skill ...` CLI commands (Phase 1a: list only).

This module implements the plumbing invoked by the thin wrappers in
main.py. Following the pattern established by the other commands_*.py
modules: main.py owns the typer surface, this module owns the logic.

Phase 1a exposes ``list`` only. ``run`` (execute a single skill via
LLM) and ``run-all`` (batch execution) land in Phase 1b.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ...infrastructure.di.container import DIContainer


def skill_list_command(
    *,
    config_path: Optional[Path],
    console: Console,
    show_patterns: bool = False,
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
    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    if container.skill_loader is None:
        console.print("[red]Skills are not enabled in the config.[/red]")
        console.print(
            "Set [bold]config.skills.enabled: true[/bold] and configure "
            "[bold]config.skills.directory[/bold] to point at your skills folder."
        )
        sys.exit(2)

    skill_dir = container.skill_loader.directory
    if not skill_dir.exists():
        console.print(
            f"[red]Skills directory does not exist:[/red] {skill_dir}"
        )
        console.print(
            "Create the directory and place skill files as "
            "[bold]<directory>/<name>/SKILL.md[/bold]."
        )
        sys.exit(2)

    skills = container.skill_loader.load_all()
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
        runnable_marker = (
            "[green]yes[/green]"
            if s.is_runnable_today
            else "[yellow]deferred[/yellow]"
        )
        table.add_row(
            s.name,
            s.metadata.category or "-",
            s.metadata.function or "scan",
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

    if show_patterns:
        _print_pattern_details(skills, console)


def _print_pattern_details(skills, console: Console) -> None:
    """Expand each skill's discovery patterns for inspection.

    Format is compact: one line per pattern, showing the description
    (from the # comment above the grep), file extensions, and a
    truncated regex. Full regexes are often long enough to wrap
    unhelpfully in a terminal.
    """
    for s in skills:
        console.print(
            f"\n[bold]{s.name}[/bold] "
            f"[dim]({len(s.discovery_patterns)} pattern"
            f"{'s' if len(s.discovery_patterns) != 1 else ''})[/dim]"
        )
        if not s.discovery_patterns:
            console.print("  [dim](no discovery patterns; LLM-only skill)[/dim]")
            continue
        for i, p in enumerate(s.discovery_patterns):
            desc = p.description or "(no description)"
            ext_str = ",".join(p.extensions)
            regex_display = p.regex if len(p.regex) <= 80 else p.regex[:77] + "..."
            console.print(f"  [{i}] {desc}")
            console.print(f"      [dim]ext:[/dim] {ext_str}")
            console.print(f"      [dim]regex:[/dim] {regex_display}")



def skill_run_command(
    *,
    skill_name: str,
    workspace: Path,
    output_format: str,
    output_file: Optional[Path],
    config_path: Optional[Path],
    skills_dir_override: Optional[Path],
    console: Console,
) -> None:
    """Execute a single skill against a workspace.

    Loads the skill by name (matching ``SkillDefinition.name``), runs
    the Step 1 grep discovery, invokes the LLM once per file with
    discovery hits, and emits a SecurityReview with the LLM's
    findings converted to SecurityFinding shape.

    Same output modes as ``skill list`` -- console table or JSON. When
    ``--output-file`` is set, JSON is written to disk; otherwise
    console output goes to stdout.
    """
    import asyncio
    import json as _json
    import sys as _sys

    # Import here to keep the module import-light (skills lives in
    # infrastructure, and the CLI adapter should not pull the whole
    # tree in at import time).
    from ...application.services.skill_runner_service import SkillRunnerService
    from ...infrastructure.skills.skill_loader import SkillLoader

    # -- Resolve the skill loader (config or override) --
    if skills_dir_override is not None:
        skill_loader = SkillLoader(directory=skills_dir_override.expanduser())
    else:
        container = DIContainer.create(
            config_path=str(config_path) if config_path else None,
        )
        if container.skill_loader is None:
            console.print("[red]Skills are not enabled in the config.[/red]")
            console.print(
                "Set [bold]config.skills.enabled: true[/bold] and configure "
                "[bold]config.skills.directory[/bold], set the "
                "[bold]FALCONEYE_SKILLS_DIR[/bold] environment variable, or "
                "pass [bold]--skills-dir[/bold] on this command."
            )
            _sys.exit(2)
        skill_loader = container.skill_loader

    # -- Load the skill by name --
    all_skills = skill_loader.load_all()
    matches = [s for s in all_skills if s.name == skill_name]
    if not matches:
        available = ", ".join(sorted(s.name for s in all_skills)) or "(none)"
        console.print(f"[red]Skill '{skill_name}' not found.[/red]")
        console.print(f"[dim]Available skills:[/dim] {available}")
        _sys.exit(2)
    skill = matches[0]

    # -- Gate on runnability early with an actionable message --
    if not skill.is_runnable_today:
        fn = skill.metadata.function or "unknown"
        console.print(
            f"[yellow]Skill '{skill_name}' has function=[bold]{fn}[/bold], "
            "which is not executable via [bold]skill run[/bold].[/yellow]"
        )
        console.print(
            "  verify skills route through fp-check (Phase 3, pending).\n"
            "  triage skills route through the triage stage (Phase 4, pending)."
        )
        _sys.exit(2)

    if not skill.discovery_patterns:
        console.print(
            f"[yellow]Skill '{skill_name}' has no Step 1 discovery patterns.[/yellow]"
        )
        console.print(
            "Whole-workspace LLM-only execution is deferred to Phase 1c. "
            "Add at least one grep command inside a bash code block under "
            "the Step 1 heading."
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

    # -- Run --
    console.print(
        f"[cyan]Running skill[/cyan] [bold]{skill.name}[/bold] "
        f"against [cyan]{workspace}[/cyan]"
    )
    console.print(
        f"[dim]{len(skill.discovery_patterns)} discovery pattern"
        f"{'s' if len(skill.discovery_patterns) != 1 else ''}"
        f", category={skill.metadata.category or 'unknown'}[/dim]"
    )

    try:
        review, summary = asyncio.run(runner.run(skill=skill, workspace=workspace))
    except ValueError as e:
        # Runner rejects unrunnable skills with ValueError -- we
        # pre-checked above but the runner is defensive too.
        console.print(f"[red]{e}[/red]")
        _sys.exit(2)

    # -- Summary line --
    _print_run_summary(console, summary)

    # -- Output --
    if output_format == "json":
        _write_findings_json(review, output_file, console)
    else:
        _print_findings_console(console, review)


def _print_run_summary(console: Console, summary) -> None:
    """One-line-per-file diagnostic + total, then blank line."""
    console.print(
        f"\n[bold]Skill run summary[/bold]  "
        f"[dim]({summary.total_discovery_hits} discovery hits, "
        f"{summary.files_evaluated} files evaluated, "
        f"{summary.files_with_llm_errors} with LLM errors, "
        f"{summary.total_findings} findings)[/dim]"
    )
    if summary.files_with_llm_errors:
        console.print("[yellow]Files with errors:[/yellow]")
        for o in summary.outcomes:
            if o.error:
                console.print(f"  {o.file_path}: [red]{o.error}[/red]")
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
    console.print(formatter.format_review(review))


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
        console.print(text)
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
) -> None:
    """Execute every runnable skill against a workspace.

    Loads all skills from the configured skills directory (respects
    ``--skills-dir``), filters out non-runnable ones (function=verify /
    triage, or missing Step 1 discovery patterns), invokes each
    remaining skill via ``SkillRunnerService.run_all``, aggregates the
    findings into a single ``SecurityReview``, and optionally
    deduplicates.

    Skipped skills are surfaced in the summary with the reason so the
    operator can see why (and fix the skill file if the skip was
    unintentional).
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
    from ...infrastructure.skills.skill_loader import SkillLoader

    # Resolve skill loader (config gating vs. override)
    if skills_dir_override is not None:
        skill_loader = SkillLoader(directory=skills_dir_override.expanduser())
    else:
        container = DIContainer.create(
            config_path=str(config_path) if config_path else None,
        )
        if container.skill_loader is None:
            console.print("[red]Skills are not enabled in the config.[/red]")
            console.print(
                "Set [bold]config.skills.enabled: true[/bold] and configure "
                "[bold]config.skills.directory[/bold], set the "
                "[bold]FALCONEYE_SKILLS_DIR[/bold] environment variable, or "
                "pass [bold]--skills-dir[/bold] on this command."
            )
            _sys.exit(2)
        skill_loader = container.skill_loader

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
    runnable = [s for s in all_skills if s.is_runnable_today and s.discovery_patterns]
    skipped_preview = len(all_skills) - len(runnable)
    console.print(
        f"[cyan]Running {len(runnable)} of {len(all_skills)} skills[/cyan] "
        f"against [cyan]{workspace}[/cyan]"
        + (f"  [dim]({skipped_preview} skipped)[/dim]" if skipped_preview else "")
    )

    runner = SkillRunnerService(llm_service=llm_service)
    review, batch = asyncio.run(
        runner.run_all(skills=all_skills, workspace=workspace, dedupe=dedupe)
    )

    # Per-skill summary table
    _print_batch_summary(console, batch, dedupe=dedupe)

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
        if o.status == BATCH_STATUS_RAN and o.summary is not None:
            n = o.summary.total_findings
            errs = o.summary.files_with_llm_errors
            err_str = f"  [red]{errs} error{'s' if errs != 1 else ''}[/red]" if errs else ""
            console.print(
                f"  [green]RAN[/green]      "
                f"[bold]{o.skill_name}[/bold]: "
                f"{n} finding{'s' if n != 1 else ''}"
                f"{err_str}"
            )
        elif o.status == BATCH_STATUS_SKIPPED_NOT_SCAN:
            console.print(
                f"  [yellow]SKIPPED[/yellow]  "
                f"[bold]{o.skill_name}[/bold]: [dim]{o.reason}[/dim]"
            )
        elif o.status == BATCH_STATUS_SKIPPED_NO_PATTERNS:
            console.print(
                f"  [yellow]SKIPPED[/yellow]  "
                f"[bold]{o.skill_name}[/bold]: [dim]{o.reason}[/dim]"
            )
        elif o.status == BATCH_STATUS_ERROR:
            console.print(
                f"  [red]ERROR[/red]    "
                f"[bold]{o.skill_name}[/bold]: {o.reason}"
            )

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
