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
