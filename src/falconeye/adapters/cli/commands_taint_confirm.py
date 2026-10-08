"""
``falconeye taint-confirm`` CLI command.

Standalone confirmation stage: take a JSON file of :class:`SecurityFinding`
records (produced by a prior ``falconeye review --output json``), a
workspace path, and enrich each finding with reachability confirmation
against the workspace's call graph.

Two output modes:

* Console (default): human-readable per-finding summary + summary line.
* JSON (``--output json --output-file confirmed.json``): the same
  records the input contained, plus ``taint_confirmation`` per finding.

The confirmation itself is done by :class:`ReviewConfirmationService`.
This module owns argument parsing, input loading, and output rendering
only.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from rich.console import Console
from rich.table import Table

from ...domain.models.confirmation import (
    CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
    CONFIRMATION_STATUS_NO_ENTRY_POINTS,
    CONFIRMATION_STATUS_NOT_CONFIRMED,
    CONFIRMATION_STATUS_REACHABLE,
    CONFIRMATION_STATUS_SKIPPED,
)
from ...domain.models.security import (
    FindingConfidence,
    SecurityFinding,
    Severity,
)
from ...infrastructure.di.container import DIContainer


def taint_confirm_command(
    *,
    path: Path,
    findings_file: Path,
    output_format: str,
    output_file: Optional[Path],
    languages: Optional[List[str]],
    config_path: Optional[Path],
    console: Console,
) -> None:
    """Entry point for ``falconeye taint-confirm``.

    Handles the full flow: load container, load findings JSON, invoke
    the confirmation service, render output. Exits non-zero on
    unrecoverable errors (bad JSON, workspace path missing, etc.);
    findings that individually can't be confirmed are reported with
    a status, not treated as errors.
    """
    if not path.exists():
        console.print(f"[red]Workspace path does not exist:[/red] {path}")
        sys.exit(2)

    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    service = container.review_confirmation_service
    if service is None:
        console.print(
            "[red]Review confirmation is not available.[/red] "
            "This usually means no taint adapter is registered for any "
            "language. Enable taint in the config for at least one "
            "supported language (currently: csharp)."
        )
        sys.exit(2)

    # ---------- Load input findings ----------
    try:
        findings = _load_findings(findings_file)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Could not load findings from {findings_file}:[/red] {e}")
        sys.exit(2)

    if not findings:
        console.print(
            f"[yellow]No findings loaded from {findings_file}.[/yellow] "
            "Nothing to confirm."
        )
        return

    console.print(
        f"[cyan]Loaded {len(findings)} finding"
        f"{'s' if len(findings) != 1 else ''}[/cyan] from {findings_file}"
    )

    # Default language list: whichever languages the confirmation
    # service supports. The service ignores unsupported entries.
    langs = languages or ["csharp"]

    # ---------- Run confirmation ----------
    confirmed, summary = asyncio.get_event_loop().run_until_complete(
        _run_confirmation(service, findings, path, langs)
    ) if _has_running_loop() else _sync_run_confirmation(
        service, findings, path, langs
    )

    # ---------- Render summary ----------
    _print_summary(console, summary)

    # ---------- Render per-finding output ----------
    if output_format == "json":
        _write_json(confirmed, output_file, console)
    else:
        _print_console(console, confirmed)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _sync_run_confirmation(service, findings, path, langs):
    """Bridge to the async service without requiring an event loop."""
    coro = _run_confirmation(service, findings, path, langs)
    return asyncio.run(coro)


async def _run_confirmation(service, findings, path, langs):
    # The confirm() method isn't currently async but wrapping in a
    # coroutine keeps the door open for future async work (e.g. LLM
    # calls during confirmation, if we add them).
    return service.confirm(
        findings=findings,
        repo_path=Path(path),
        languages=langs,
    )


def _load_findings(findings_file: Path) -> List[SecurityFinding]:
    """Parse a findings JSON file.

    Accepts two shapes:
      * A bare list of finding dicts
      * A dict with a top-level ``findings`` array (SecurityReview
        format, produced by ``falconeye review --output json``)
    """
    raw = json.loads(findings_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "findings" in raw:
        records = raw["findings"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(
            "Findings JSON must be a list of finding records or an object "
            "with a top-level 'findings' array."
        )
    return [_finding_from_dict(r) for r in records]


def _finding_from_dict(d: dict) -> SecurityFinding:
    """Rebuild a SecurityFinding from its dict form.

    Tolerates missing optional fields (older JSON files). Ignores an
    incoming ``taint_confirmation`` -- confirmation will be re-run and
    a new value attached.
    """
    return SecurityFinding(
        id=UUID(d["id"]) if "id" in d and d["id"] else UUID(int=0),
        issue=d.get("issue", ""),
        reasoning=d.get("reasoning", ""),
        mitigation=d.get("mitigation", ""),
        severity=Severity(d.get("severity", "medium")),
        confidence=FindingConfidence(d.get("confidence", "medium")),
        file_path=d.get("file_path", ""),
        code_snippet=d.get("code_snippet", ""),
        line_start=d.get("line_start"),
        line_end=d.get("line_end"),
        cwe_id=d.get("cwe_id"),
        tags=list(d.get("tags", [])),
        evidence_lines=list(d.get("evidence_lines", [])),
        assumed_context=d.get("assumed_context", ""),
        why_not_false_positive=d.get("why_not_false_positive", ""),
        reasoning_trace=d.get("reasoning_trace", ""),
        validation_warnings=list(d.get("validation_warnings", [])),
        verifier_verdict=d.get("verifier_verdict"),
        verifier_reasoning=d.get("verifier_reasoning", ""),
        verifier_counterargument=d.get("verifier_counterargument", ""),
    )


def _print_summary(console: Console, summary) -> None:
    console.print(
        f"\n[bold]Confirmation summary[/bold]  "
        f"[dim]({summary.entry_points_detected} entry points, "
        f"{summary.call_graph_nodes} nodes, "
        f"{summary.call_graph_edges} edges)[/dim]"
    )
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    table.add_column("Meaning")
    table.add_row(
        f"[green]{CONFIRMATION_STATUS_REACHABLE}[/green]",
        str(summary.reachable),
        "Reachable from at least one entry point (highest confidence)",
    )
    table.add_row(
        f"[yellow]{CONFIRMATION_STATUS_NOT_CONFIRMED}[/yellow]",
        str(summary.not_confirmed),
        "Enclosing method found but not reachable (dead code? DI?)",
    )
    table.add_row(
        f"[yellow]{CONFIRMATION_STATUS_NO_ENCLOSING_METHOD}[/yellow]",
        str(summary.no_enclosing_method),
        "Finding line not inside any parsed method (top-level? parse fail?)",
    )
    table.add_row(
        f"[yellow]{CONFIRMATION_STATUS_NO_ENTRY_POINTS}[/yellow]",
        str(summary.no_entry_points),
        "No entry points detected in workspace",
    )
    table.add_row(
        f"[dim]{CONFIRMATION_STATUS_SKIPPED}[/dim]",
        str(summary.skipped),
        "Language not supported for confirmation",
    )
    console.print(table)
    console.print(
        "\n[dim]Note: 'not_confirmed' does NOT mean 'false positive'. "
        "Findings retain their original confidence; reachability "
        "confirmation is additive signal that boosts confidence for "
        "provably-reachable code.[/dim]\n"
    )


def _print_console(
    console: Console,
    findings: List[SecurityFinding],
) -> None:
    """Render per-finding lines to the console.

    Compact one-line-per-finding format:
        [status] file:line -- issue (severity)
        -> chain: A -> B -> C   (only for reachable)
    """
    for f in findings:
        conf = f.taint_confirmation
        status = conf.status if conf else "no_confirmation"
        color = _status_color(status)
        console.print(
            f"[{color}][{status}][/{color}] "
            f"{f.file_path}:{f.line_start or '?'} -- "
            f"{f.issue} [dim]({f.severity.value})[/dim]"
        )
        if conf and conf.is_reachable and conf.shortest_chain:
            arrow = " [dim]->[/dim] "
            chain = arrow.join(conf.shortest_chain)
            console.print(f"    [dim]chain:[/dim] {chain}")


def _status_color(status: str) -> str:
    if status == CONFIRMATION_STATUS_REACHABLE:
        return "green"
    if status in (
        CONFIRMATION_STATUS_NOT_CONFIRMED,
        CONFIRMATION_STATUS_NO_ENCLOSING_METHOD,
        CONFIRMATION_STATUS_NO_ENTRY_POINTS,
    ):
        return "yellow"
    return "dim"


def _write_json(
    findings: List[SecurityFinding],
    output_file: Optional[Path],
    console: Console,
) -> None:
    payload = [f.to_dict() for f in findings]
    text = json.dumps(payload, indent=2, default=str)
    if output_file is None:
        console.print(text)
    else:
        output_file.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote {len(findings)} findings to[/green] {output_file}")
