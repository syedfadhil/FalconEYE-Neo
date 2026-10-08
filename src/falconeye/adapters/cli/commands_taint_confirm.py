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

    Two JSON shapes are in the wild:

    * The FLAT shape produced by ``SecurityFinding.to_dict()`` -- what
      ``falconeye taint-confirm --output json`` emits, and what a
      caller building findings dicts by hand would naturally write.
      Fields are top-level: ``confidence``, ``file_path``,
      ``line_start``, ``verifier_verdict``, etc.

    * The WRAPPED shape produced by the JSON reporter used with
      ``falconeye review --output json``. Confidence is wrapped in
      ``{"value": ..., "level": ...}``; location is wrapped in
      ``{"file_path": ..., "line_start": ..., "line_end": ...}``;
      verifier fields are wrapped in ``{"verdict": ..., "reasoning":
      ..., "counterargument": ...}``.

    This loader accepts either shape, extracting fields from whichever
    form the input uses. Missing optional fields default sanely so
    older JSON files still load. An incoming ``taint_confirmation``
    is dropped -- confirmation will be re-run and a fresh value
    attached.
    """
    confidence = _confidence_from_json(d.get("confidence"))
    severity = _severity_from_json(d.get("severity"))
    file_path, line_start, line_end = _location_from_json(d)
    verifier_verdict, verifier_reasoning, verifier_counter = _verifier_from_json(d)

    return SecurityFinding(
        id=UUID(d["id"]) if d.get("id") else UUID(int=0),
        issue=d.get("issue", ""),
        reasoning=d.get("reasoning", ""),
        mitigation=d.get("mitigation", ""),
        severity=severity,
        confidence=confidence,
        file_path=file_path,
        code_snippet=d.get("code_snippet", ""),
        line_start=line_start,
        line_end=line_end,
        cwe_id=d.get("cwe_id"),
        tags=list(d.get("tags", []) or []),
        evidence_lines=list(d.get("evidence_lines", []) or []),
        assumed_context=d.get("assumed_context", "") or "",
        why_not_false_positive=d.get("why_not_false_positive", "") or "",
        reasoning_trace=d.get("reasoning_trace", "") or "",
        validation_warnings=list(d.get("validation_warnings", []) or []),
        verifier_verdict=verifier_verdict,
        verifier_reasoning=verifier_reasoning,
        verifier_counterargument=verifier_counter,
    )


def _confidence_from_json(value) -> FindingConfidence:
    """Accept either a bare string or a ``{"value": ...}`` dict.

    The JSON reporter emits confidence as ``{"value": "high", "level":
    "high"}``. Both keys hold the same information; ``value`` matches
    the FindingConfidence enum value directly.
    """
    if value is None:
        return FindingConfidence("medium")
    if isinstance(value, str):
        return FindingConfidence(value)
    if isinstance(value, dict):
        # Prefer "value" (enum value); fall back to "level" (enum name lowercased).
        raw = value.get("value") or value.get("level") or "medium"
        return FindingConfidence(raw)
    # Fallback: try str() and see if it parses.
    return FindingConfidence(str(value))


def _severity_from_json(value) -> Severity:
    """Accept either a bare string or a wrapped ``{"value": ...}`` dict.

    The JSON reporter currently emits severity as a bare string, but
    accept the wrapped shape for symmetry with confidence -- costs
    nothing and future-proofs against a similar wrap being added.
    """
    if value is None:
        return Severity("medium")
    if isinstance(value, str):
        return Severity(value)
    if isinstance(value, dict):
        raw = value.get("value") or value.get("level") or "medium"
        return Severity(raw)
    return Severity(str(value))


def _location_from_json(d: dict):
    """Return (file_path, line_start, line_end) from either shape.

    The JSON reporter wraps location fields in a ``location`` object;
    the flat shape has them at the top level. This routine prefers
    the wrapped form when present (JSON reporter output is the more
    common source in practice) and falls back to top-level fields.
    """
    loc = d.get("location")
    if isinstance(loc, dict):
        return (
            loc.get("file_path", "") or "",
            loc.get("line_start"),
            loc.get("line_end"),
        )
    return (
        d.get("file_path", "") or "",
        d.get("line_start"),
        d.get("line_end"),
    )


def _verifier_from_json(d: dict):
    """Return (verdict, reasoning, counterargument) from either shape.

    Flat: verifier_verdict / verifier_reasoning / verifier_counterargument
    Wrapped: verifier.verdict / verifier.reasoning / verifier.counterargument
    """
    v = d.get("verifier")
    if isinstance(v, dict):
        return (
            v.get("verdict"),
            v.get("reasoning", "") or "",
            v.get("counterargument", "") or "",
        )
    return (
        d.get("verifier_verdict"),
        d.get("verifier_reasoning", "") or "",
        d.get("verifier_counterargument", "") or "",
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
