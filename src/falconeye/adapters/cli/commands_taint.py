"""
`falconeye taint` CLI command implementation.

Registered via a thin wrapper in main.py (matching the pattern used
by the other subcommands: main.py owns the typer.Option/Argument
surface, this module owns the actual logic).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..formatters.formatter_factory import FormatterFactory
from ...infrastructure.di.container import DIContainer


def taint_command(
    *,
    path: Path,
    output_format: Optional[str],
    output_file: Optional[Path],
    max_hops: Optional[int],
    verify: bool,
    config_path: Optional[Path],
    console: Console,
) -> None:
    """Scan a codebase for taint-based vulnerabilities.

    Walks the repo for source files in enabled languages, builds a
    call graph, identifies candidate source-to-sink paths, verifies
    each with an LLM, and reports confirmed findings.

    Requires taint analysis to be enabled for at least one language
    in the config (``config.languages.per_language.<lang>.taint.enabled``).
    If nothing is enabled, this exits non-zero with an explanatory
    error rather than silently producing zero findings.
    """
    container = DIContainer.create(
        config_path=str(config_path) if config_path else None,
    )
    if container.taint_scan_service is None:
        console.print(
            "[red]Taint analysis is not enabled in the config.[/red]"
        )
        console.print(
            "Set [bold]config.languages.per_language.<lang>.taint.enabled: true[/bold] "
            "for at least one language."
        )
        sys.exit(2)

    # Collect the opted-in languages from config so the scan service
    # only walks files with matching extensions.
    enabled_languages = [
        lang
        for lang, settings in container.config.languages.per_language.items()
        if settings.taint.enabled and lang in container.config.languages.enabled
    ]
    if not enabled_languages:
        console.print("[red]No taint-enabled languages in config.[/red]")
        sys.exit(2)

    # Determine effective max_hops. CLI --max-hops overrides config.
    # When multiple languages have different configured values, use
    # the max (permissive default; the operator can prune afterward).
    if max_hops is not None:
        effective_max_hops = max_hops
    else:
        per_lang = [
            container.config.languages.per_language[lang].taint.max_hops
            for lang in enabled_languages
        ]
        effective_max_hops = max(per_lang) if per_lang else 5

    console.print(
        f"[cyan]Taint scanning[/cyan] {path} "
        f"(languages={','.join(enabled_languages)}, "
        f"max_hops={effective_max_hops}, "
        f"verify={verify})"
    )

    result = asyncio.run(container.taint_scan_service.scan(
        repo_path=path,
        languages=enabled_languages,
        max_hops=effective_max_hops,
        verify=verify,
    ))

    console.print(
        f"[green]Scan complete.[/green] "
        f"Files analyzed: {result.review.files_analyzed}. "
        f"Graph: {len(result.graph)} nodes, {result.graph.edge_count()} edges. "
        f"Candidates: {len(result.candidate_paths)}. "
        f"Findings: {len(result.review.findings)}."
    )

    # Verdict breakdown -- explains findings=0 when candidates>0.
    # Prints only in verify mode (--no-verify has no verdicts to show).
    if verify and result.candidate_paths:
        counts = result.verdict_counts
        tail = ""
        if counts.get("call_failed", 0):
            tail = f"  (call_failed={counts['call_failed']})"
        console.print(
            f"  Verdicts:  "
            f"confirmed={counts['confirmed']}  "
            f"uncertain={counts['uncertain']}  "
            f"false_positive={counts['false_positive']}"
            + tail
        )
        if counts["false_positive"] > 0 and len(result.review.findings) == 0:
            console.print(
                "  [yellow]All candidates rejected by verifier.[/yellow] "
                "Run with [bold]--no-verify[/bold] to see raw candidates and inspect why."
            )

    # Format and output using the same formatters the review command uses.
    fmt_name = output_format or container.config.output.default_format
    formatter = FormatterFactory.create(fmt_name)
    rendered = formatter.format_review(result.review)

    if output_file is not None:
        output_file.write_text(rendered, encoding="utf-8")
        console.print(f"Results written to [cyan]{output_file}[/cyan]")
    else:
        # For console formatter, route through the rich console so ANSI
        # colours render. Other formatters (json/sarif/html) are raw text.
        if fmt_name == "console":
            console.print(rendered)
        else:
            sys.stdout.write(rendered)
            sys.stdout.write("\n")
