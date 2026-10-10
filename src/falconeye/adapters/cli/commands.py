"""CLI command implementations."""

import asyncio
import logging
import time as _time
from pathlib import Path
from typing import Optional, Callable
import typer
from rich.console import Console
from rich.markup import escape as _esc
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from ...infrastructure.di.container import DIContainer
from ...infrastructure.config.config_loader import ConfigLoader
from ...infrastructure.presentation.error_presenter import ErrorPresenter
from ...infrastructure.logging import FalconEyeLogger
from ...infrastructure.workspace_files import is_excluded
from ...application.commands.index_codebase import IndexCodebaseCommand
from ...application.commands.review_file import ReviewFileCommand
from ..formatters.formatter_factory import FormatterFactory
from ...domain.models.security import SecurityFinding, Severity, FindingConfidence


def _format_finding_brief(finding: SecurityFinding, console: Console, finding_number: int) -> None:
    """
    Show a brief real-time notification when a finding is detected during analysis.
    Full details are displayed after LLM enrichment completes.
    """
    severity_colors = {
        Severity.CRITICAL: "red",
        Severity.HIGH: "bright_red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "blue",
        Severity.INFO: "cyan",
    }
    color = severity_colors.get(finding.severity, "white")
    severity_text = finding.severity.value.upper()
    console.print(
        f"  [dim]#{finding_number}[/dim] "
        f"[{color}][{severity_text}][/{color}] "
        f"[white]{finding.issue}[/white]"
    )


def _format_finding_realtime(finding: SecurityFinding, console: Console, finding_number: int = None) -> None:
    """
    Format and display a security finding in real-time with Rich formatting.
    
    Args:
        finding: SecurityFinding to display
        console: Rich console instance
        finding_number: Optional finding number for display
    """
    # Determine severity color and border style
    severity_colors = {
        Severity.CRITICAL: ("red", "bold red"),
        Severity.HIGH: ("bright_red", "red"),
        Severity.MEDIUM: ("yellow", "yellow"),
        Severity.LOW: ("blue", "blue"),
        Severity.INFO: ("cyan", "cyan"),
    }
    
    border_color, title_color = severity_colors.get(finding.severity, ("white", "white"))
    
    # Create title with severity badge
    title_parts = []
    if finding_number is not None:
        title_parts.append(f"[dim]#{finding_number}[/dim] ")
    
    severity_text = finding.severity.value.upper()
    title_parts.append(f"[{title_color}]{severity_text}[/{title_color}]")
    # _esc() prevents raw finding text (which can contain characters
    # that Rich interprets as markup tags -- e.g. `[/\w]` from a regex
    # pattern in minified JS) from crashing the renderer.
    title_parts.append(f" [bold white]{_esc(finding.issue)}[/bold white]")
    
    title = "".join(title_parts)
    
    # Build finding content
    content_parts = []
    
    # Location information
    if finding.file_path:
        location = f"[cyan]{_esc(finding.file_path)}[/cyan]"
        if finding.line_start:
            if finding.line_end and finding.line_end != finding.line_start:
                location += f" [dim](lines {finding.line_start}-{finding.line_end})[/dim]"
            else:
                location += f" [dim](line {finding.line_start})[/dim]"
        content_parts.append(location)
        content_parts.append("")
    
    # Confidence badge
    confidence_colors = {
        FindingConfidence.HIGH: "green",
        FindingConfidence.MEDIUM: "yellow",
        FindingConfidence.LOW: "orange",
    }
    conf_color = confidence_colors.get(finding.confidence, "white")
    content_parts.append(f"[{conf_color}]Confidence: {finding.confidence.value.upper()}[/{conf_color}]")
    content_parts.append("")
    
    # Reasoning
    if finding.reasoning:
        content_parts.append("[bold]Description:[/bold]")
        content_parts.append(f"  {_esc(finding.reasoning)}")
        content_parts.append("")
    
    # Mitigation
    if finding.mitigation:
        content_parts.append("[bold green]Recommendation:[/bold green]")
        content_parts.append(f"  {_esc(finding.mitigation)}")
        content_parts.append("")

    # Validation warnings (from structured validation pass). These
    # signal issues the automated checks found -- unrecognized CWEs,
    # snippets that don't appear in the file, line numbers out of
    # range, suspicious CVE references. Display prominently so a
    # reviewer knows to double-check before trusting the finding.
    validation_warnings = getattr(finding, "validation_warnings", []) or []
    if validation_warnings:
        content_parts.append("[bold yellow]Validation warnings:[/bold yellow]")
        for w in validation_warnings:
            content_parts.append(f"  [yellow]- {_esc(w)}[/yellow]")
        content_parts.append("")

    # Verifier verdict (from the adversarial two-pass review). Only
    # shown when the verifier ran. false_positive verdicts don't
    # reach here (the verifier drops them). We display verdict +
    # counterargument so the reviewer can see why the verifier agreed
    # or hesitated.
    v_verdict = getattr(finding, "verifier_verdict", None)
    if v_verdict:
        verdict_color = {
            "confirmed": "green",
            "downgrade": "yellow",
            "flag_uncertain": "orange3",
        }.get(v_verdict, "cyan")
        content_parts.append(
            f"[bold {verdict_color}]Verifier: {_esc(v_verdict)}[/bold {verdict_color}]"
        )
        v_counter = getattr(finding, "verifier_counterargument", "") or ""
        if v_counter:
            content_parts.append(
                f"  [dim]Counterargument considered: {_esc(v_counter)}[/dim]"
            )
        content_parts.append("")
    
    # Code snippet (if verbose or if available)
    if finding.code_snippet:
        content_parts.append("[dim]Code snippet:[/dim]")
        # Indent code snippet
        for line in finding.code_snippet.split("\n")[:5]:  # Limit to 5 lines
            content_parts.append(f"  [dim]{_esc(line)}[/dim]")
        if finding.code_snippet.count("\n") > 5:
            content_parts.append("  [dim]...[/dim]")
    
    content = "\n".join(content_parts)
    
    # Create panel with styled border
    panel = Panel(
        content,
        title=title,
        border_style=border_color,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    
    console.print(panel)
    console.print("")  # Spacing between findings


def _configure_logger_verbosity(container, verbose: bool) -> None:
    """
    Apply config.logging and the verbose flag to the shared logger.

    Logs always go to config.logging.file (when set). The console gets
    them only in verbose mode, so normal runs show the progress display
    and summaries, not log lines.

    Args:
        container: DI container with config
        verbose: Whether verbose mode is enabled
    """
    from .logging_setup import configure_cli_logging

    configure_cli_logging(container.config, verbose)


def _estimate_time_based_progress(elapsed_seconds: float) -> int:
    """
    Estimate progress percentage based on elapsed time.

    Uses a logarithmic-like curve: fast initial progress, slowing over time.
    This provides smooth progress feedback when actual progress can't be measured.

    Args:
        elapsed_seconds: Time elapsed since operation started

    Returns:
        Estimated progress percentage (0-95, never 100)
    """
    if elapsed_seconds < 10:
        # First 10 seconds: 0-20%
        return int((elapsed_seconds / 10) * 20)
    elif elapsed_seconds < 30:
        # 10-30 seconds: 20-60%
        return 20 + int(((elapsed_seconds - 10) / 20) * 40)
    elif elapsed_seconds < 60:
        # 30-60 seconds: 60-85%
        return 60 + int(((elapsed_seconds - 30) / 30) * 25)
    else:
        # After 60 seconds: 85-95% (slow increase)
        return 85 + min(10, int((elapsed_seconds - 60) / 10))


def index_command(
    path: Path,
    language: Optional[str],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    exclude: Optional[list[str]],
    project_id: Optional[str],
    force_reindex: bool,
    config_path: Optional[str],
    verbose: bool,
    backend: Optional[str] = None,
    console: Console = None,
):
    """
    Execute index command.

    Args:
        path: Path to codebase
        language: Language name
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        exclude: Exclusion patterns
        project_id: Explicit project ID
        force_reindex: Force re-index all files
        config_path: Config file path
        verbose: Enable verbose output
        backend: LLM backend override
        console: Rich console
    """
    if verbose:
        console.print(Panel.fit(
            "[bold]FalconEYE Indexer[/bold]",
            border_style="blue"
        ))

    # Create DI container
    container = DIContainer.create(config_path, backend_override=backend)

    # Configure logger verbosity
    _configure_logger_verbosity(container, verbose)

    # Use config values if not specified
    if chunk_size is None:
        chunk_size = container.config.chunking.default_size
    if chunk_overlap is None:
        chunk_overlap = container.config.chunking.default_overlap
    # -e adds to the configured defaults rather than replacing them, so
    # `-e "tests/*"` doesn't silently stop excluding node_modules.
    exclude = _effective_exclusions(container, exclude)

    # Create command
    command = IndexCodebaseCommand(
        codebase_path=path,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        excluded_patterns=exclude,
        project_id=project_id,
        force_reindex=force_reindex,
        embedding_batch_size=container.config.llm.embedding_batch_size,
        max_chunk_chars=container.config.chunking.max_chunk_chars,
        max_chunk_size=container.config.chunking.max_chunk_size,
        # Concurrency: how many files to embed + store in parallel. Bounded
        # by a semaphore in the handler so the embedding model isn't
        # swamped. Sourced from config.analysis.batch_size (previously the
        # setting had no effect).
        concurrency=container.config.analysis.batch_size,
    )

    # Discover files first to get total count for progress
    from ...domain.services.language_detector import LanguageDetector
    language_detector = container.language_detector
    
    # Detect language if not specified
    if language is None:
        detected_language = language_detector.detect_language(path)
    else:
        detected_language = language
    
    # Discover files to get total count
    try:
        detected_languages = language_detector.detect_all_languages(path)
    except Exception:
        detected_languages = [detected_language]
    
    files = []
    for lang in detected_languages:
        extensions = language_detector.LANGUAGE_EXTENSIONS.get(lang, [])
        for ext in extensions:
            files.extend(list(path.rglob(f"*{ext}")))
    
    # Filter excluded patterns
    filtered_files = [
        file_path for file_path in files
        if not is_excluded(file_path.relative_to(path), exclude)
    ]

    total_files = len(filtered_files)
    
    # Track progress
    processed_files = [0]
    
    def update_progress(current: int, total: int):
        """Update progress callback."""
        processed_files[0] = current
        if total > 0:
            percentage = int((current / total) * 100)
            return percentage
        return 0

    # Execute with progress
    if verbose:
        # Verbose mode: show spinner and all logs
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing codebase...", total=None)
            try:
                codebase = asyncio.run(container.index_handler.handle(command))
                progress.update(task, description="[green]Indexing complete!")
            except KeyboardInterrupt:
                progress.update(task, description="[yellow]Indexing cancelled")
                error_msg = ErrorPresenter.present(KeyboardInterrupt(), verbose=verbose)
                console.print(f"\n{error_msg}")
                raise SystemExit(1)
            except Exception as e:
                progress.update(task, description="[red]Indexing failed!")
                error_msg = ErrorPresenter.present(e, verbose=verbose)
                console.print(f"\n{error_msg}")
                raise SystemExit(1)
    else:
        # Non-verbose mode: show progress percentage only
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"Indexing codebase...",
                total=100  # Use percentage (0-100)
            )
            
            # Start with 0%
            progress.update(task, completed=0)
            
            # Run indexing in background and update progress periodically
            import threading
            import time
            
            progress_value = [0]  # Use list for mutable reference
            indexing_done = [False]
            indexing_error = [None]
            
            def update_progress_loop():
                """Update progress bar while indexing is running."""
                while not indexing_done[0] and progress_value[0] < 95:
                    time.sleep(0.5)  # Update every 500ms
                    if progress_value[0] < 90:
                        progress_value[0] += 2  # Gradually increase
                    progress.update(task, completed=progress_value[0])
            
            # Start progress updater
            progress_thread = threading.Thread(target=update_progress_loop, daemon=True)
            progress_thread.start()
            
            try:
                codebase = asyncio.run(container.index_handler.handle(command))
                indexing_done[0] = True
                progress.update(task, completed=100, description="[green]Indexing complete!")
            except KeyboardInterrupt:
                indexing_done[0] = True
                progress.update(task, description="[yellow]Indexing cancelled")
                raise SystemExit(1)
            except Exception as e:
                indexing_done[0] = True
                indexing_error[0] = e
                progress.update(task, description="[red]Indexing failed!")
                error_msg = ErrorPresenter.present(e, verbose=verbose)
                console.print(f"\n{error_msg}")
                raise SystemExit(1)

    # Display summary
    if not verbose:
        console.print("")  # Add newline after progress bar
    stats = getattr(container.index_handler, "last_run_stats", None)
    failed = len(stats.files_failed) if stats is not None else 0
    console.print(f"[green]Indexed {codebase.total_files - failed} files[/green]")

    # Show all detected languages
    all_langs = codebase.all_languages
    if len(all_langs) == 1:
        console.print(f"[green]Language: {all_langs[0]}[/green]")
    else:
        langs_str = ", ".join(all_langs)
        console.print(f"[green]Languages: {langs_str}[/green]")

    console.print(f"[green]Total lines: {codebase.total_lines}[/green]")
    if stats is not None:
        _print_index_problems(console, stats)


def _print_index_problems(console: Console, stats) -> None:
    """Warn about files missing from the RAG index, in whole or in part.

    These files are still reviewed (review reads them from disk); what's
    lost is their use as related-code context for other files. Paths and
    errors are printed as Text: both are arbitrary strings.
    """
    shown = 5
    if stats.files_partial:
        console.print(Text(
            f"{len(stats.files_partial)} file(s) indexed partially: "
            f"{stats.chunks_skipped} chunk(s) could not be embedded and are "
            "missing from RAG context. Fix the cause, then re-run with "
            "--force-reindex to retry them.",
            style="yellow",
        ))
        for path, skipped, total, error in stats.files_partial[:shown]:
            console.print(Text(f"  - {path}: {skipped}/{total} chunks skipped ({error})", style="dim"))
    if stats.files_failed:
        console.print(Text(
            f"{len(stats.files_failed)} file(s) could not be indexed; they get "
            "no RAG context and are retried on the next run:",
            style="yellow",
        ))
        for path, error in stats.files_failed[:shown]:
            console.print(Text(f"  - {path}: {error}", style="dim"))
    hidden = max(0, len(stats.files_partial) - shown) + max(0, len(stats.files_failed) - shown)
    if hidden:
        console.print(Text(f"  ... and {hidden} more", style="dim"))


def _effective_exclusions(container, extra) -> list:
    """config.file_discovery.default_exclusions plus any CLI -e patterns."""
    return list(container.config.file_discovery.default_exclusions) + list(extra or [])


def _collect_review_files(path: Path, languages, extensions_by_language, exclude):
    """Files a directory review analyzes, plus how many were excluded.

    One walk over the tree for the union of the languages' extensions
    (the prior implementation ran rglob() once per extension). Sorted,
    so the checkpoint session ID is stable across runs regardless of
    filesystem iteration order. Exclusions use the same matcher as the
    index step.
    """
    wanted_exts = set()
    for lang in languages:
        wanted_exts.update(extensions_by_language.get(lang, []))
    candidates = sorted({
        p for p in path.rglob("*")
        if p.is_file() and p.suffix in wanted_exts
    })
    files = [p for p in candidates if not is_excluded(p.relative_to(path), exclude)]
    return files, len(candidates) - len(files)


def review_command(
    path: Path,
    language: Optional[str],
    validate: bool,
    top_k: Optional[int],
    output_format: Optional[str],
    output_file: Optional[Path],
    severity: Optional[str],
    config_path: Optional[str],
    verbose: bool,
    backend: Optional[str] = None,
    sage: bool = False,
    resume: Optional[bool] = None,
    concurrency: Optional[int] = None,
    confirm_taint: bool = False,
    console: Console = None,
    exclude: Optional[list[str]] = None,
):
    """
    Execute review command.

    Args:
        path: Path to review
        language: Language name
        validate: Enable validation
        top_k: Context count
        output_format: Output format
        output_file: Output file
        severity: Minimum severity
        config_path: Config file path
        verbose: Verbose output
        backend: LLM backend override
        sage: Enable SAGE persistent memory
        resume: Resume from a matching checkpoint if one exists.
                None = auto-prompt on match, True = force resume, False = fresh scan
        concurrency: Max files analyzed in parallel. Defaults to
                     config.analysis.batch_size.
        confirm_taint: When True, run call-graph reachability
                       confirmation on findings before output.
                       Enriches each finding with a taint_confirmation
                       block. Requires taint to be enabled for at
                       least one language in config.
        console: Rich console
        exclude: Extra exclusion patterns for directory reviews, added to
                 config.file_discovery.default_exclusions (the same
                 list the index step uses).
    """
    if verbose:
        console.print(Panel.fit(
            "[bold]FalconEYE Security Review[/bold]",
            border_style="blue"
        ))

    # Create DI container
    container = DIContainer.create(config_path, backend_override=backend, sage_override=sage)

    # Pre-scan SAGE connectivity probe — gives early feedback instead of
    # failing silently mid-scan on the first recall attempt.
    if container.memory_service:
        try:
            sage_ok = asyncio.run(container.memory_service.health_check())
            if sage_ok:
                console.print("[green]SAGE memory service connected[/green]")
            else:
                console.print("[yellow]SAGE memory service unreachable — scanning without persistent memory[/yellow]")
                container.memory_service = None
        except Exception:
            console.print("[yellow]SAGE memory service unreachable — scanning without persistent memory[/yellow]")
            container.memory_service = None

    # Configure logger verbosity
    _configure_logger_verbosity(container, verbose)

    # Use config values if not specified
    if top_k is None:
        top_k = container.config.analysis.top_k_context
    if output_format is None:
        output_format = container.config.output.default_format

    # Detect language if not specified
    if language is None:
        language = container.language_detector.detect_language(path)

    # Check if path is directory or file
    if path.is_dir():
        # Directory - review all files from ALL detected languages

        # Detect all languages in the codebase
        try:
            all_languages = container.language_detector.detect_all_languages(path)
            console.print(f"[cyan]Detected languages: {', '.join(all_languages)}[/cyan]")
        except Exception:
            all_languages = [language]

        # Same exclusion list the index step uses, so `scan` doesn't send
        # files to the LLM that it deliberately left out of the index.
        files, excluded_count = _collect_review_files(
            path,
            all_languages,
            container.language_detector.LANGUAGE_EXTENSIONS,
            _effective_exclusions(container, exclude),
        )
        if excluded_count:
            console.print(Text(
                f"Skipping {excluded_count} file(s) matching exclusion patterns",
                style="dim",
            ))

        if not files:
            console.print(f"[yellow]No source files found in {path}[/yellow]")
            return

        # ---------- Resume / checkpoint setup ----------
        # Build a stable session ID from the inputs that affect results.
        # If a matching checkpoint exists we can resume from it.
        chat_model = getattr(container.config.llm.model, "analysis", "")
        session_id = container.checkpoint_service.compute_session_id(
            codebase_path=path,
            file_paths=files,
            top_k=top_k,
            validate=validate,
            chat_model=str(chat_model),
        )

        existing_ckpt = container.checkpoint_service.find_existing(session_id)
        resume_state = None
        if existing_ckpt is not None:
            done = len(existing_ckpt.completed_files)
            total = existing_ckpt.total_files
            fnd = len(existing_ckpt.findings)
            if resume is True:
                resume_state = existing_ckpt
                console.print(
                    f"[cyan]Resuming previous scan[/cyan] "
                    f"(session {session_id}): {done}/{total} files completed, "
                    f"{fnd} findings restored"
                )
            elif resume is False:
                console.print(
                    f"[yellow]Ignoring existing checkpoint "
                    f"({done}/{total} files) -- starting fresh scan[/yellow]"
                )
            else:
                # Interactive prompt (default when --resume is not set)
                console.print(
                    f"[cyan]Found existing checkpoint[/cyan] "
                    f"for this scan: {done}/{total} files completed, "
                    f"{fnd} findings collected."
                )
                try:
                    answer = typer.prompt(
                        "Resume from checkpoint? [Y/n]",
                        default="Y",
                        show_default=False,
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "y"
                if answer in ("", "y", "yes"):
                    resume_state = existing_ckpt
                    console.print(
                        f"[cyan]Resuming[/cyan] -- {done} files will be skipped, "
                        f"{fnd} prior findings preserved"
                    )
                else:
                    console.print("[yellow]Discarding checkpoint -- fresh scan[/yellow]")

        # Concurrency
        if concurrency is None:
            concurrency = container.config.analysis.batch_size
        concurrency = max(1, int(concurrency))
        if concurrency > 1:
            console.print(f"[dim]Analyzing up to {concurrency} files in parallel[/dim]")

        # Create aggregate review and seed with any restored findings
        from ...domain.models.security import SecurityReview
        aggregate_review = SecurityReview.create(
            codebase_path=str(path),
            language=language,
        )

        # ---------- Incremental report writer ----------
        # Resolve the output path up front so we can rewrite the report
        # after every completed file. The report on disk therefore stays
        # in sync with what's on screen -- users can peek at partial
        # results during a long scan instead of waiting for full
        # completion.
        _incremental_path: Optional[Path] = None
        _incremental_html_path: Optional[Path] = None
        _incremental_formatter = None
        _incremental_html_formatter = None

        if output_file is not None:
            _incremental_path = output_file
            _incremental_formatter = FormatterFactory.create(
                output_format,
                use_color=container.config.output.color,
                verbose=verbose,
            )
        elif output_format == "json" and container.config.output.save_to_file:
            # Match the same auto-save behavior the end-of-scan block
            # does, but pre-resolve the filename now so all incremental
            # writes target one stable path (rather than making one file
            # per file completed).
            from datetime import datetime as _dt
            output_dir = Path(container.config.output.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
            project_name = path.name if path.is_dir() else path.stem
            _incremental_path = output_dir / f"falconeye_{project_name}_{timestamp}.json"
            _incremental_html_path = output_dir / f"falconeye_{project_name}_{timestamp}.html"
            _incremental_formatter = FormatterFactory.create(
                output_format,
                use_color=container.config.output.color,
                verbose=verbose,
            )
            _incremental_html_formatter = FormatterFactory.create("html")
            console.print(
                f"[dim]Report will stream to {_incremental_path.name}[/dim]"
            )

        def _incremental_writer(review) -> None:
            """
            Atomically rewrite the report file with the current aggregate
            review. Called after every completed file so the file on disk
            is always current.

            Atomicity: write to a hidden .tmp sibling, then os.replace()
            it onto the target path. Guarantees the file on disk is
            either the previous complete version or the new complete
            version -- never half-written even if the process is killed
            mid-write.
            """
            if _incremental_path is None or _incremental_formatter is None:
                return
            import os as _os
            import tempfile as _tempfile

            def _atomic_write(target: Path, content: str) -> None:
                fd, tmp = _tempfile.mkstemp(
                    dir=str(target.parent),
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                )
                try:
                    with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                        fh.write(content)
                        fh.flush()
                        _os.fsync(fh.fileno())
                    _os.replace(tmp, target)
                except Exception:
                    try:
                        _os.unlink(tmp)
                    except FileNotFoundError:
                        pass
                    raise

            _atomic_write(_incremental_path, _incremental_formatter.format_review(review))
            if _incremental_html_path is not None and _incremental_html_formatter is not None:
                _atomic_write(
                    _incremental_html_path,
                    _incremental_html_formatter.format_review(review),
                )

        # ---------- Run the concurrent scan ----------
        async def _run_directory_scan():
            await container.checkpoint_service.start(
                session_id=session_id,
                codebase_path=path,
                total_files=len(files),
                resume_from=resume_state,
            )

            # Seed aggregate with restored findings (if resuming).
            if resume_state is not None:
                for f in container.checkpoint_service.restored_findings():
                    aggregate_review.add_finding(f)

            # Files still to analyze this run.
            pending = [
                fp for fp in files
                if not container.checkpoint_service.already_completed(fp)
            ]
            skipped = len(files) - len(pending)

            failed_files_local = []
            sem = asyncio.Semaphore(concurrency)
            handler = container.review_file_handler

            # Track which files are currently in-flight so the progress
            # bar can show *something* useful before the first file
            # completes. Without this, a large concurrency + slow first
            # LLM call = several minutes stuck at "0%" with no feedback.
            in_flight: set = set()
            completed_count = 0

            # Progress bar setup
            if verbose:
                progress_columns = [
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ]
            else:
                progress_columns = [
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ]

            with Progress(
                *progress_columns,
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"Security Review: Analyzing {len(pending)} files"
                    + (f" (skipped {skipped} from checkpoint)" if skipped else "")
                    + "...",
                    total=len(pending) if pending else 1,
                )
                if not pending:
                    progress.update(task, completed=1,
                                    description="[green]All files already analyzed[/green]")

                # Heartbeat coroutine: updates the progress bar text every
                # 2 seconds with which files are currently in-flight and
                # how long the scan has been running. Runs alongside the
                # gather and cancels itself when the gather finishes.
                heartbeat_start = _time.time()

                def _fmt_elapsed(sec: float) -> str:
                    if sec < 60:
                        return f"{int(sec)}s"
                    m, s = divmod(int(sec), 60)
                    return f"{m}m{s:02d}s"

                async def _heartbeat():
                    try:
                        while True:
                            await asyncio.sleep(2)
                            elapsed = _fmt_elapsed(_time.time() - heartbeat_start)
                            if in_flight:
                                # Show up to 2 filenames + count
                                names = sorted(in_flight, key=lambda p: p.name)
                                shown = ", ".join(p.name for p in names[:2])
                                more = f" (+{len(names) - 2} more)" if len(names) > 2 else ""
                                desc = (
                                    f"[{elapsed}] {completed_count}/{len(pending)} done, "
                                    f"{len(in_flight)} in-flight: {shown}{more}"
                                )
                            else:
                                desc = f"[{elapsed}] {completed_count}/{len(pending)} done"
                            progress.update(task, description=desc)
                    except asyncio.CancelledError:
                        return

                async def _analyze_one(file_path):
                    nonlocal completed_count
                    async with sem:
                        in_flight.add(file_path)
                        try:
                            file_language = container.language_detector.detect_language(file_path)
                            file_system_prompt = container.get_system_prompt(file_language)
                            cmd = ReviewFileCommand(
                                file_path=file_path,
                                language=file_language,
                                system_prompt=file_system_prompt,
                                validate_findings=validate,
                                top_k_context=top_k,
                                # Streaming callbacks disabled in concurrent mode:
                                # interleaved output from parallel files is worse
                                # than useless. Use single-file mode or
                                # --concurrency 1 with -v if you want streaming.
                                stream_callback=None,
                                finding_callback=None,
                            )
                            review = await handler.handle(cmd)
                            await container.checkpoint_service.mark_completed(
                                file_path, review.findings
                            )
                            for finding in review.findings:
                                aggregate_review.add_finding(finding)

                            # Incremental report write. Called once per
                            # file so the report on disk always reflects
                            # current progress -- lets you peek at partial
                            # results during a long scan.
                            if _incremental_writer is not None:
                                try:
                                    _incremental_writer(aggregate_review)
                                except Exception as _we:
                                    # Report write errors are non-fatal.
                                    console.print(
                                        f"[yellow]Warning: could not update report file: {_esc(str(_we))}[/yellow]"
                                    )

                            # Render results. Wrapped in its own try/except
                            # so that a rendering crash (e.g. Rich markup
                            # collision from minified JS regex) does NOT
                            # get counted as an analysis failure -- the
                            # analysis and checkpoint mark_completed both
                            # already succeeded above.
                            if review.findings:
                                try:
                                    progress.stop()
                                    console.print(
                                        f"\n[bold cyan]Findings in "
                                        f"[white]{_esc(file_path.name)}[/white]:[/bold cyan]"
                                    )
                                    for idx, finding in enumerate(review.findings, start=1):
                                        _format_finding_realtime(finding, console, finding_number=idx)
                                    console.print("")
                                except Exception as _re:
                                    console.print(
                                        f"\n[yellow]Note: could not render "
                                        f"findings for {_esc(file_path.name)} "
                                        f"(display error, {len(review.findings)} "
                                        f"finding(s) still saved to report). "
                                        f"Cause: {type(_re).__name__}[/yellow]"
                                    )
                                finally:
                                    progress.start()
                            return None
                        except Exception as e:
                            error_type = type(e).__name__
                            error_message = str(e) or "Unknown error"
                            await container.checkpoint_service.mark_failed(
                                file_path, f"{error_type}: {error_message[:200]}"
                            )
                            failed_files_local.append(
                                (file_path, f"{error_type}: {error_message[:100]}")
                            )
                            console.print(
                                f"\n[yellow]Warning: Failed to analyze {_esc(file_path.name)}[/yellow]"
                            )
                            console.print(
                                f"[dim]Error: {error_type}: {_esc(error_message[:200])}[/dim]"
                            )
                            return None
                        finally:
                            in_flight.discard(file_path)
                            completed_count += 1
                            progress.advance(task)

                # Fan out with heartbeat running alongside.
                heartbeat_task = asyncio.create_task(_heartbeat())
                try:
                    await asyncio.gather(*(_analyze_one(fp) for fp in pending))
                finally:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                progress.update(
                    task,
                    description="[green]Security Review complete!",
                    completed=len(pending) if pending else 1,
                )

            # Successful full completion -> remove checkpoint.
            await container.checkpoint_service.finish(remove=True)
            return failed_files_local

        try:
            failed_files = asyncio.run(_run_directory_scan())
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Analysis interrupted.[/yellow] "
                "Progress saved. Re-run the same command (add [cyan]--resume[/cyan] "
                "to skip the prompt) to continue from where it stopped."
            )
            raise SystemExit(130)

        # Display summary of failed files if any
        if failed_files:
            console.print("")
            console.print(f"[yellow]{len(failed_files)} file(s) failed analysis:[/yellow]")
            for failed_path, error in failed_files[:5]:
                console.print(f"  [dim]• {failed_path.name}: {error}[/dim]")
            if len(failed_files) > 5:
                console.print(f"  [dim]... and {len(failed_files) - 5} more[/dim]")
            console.print(
                f"[dim]Successfully analyzed: "
                f"{len(files) - len(failed_files)}/{len(files)} files[/dim]"
            )

        review = aggregate_review
    else:
        # Single file - get system prompt for the detected language
        system_prompt = container.get_system_prompt(language)
        
        # Create streaming callback to show LLM thought process
        stream_buffer = []
        
        def stream_callback(token: str):
            """Callback to display LLM streaming tokens."""
            stream_buffer.append(token)
            # Display in real-time (only in verbose mode)
            if verbose:
                console.print(token, end="", style="dim italic")
        
        command = ReviewFileCommand(
            file_path=path,
            language=language,
            system_prompt=system_prompt,
            validate_findings=validate,
            top_k_context=top_k,
            stream_callback=stream_callback if verbose else None,
        )

        if verbose:
            # Verbose mode: show spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Security Review: Analyzing code...", total=None)

                try:
                    # Show progress updates during analysis
                    async def handle_with_progress_single(cmd):
                        """Wrapper to show progress during single file review."""
                        import time as time_module  # Import inside function to avoid closure issues
                        handler = container.review_file_handler
                        handler_task = asyncio.create_task(handler.handle(cmd))
                        
                        # Show periodic progress updates while waiting
                        status_messages = [
                            "[dim]Assembling context...[/dim]",
                            "[dim]Analyzing with AI (this may take a minute)...[/dim]",
                            "[dim]Processing results...[/dim]",
                        ]
                        status_index = 0
                        last_update = time_module.time()
                        
                        # In verbose mode, show a header for LLM output
                        console.print("\n[bold cyan]AI Analysis (Streaming):[/bold cyan]")
                        console.print("[dim]" + "─" * 60 + "[/dim]")
                        
                        while not handler_task.done():
                            await asyncio.sleep(3)  # Update every 3 seconds
                            current_time = time_module.time()
                            
                            # Update status message periodically
                            if current_time - last_update >= 3:
                                if status_index < len(status_messages):
                                    progress.update(task, description=status_messages[status_index])
                                    status_index += 1
                                else:
                                    # Keep showing "Analyzing with AI" message
                                    progress.update(task, description=status_messages[1])
                                last_update = current_time
                        
                        return await handler_task
                    
                    review = asyncio.run(handle_with_progress_single(command))
                    
                    # Add newline after streaming
                    if stream_buffer:
                        console.print("\n")
                    
                    # Display LLM-enriched findings
                    if review.findings:
                        console.print(f"\n[bold cyan]Security Findings:[/bold cyan]")
                        console.print("")

                        for idx, finding in enumerate(review.findings, start=1):
                            _format_finding_realtime(finding, console, finding_number=idx)

                    progress.update(task, description="[green]Analysis complete!")

                except KeyboardInterrupt:
                    progress.update(task, description="[yellow]Analysis cancelled")
                    error_msg = ErrorPresenter.present(KeyboardInterrupt(), verbose=verbose)
                    console.print(f"\n{error_msg}")
                    raise SystemExit(1)

                except Exception as e:
                    progress.update(task, description="[red]Analysis failed!")
                    error_type = type(e).__name__
                    error_message = str(e) if str(e) else "Unknown error"
                    
                    console.print(f"\n[red]Analysis failed for {path.name}[/red]")
                    console.print(f"[dim]Error: {error_type}: {error_message[:200]}[/dim]")
                    
                    error_msg = ErrorPresenter.present(e, verbose=verbose)
                    console.print(f"\n{error_msg}")
                    raise SystemExit(1)
        else:
            # Non-verbose mode: show progress bar with percentage
            import threading
            import time
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    "Security Review: Analyzing code...",
                    total=100  # Use percentage (0-100)
                )
                
                # Start with 0%
                progress.update(task, completed=0)
                
                # Track progress
                review_progress = [0]
                review_done = [False]
                review_error = [None]
                review_start_time = time.time()  # Track start time for progress estimation

                def update_review_progress_loop():
                    """Update progress bar while review is running (time-based)."""
                    import time as time_module  # Import inside function to avoid closure issues
                    while not review_done[0] and review_progress[0] < 95:
                        time_module.sleep(0.5)  # Update every 500ms
                        elapsed_time = time_module.time() - review_start_time

                        # Use helper function for consistent progress estimation
                        progress_pct = _estimate_time_based_progress(elapsed_time)

                        # Estimate time remaining
                        if elapsed_time > 5:  # After 5 seconds, start estimating
                            # Rough estimate: if we're at X%, remaining is (100-X)/X * elapsed
                            if progress_pct > 0:
                                estimated_total = elapsed_time / (progress_pct / 100)
                                estimated_remaining = estimated_total - elapsed_time
                                if estimated_remaining > 0:
                                    time_remaining_str = f"{int(estimated_remaining // 60)}m {int(estimated_remaining % 60)}s" if estimated_remaining > 60 else f"{int(estimated_remaining)}s"
                                    description = f"Analyzing code... (est. {time_remaining_str} remaining)"
                                else:
                                    description = "Analyzing code... (almost done)"
                            else:
                                description = "Analyzing code..."
                        else:
                            description = "Analyzing code..."

                        review_progress[0] = progress_pct
                        progress.update(task, completed=progress_pct, description=description)
                
                # Start progress updater
                progress_thread = threading.Thread(target=update_review_progress_loop, daemon=True)
                progress_thread.start()
                
                try:
                    # Show progress updates during analysis
                    async def handle_with_progress_single(cmd):
                        """Wrapper to show progress during single file review."""
                        import time as time_module  # Import inside function to avoid closure issues
                        handler = container.review_file_handler
                        handler_task = asyncio.create_task(handler.handle(cmd))
                        
                        # Show periodic progress updates while waiting
                        status_messages = [
                            "[dim]Assembling context...[/dim]",
                            "[dim]Analyzing with AI (this may take a minute)...[/dim]",
                            "[dim]Processing results...[/dim]",
                        ]
                        status_index = 0
                        last_update = time_module.time()
                        
                        # In verbose mode, show a header for LLM output
                        if verbose:
                            console.print("\n[bold cyan]AI Analysis (Streaming):[/bold cyan]")
                            console.print("[dim]" + "─" * 60 + "[/dim]")
                        
                        while not handler_task.done():
                            await asyncio.sleep(3)  # Update every 3 seconds
                            current_time = time_module.time()
                            
                            # Update status message periodically
                            if current_time - last_update >= 3:
                                if status_index < len(status_messages):
                                    progress.update(task, completed=min(90, 30 + status_index * 20), description=status_messages[status_index])
                                    status_index += 1
                                else:
                                    # Keep showing "Analyzing with AI" message and gradually increase progress
                                    current_progress = min(90, review_progress[0] + 5)
                                    review_progress[0] = current_progress
                                    progress.update(task, completed=current_progress, description=status_messages[1])
                                last_update = current_time
                        
                        return await handler_task
                    
                    review = asyncio.run(handle_with_progress_single(command))
                    review_done[0] = True
                    
                    # Display findings immediately if any are found
                    if review.findings:
                        console.print(f"\n[bold cyan]Security Findings:[/bold cyan]")
                        console.print("")

                        for idx, finding in enumerate(review.findings, start=1):
                            _format_finding_realtime(finding, console, finding_number=idx)

                    progress.update(
                        task,
                        completed=100,
                        description="[green]Security Review complete![/green]"
                    )
                except KeyboardInterrupt:
                    review_done[0] = True
                    progress.update(task, description="[yellow]Analysis cancelled")
                    error_msg = ErrorPresenter.present(KeyboardInterrupt(), verbose=verbose)
                    console.print(f"\n{error_msg}")
                    raise SystemExit(1)
                except Exception as e:
                    review_done[0] = True
                    review_error[0] = e
                    progress.update(task, description="[red]Analysis failed!")
                    error_msg = ErrorPresenter.present(e, verbose=verbose)
                    console.print(f"\n{error_msg}")
                    raise SystemExit(1)

    # Optionally run reachability confirmation on the findings.
    # Enriches each finding's taint_confirmation field in place on
    # a rebuilt review object; safe because SecurityReview.findings
    # is a mutable list and dataclasses.replace produces new frozen
    # findings. Skipped silently if the service isn't available
    # (no taint-enabled language configured) so --confirm-taint on
    # a project without taint config produces a clear warning
    # instead of a hard error.
    if confirm_taint and review.findings:
        confirmation_service = getattr(container, "review_confirmation_service", None)
        if confirmation_service is None:
            console.print(
                "[yellow]--confirm-taint requested but no taint-enabled "
                "language is configured. Skipping confirmation.[/yellow]"
            )
        else:
            _run_taint_confirmation(
                review=review,
                repo_path=path if path.is_dir() else path.parent,
                confirmation_service=confirmation_service,
                config=container.config,
                console=console,
            )

    # Format output
    formatter = FormatterFactory.create(
        output_format,
        use_color=container.config.output.color,
        verbose=verbose
    )

    output = formatter.format_review(review)

    # Determine whether the incremental writer already streamed the
    # report during the scan. When it did, skip the redundant
    # end-of-scan write (the incremental path is already current) and
    # only print the confirmation message.
    _streamed_path = None
    if path.is_dir():
        # These names come from the directory-review branch; if the scan
        # didn't run (empty file list, single-file mode), they aren't set.
        _streamed_path = locals().get("_incremental_path")

    # Display or save
    if output_file:
        if _streamed_path is not None and _streamed_path == output_file:
            # Already written incrementally with the current findings.
            console.print(f"\n[green]Results saved to {output_file}[/green] [dim](streamed during scan)[/dim]")
        else:
            output_file.write_text(output)
            console.print(f"\n[green]Results saved to {output_file}[/green]")
    elif output_format == "json" and container.config.output.save_to_file:
        if _streamed_path is not None:
            # Incremental writer already saved this at a stable timestamped path.
            console.print(f"\n[green]Results saved to {_streamed_path}[/green] [dim](streamed during scan)[/dim]")
            _streamed_html = locals().get("_incremental_html_path")
            if _streamed_html is not None:
                console.print(f"[green]HTML report saved to {_streamed_html}[/green] [dim](streamed during scan)[/dim]")
        else:
            # Auto-save JSON to default location (single-file mode path)
            from datetime import datetime
            output_dir = Path(container.config.output.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = path.name if path.is_dir() else path.stem
            auto_file = output_dir / f"falconeye_{project_name}_{timestamp}.json"

            auto_file.write_text(output)
            console.print(f"\n[green]Results saved to {auto_file}[/green]")

            # Also generate HTML report
            html_formatter = FormatterFactory.create("html")
            html_output = html_formatter.format_review(review)
            html_file = output_dir / f"falconeye_{project_name}_{timestamp}.html"
            html_file.write_text(html_output)
            console.print(f"[green]HTML report saved to {html_file}[/green]")
    else:
        # Show final summary - add separator if findings were already shown during scan
        if review.findings and path.is_dir():
            console.print("")  # Extra spacing
            console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
            console.print("[bold cyan]Final Summary[/bold cyan]")
            console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print("")
        # Verbatim: the report carries LLM-written text and code
        # snippets. console.print would parse them as Rich markup
        # (``args[name]`` vanishes, ``a[/b]`` raises MarkupError) and
        # re-wrap JSON/SARIF lines.
        console.out(output, highlight=False)


def scan_command(
    path: Path,
    language: Optional[str],
    validate: bool,
    output_format: Optional[str],
    output_file: Optional[Path],
    project_id: Optional[str],
    force_reindex: bool,
    config_path: Optional[str],
    verbose: bool,
    backend: Optional[str] = None,
    sage: bool = False,
    resume: Optional[bool] = None,
    concurrency: Optional[int] = None,
    console: Console = None,
    exclude: Optional[list[str]] = None,
):
    """
    Execute scan command (index + review).

    Args:
        path: Path to scan
        language: Language name
        validate: Enable validation
        output_format: Output format
        output_file: Output file
        project_id: Explicit project ID
        force_reindex: Force re-index all files
        config_path: Config file path
        verbose: Verbose output
        backend: LLM backend override
        sage: Enable SAGE persistent memory
        resume: Resume review from checkpoint if one exists
        concurrency: Max files analyzed in parallel
        console: Rich console
        exclude: Extra exclusion patterns, applied to both steps on top
                 of config.file_discovery.default_exclusions
    """
    console.print(Panel.fit(
        "[bold]FalconEYE Full Scan[/bold]",
        border_style="blue"
    ))

    # Run index first
    console.print("\n[bold]Step 1: Indexing...[/bold]")
    index_command(
        path=path,
        language=language,
        chunk_size=None,
        chunk_overlap=None,
        exclude=exclude,
        project_id=project_id,
        force_reindex=force_reindex,
        config_path=config_path,
        verbose=verbose,
        backend=backend,
        console=console,
    )

    # Then review
    console.print("\n[bold]Step 2: Security Review...[/bold]")
    review_command(
        path=path,
        language=language,
        validate=validate,
        top_k=None,
        output_format=output_format,
        output_file=output_file,
        severity=None,
        config_path=config_path,
        verbose=verbose,
        backend=backend,
        sage=sage,
        resume=resume,
        concurrency=concurrency,
        console=console,
        exclude=exclude,
    )


def feedback_command(
    finding_id: str,
    valid: bool,
    severity: Optional[str],
    reason: str,
    config_path: Optional[str],
    sage_url: Optional[str],
    console: Console,
):
    """
    Execute feedback command.

    Records user feedback on a security finding in SAGE persistent memory.

    Args:
        finding_id: UUID of the finding
        valid: Whether the finding is a true positive
        severity: Optional severity correction (critical/high/medium/low)
        reason: Reason for feedback
        config_path: Config file path
        sage_url: Optional SAGE base URL override
        console: Rich console
    """
    # Build reason string with optional severity correction
    reason_parts = []
    if severity:
        reason_parts.append(f"Severity correction: -> {severity}.")
    if reason:
        reason_parts.append(reason)
    full_reason = " ".join(reason_parts)

    verdict = "TRUE POSITIVE" if valid else "FALSE POSITIVE"

    # Create DI container with SAGE enabled
    try:
        container = DIContainer.create(config_path, sage_override=True)
    except Exception as e:
        console.print(f"[red]Error initializing FalconEYE: {e}[/red]")
        raise SystemExit(1)

    # Override SAGE URL if provided
    if sage_url and container.memory_service:
        container.memory_service.reconfigure(sage_url)

    if not container.memory_service:
        console.print("[red]SAGE memory service is not available.[/red]")
        console.print("[dim]Ensure SAGE is running and reachable. Use --sage-url to specify a custom URL.[/dim]")
        raise SystemExit(1)

    # Record the feedback
    try:
        asyncio.run(container.memory_service.record_feedback(
            finding_id=finding_id,
            is_valid=valid,
            reason=full_reason,
        ))
        console.print(f"[green]Feedback recorded for finding {finding_id}: {verdict}[/green]")
        if severity:
            console.print(f"[cyan]Severity correction: -> {severity}[/cyan]")
        if reason:
            console.print(f"[dim]Reason: {reason}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to record feedback: {e}[/red]")
        raise SystemExit(1)


def info_command(config_path: Optional[str], console: Console):
    """
    Execute info command.

    Args:
        config_path: Config file path
        console: Rich console
    """
    console.print(Panel.fit(
        "[bold]FalconEYE System Information[/bold]",
        border_style="blue"
    ))

    try:
        # Create DI container
        container = DIContainer.create(config_path)

        # Version info
        console.print("\n[bold]Version:[/bold]")
        console.print("  FalconEYE: 2.0.0")
        console.print("  Analysis: AI-powered (ZERO pattern matching)")

        # LLM info
        console.print("\n[bold]LLM Configuration:[/bold]")
        console.print(f"  Provider: {container.config.llm.provider}")
        console.print(f"  Analysis Model: {container.config.llm.model.analysis}")
        console.print(f"  Embedding Model: {container.config.llm.model.embedding}")
        console.print(f"  Base URL: {container.config.llm.base_url}")

        # Check LLM health
        try:
            is_healthy = asyncio.run(container.llm_service.health_check())
            if is_healthy:
                console.print("  Status: [green]Connected[/green]")
            else:
                console.print("  Status: [red]Not available[/red]")
        except Exception:
            console.print("  Status: [red]Connection failed[/red]")

        # MLX info
        console.print("\n[bold]MLX Backend (Apple Silicon):[/bold]")
        try:
            from ...infrastructure.llm_providers import is_apple_silicon, is_mlx_available
            console.print(f"  Apple Silicon: {'Yes' if is_apple_silicon() else 'No'}")
            console.print(f"  MLX Installed: {'Yes' if is_mlx_available() else 'No'}")
            if is_apple_silicon() and is_mlx_available():
                console.print("  Status: [green]Available[/green] (use --backend mlx)")
            elif is_apple_silicon():
                console.print("  Status: [yellow]Not installed[/yellow] (pip install falconeye[mlx])")
            else:
                console.print("  Status: [dim]Not applicable (requires Apple Silicon)[/dim]")
        except Exception:
            console.print("  Status: [dim]Could not determine[/dim]")

        # Language support
        console.print("\n[bold]Supported Languages:[/bold]")
        languages = container.plugin_registry.get_supported_languages()
        console.print(f"  {', '.join(languages)}")

        # Storage info
        console.print("\n[bold]Storage:[/bold]")
        console.print(f"  Vector Store: {container.config.vector_store.persist_directory}")
        console.print(f"  Metadata: {container.config.metadata.persist_directory}")

        # Configuration info
        console.print("\n[bold]Configuration:[/bold]")
        config_info = ConfigLoader.get_config_info()
        if config_info["existing_configs"]:
            console.print("  Active configs:")
            for cfg in config_info["existing_configs"]:
                console.print(f"    - {cfg}")
        else:
            console.print("  Using default configuration")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise


def config_command(
    init: bool,
    path: Optional[str],
    show: bool,
    console: Console,
):
    """
    Execute config command.

    Args:
        init: Create default config
        path: Config file path
        show: Show current config
        console: Rich console
    """
    console.print(Panel.fit(
        "[bold]FalconEYE Configuration[/bold]",
        border_style="blue"
    ))

    if init:
        # Create default configuration
        try:
            config_path = ConfigLoader.create_default_config(path)
            console.print(f"\n[green]Configuration file created: {config_path}[/green]")
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise

    elif show:
        # Show current configuration
        try:
            config = ConfigLoader.load(path)
            yaml_str = config.to_yaml()
            console.print("\n[bold]Current Configuration:[/bold]")
            console.print(yaml_str)
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            raise

    else:
        # Show config info
        config_info = ConfigLoader.get_config_info()

        console.print("\n[bold]Configuration Files:[/bold]")
        if config_info["existing_configs"]:
            for cfg in config_info["existing_configs"]:
                console.print(f"  [green]{cfg}[/green]")
        else:
            console.print("  No configuration files found")

        console.print("\n[bold]Environment Overrides:[/bold]")
        if config_info["env_overrides"]:
            for env_var in config_info["env_overrides"]:
                console.print(f"  {env_var}")
        else:
            console.print("  None")

        console.print("\n[bold]Default Locations:[/bold]")
        for default_path in config_info["default_paths"]:
            console.print(f"  {default_path}")

def _run_taint_confirmation(
    *,
    review,
    repo_path: Path,
    confirmation_service,
    config,
    console: Console,
) -> None:
    """Enrich a SecurityReview's findings with taint reachability.

    Called from ``review_command`` when ``--confirm-taint`` is set.
    Mutates ``review.findings`` in place with new SecurityFinding
    instances that carry ``taint_confirmation`` populated.

    Failures are non-fatal: any exception during confirmation is
    logged to the console and the review proceeds without
    enrichment. Rationale -- the review's primary output is the
    findings themselves; confirmation is additive signal. A broken
    call-graph build (e.g. tree-sitter crash on an odd source file)
    shouldn't prevent the operator from seeing what the LLM found.
    """
    # Which taint-enabled languages are eligible for confirmation?
    taint_languages = [
        lang
        for lang, settings in config.languages.per_language.items()
        if settings.taint.enabled and lang in config.languages.enabled
    ]
    if not taint_languages:
        console.print(
            "[yellow]No taint-enabled languages in config; "
            "skipping --confirm-taint.[/yellow]"
        )
        return

    try:
        confirmed, summary = confirmation_service.confirm(
            findings=review.findings,
            repo_path=repo_path,
            languages=taint_languages,
        )
    except Exception as e:
        # Text, not markup: the exception message is arbitrary.
        console.print(Text(
            f"Taint confirmation failed ({type(e).__name__}: {e}). "
            "Continuing without confirmation enrichment.",
            style="yellow",
        ))
        return

    # Replace findings in place. SecurityReview.findings is a list;
    # we clear and extend to keep any identity-based references to the
    # review object valid.
    review.findings.clear()
    review.findings.extend(confirmed)

    # One-line diagnostic so the operator can see the enrichment ran.
    console.print(
        f"[dim]Taint confirmation:[/dim] "
        f"[green]{summary.reachable} reachable[/green], "
        f"[yellow]{summary.not_confirmed} not_confirmed[/yellow], "
        f"{summary.no_enclosing_method} no_method, "
        f"{summary.skipped} skipped "
        f"[dim]({summary.entry_points_detected} entry points, "
        f"{summary.call_graph_nodes} nodes)[/dim]"
    )
