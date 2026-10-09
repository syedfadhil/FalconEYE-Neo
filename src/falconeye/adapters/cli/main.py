"""Main CLI application entry point."""

import sys
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console

from .banner import print_banner
from .commands import (
    index_command,
    review_command,
    scan_command,
    feedback_command,
    info_command,
    config_command,
)
from .commands_upgrade import upgrade_command
from .commands_taint import taint_command
from .commands_skill import (
    skill_lint_command,
    skill_list_command,
    skill_run_all_command,
    skill_run_command,
)
from .commands_taint_confirm import taint_confirm_command
from .commands_projects import (
    projects_list_command,
    projects_info_command,
    projects_delete_command,
    projects_cleanup_command,
)

# Create Typer app
app = typer.Typer(
    name="falconeye",
    help="FalconEYE v2.0 - AI-Powered Security Code Review",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Rich console for output
console = Console()


@app.command(name="index")
def index(
    path: Path = typer.Argument(
        ...,
        help="Path to codebase to index",
        exists=True,
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Programming language (auto-detected if not specified)",
    ),
    chunk_size: Optional[int] = typer.Option(
        None,
        "--chunk-size",
        help="Lines per chunk",
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None,
        "--chunk-overlap",
        help="Lines of overlap between chunks",
    ),
    exclude: Optional[list[str]] = typer.Option(
        None,
        "--exclude", "-e",
        help="Exclusion patterns (can specify multiple)",
    ),
    project_id: Optional[str] = typer.Option(
        None,
        "--project-id",
        help="Explicit project ID (auto-detected if not specified)",
    ),
    force_reindex: bool = typer.Option(
        False,
        "--force-reindex",
        help="Force re-index all files (skip smart re-indexing)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output with technical details and full logs",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend", "-b",
        help="LLM backend: ollama (default) or mlx (Apple Silicon)",
    ),
):
    """
    Index a codebase for analysis.

    Indexes source code files, generates embeddings, and extracts metadata.
    This is required before running security reviews.

    Usage:
        falconeye index <path>        # Normal mode: Progress bar only
        falconeye index <path> -v     # Verbose mode: Full indexing logs
    """
    index_command(
        path=path,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        exclude=exclude,
        project_id=project_id,
        force_reindex=force_reindex,
        config_path=config,
        verbose=verbose,
        backend=backend,
        console=console,
    )


@app.command(name="review")
def review(
    path: Path = typer.Argument(
        ...,
        help="Path to file or directory to review",
        exists=True,
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Programming language (auto-detected if not specified)",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Enable AI validation to reduce false positives",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        help="Number of similar chunks for context",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output format (console, json, sarif, html)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Save results to file",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        help="Minimum severity to report (critical, high, medium, low)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output with full logs",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend", "-b",
        help="LLM backend: ollama (default) or mlx (Apple Silicon)",
    ),
    sage: bool = typer.Option(
        False,
        "--sage",
        help="Enable SAGE persistent memory for historical context",
    ),
    resume: Optional[bool] = typer.Option(
        None,
        "--resume/--no-resume",
        help=(
            "Resume from checkpoint if one exists for this scan. "
            "Default: prompt interactively. --resume skips the prompt; "
            "--no-resume ignores any existing checkpoint and starts fresh."
        ),
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        help=(
            "Max files analyzed in parallel. Defaults to config.analysis.batch_size. "
            "Set to 1 for legacy sequential behavior (also required for interactive "
            "per-file LLM streaming under -v)."
        ),
        min=1,
    ),
    confirm_taint: bool = typer.Option(
        False,
        "--confirm-taint",
        help=(
            "Run call-graph reachability confirmation on findings before output. "
            "Enriches each finding with a taint_confirmation block indicating whether "
            "its enclosing method is reachable from any detected entry point (e.g. "
            "HTTP controllers, middleware). Requires taint to be enabled for at least "
            "one language in config."
        ),
    ),
):
    """
    Review a file or directory for security vulnerabilities.

    Performs AI-powered security analysis on the specified code.
    The codebase should be indexed first for best results (RAG context).

    Usage:
        falconeye review <path>              # Normal mode: Progress bar and results
        falconeye review <path> -v           # Verbose mode: Full logs and LLM streaming
        falconeye review <path> --resume     # Resume interrupted scan without prompting
        falconeye review <path> --concurrency 8   # Analyze 8 files in parallel
    """
    review_command(
        path=path,
        language=language,
        validate=validate,
        top_k=top_k,
        output_format=output_format,
        output_file=output_file,
        severity=severity,
        config_path=config,
        verbose=verbose,
        backend=backend,
        sage=sage,
        resume=resume,
        concurrency=concurrency,
        confirm_taint=confirm_taint,
        console=console,
    )


@app.command(name="scan")
def scan(
    path: Path = typer.Argument(
        ...,
        help="Path to codebase to scan",
        exists=True,
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Programming language (auto-detected if not specified)",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Enable AI validation to reduce false positives",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output format (console, json, sarif, html)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Save results to file",
    ),
    project_id: Optional[str] = typer.Option(
        None,
        "--project-id",
        help="Explicit project ID (auto-detected if not specified)",
    ),
    force_reindex: bool = typer.Option(
        False,
        "--force-reindex",
        help="Force re-index all files (skip smart re-indexing)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output with full logs",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend", "-b",
        help="LLM backend: ollama (default) or mlx (Apple Silicon)",
    ),
    sage: bool = typer.Option(
        False,
        "--sage",
        help="Enable SAGE persistent memory for historical context",
    ),
    resume: Optional[bool] = typer.Option(
        None,
        "--resume/--no-resume",
        help="Resume review from checkpoint if one exists",
    ),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        help="Max files analyzed in parallel (defaults to config.analysis.batch_size)",
        min=1,
    ),
):
    """
    Index and review in one command.

    Convenience command that indexes the codebase and then performs
    a security review.

    Usage Examples:
        falconeye scan <path>              # Normal mode: Shows progress bar and final results only
        falconeye scan <path> -v           # Verbose mode: Shows full logs, LLM streaming, and detailed progress
        falconeye scan <path> --verbose    # Verbose mode: Same as -v (alternative syntax)

    Output Modes:
        Normal mode (-v not used):
            - Progress bar showing file count and current file being analyzed
            - Findings displayed in real-time as they're detected
            - Final summary with all findings
            - Clean output without detailed logs

        Verbose mode (-v or --verbose):
            - All indexing logs (file processing, chunking, embedding generation)
            - LLM thought process streaming (AI analysis in real-time)
            - All security analysis logs
            - Detailed progress information
            - Full error details if analysis fails
    """
    scan_command(
        path=path,
        language=language,
        validate=validate,
        output_format=output_format,
        output_file=output_file,
        project_id=project_id,
        force_reindex=force_reindex,
        config_path=config,
        verbose=verbose,
        backend=backend,
        sage=sage,
        resume=resume,
        concurrency=concurrency,
        console=console,
    )


@app.command(name="feedback")
def feedback(
    finding_id: str = typer.Argument(..., help="Finding UUID from scan results"),
    valid: bool = typer.Option(
        True,
        "--valid/--invalid",
        help="Mark as true positive (--valid) or false positive (--invalid)",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        help="Correct severity (critical/high/medium/low)",
    ),
    reason: str = typer.Option(
        "",
        "--reason", "-r",
        help="Reason for feedback",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    sage_url: Optional[str] = typer.Option(
        None,
        "--sage-url",
        help="SAGE API base URL override",
    ),
):
    """
    Submit feedback on a security finding.

    Mark a finding as a true or false positive, optionally correcting severity.
    Feedback is stored in SAGE persistent memory to improve future scans.

    Usage:
        falconeye feedback <finding-id> --invalid -r "Not exploitable"
        falconeye feedback <finding-id> --valid --severity low -r "Low impact"
    """
    feedback_command(
        finding_id=finding_id,
        valid=valid,
        severity=severity,
        reason=reason,
        config_path=config,
        sage_url=sage_url,
        console=console,
    )


@app.command(name="info")
def info(
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """
    Show system information.

    Displays FalconEYE version, LLM status, and configuration.
    """
    info_command(config_path=config, console=console)


@app.command(name="config")
def config_cmd(
    init: bool = typer.Option(
        False,
        "--init",
        help="Create default configuration file",
    ),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Configuration file path",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration",
    ),
):
    """
    Manage configuration.

    Create, view, or validate configuration files.
    """
    config_command(
        init=init,
        path=path,
        show=show,
        console=console,
    )


@app.command(name="upgrade")
def upgrade():
    """
    Upgrade FalconEYE to the latest version.

    Pulls the latest changes from the remote git repository (origin/main)
    and reinstalls Python package dependencies from pyproject.toml.

    Steps performed:
        1. Detects your FalconEYE git clone location automatically
        2. Runs: git fetch + git pull origin main
        3. Reinstalls dependencies: pip install -e . (or pip install .)
        4. Reports the version before and after

    Requirements:
        FalconEYE must be installed from a git clone (pip install -e .).
        Installing from a package archive is not supported — reinstall
        from the repository instead: https://github.com/FalconEYE-ai/FalconEYE

    Usage:
        falconeye upgrade
    """
    upgrade_command(console=console)


@app.command(name="taint")
def taint(
    path: Path = typer.Argument(
        ...,
        help="Path to codebase to scan for taint flows",
        exists=True,
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output format (console, json, sarif, html). Defaults to config.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Save results to file",
    ),
    max_hops: Optional[int] = typer.Option(
        None,
        "--max-hops",
        min=0,
        max=20,
        help="Override max_hops from config (0-20; 0 = intra-method only)",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="LLM-verify candidate paths. --no-verify emits raw candidates tagged 'taint:unverified' -- useful for debugging.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
):
    """
    Scan a codebase for taint-based vulnerabilities.

    Walks the repo for source files in taint-enabled languages, builds a
    call graph, identifies candidate source-to-sink paths, verifies each
    candidate with an LLM, and reports confirmed findings.

    Requires taint analysis to be enabled in the config file for at
    least one language, e.g.:

        languages:
          per_language:
            csharp:
              taint:
                enabled: true

    Usage:
        falconeye taint C:/path/to/repo
        falconeye taint . --output json --output-file findings.json
        falconeye taint . --no-verify   # dump raw candidates for debugging
    """
    taint_command(
        path=path,
        output_format=output_format,
        output_file=output_file,
        max_hops=max_hops,
        verify=verify,
        config_path=config,
        console=console,
    )


@app.command(name="taint-confirm")
def taint_confirm(
    path: Path = typer.Argument(
        ...,
        help="Path to codebase (used to build call graph and detect entry points)",
        exists=True,
    ),
    findings_file: Path = typer.Option(
        ...,
        "--findings", "-f",
        help="JSON file with findings from `falconeye review --output json`",
        exists=True,
    ),
    output_format: str = typer.Option(
        "console",
        "--output", "-o",
        help="Output format: console or json",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Write enriched findings JSON to this file (with --output json)",
    ),
    languages: Optional[List[str]] = typer.Option(
        None,
        "--language",
        help="Languages to include (defaults to csharp)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
):
    """
    Confirm review findings by call-graph reachability analysis.

    Takes a JSON findings file produced by `falconeye review --output
    json`, builds a call graph from the workspace at PATH, detects
    entry points (HTTP controllers, middleware, etc.), and enriches
    each finding with a taint_confirmation block indicating whether
    its enclosing method is reachable from any entry point.

    Confirmation is ADDITIVE signal, not subtractive. Findings marked
    "not_confirmed" retain their original confidence -- they are NOT
    dismissed. The confirmation is a boost for reachable code, not a
    filter for the rest.

    Usage:
        falconeye review Portal --output json --output-file findings.json
        falconeye taint-confirm Portal --findings findings.json
        falconeye taint-confirm Portal -f findings.json -o json --output-file confirmed.json
    """
    taint_confirm_command(
        path=path,
        findings_file=findings_file,
        output_format=output_format,
        output_file=output_file,
        languages=languages,
        config_path=config,
        console=console,
    )


# Create projects subcommand group
projects_app = typer.Typer(
    name="projects",
    help="Manage indexed projects",
    no_args_is_help=True,
)

@projects_app.command(name="list")
def projects_list(
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """
    List all indexed projects.

    Shows a summary of all projects in the registry.
    """
    projects_list_command(config_path=config, console=console)


@projects_app.command(name="info")
def projects_info(
    project_id: str = typer.Argument(
        ...,
        help="Project ID to show info for",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """
    Show detailed information about a project.

    Displays project metadata, file statistics, and configuration.
    """
    projects_info_command(
        project_id=project_id,
        config_path=config,
        console=console,
    )


@projects_app.command(name="delete")
def projects_delete(
    project_id: str = typer.Argument(
        ...,
        help="Project ID to delete",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """
    Delete a project from the index.

    Removes all indexed data for the specified project.
    This action cannot be undone.
    """
    projects_delete_command(
        project_id=project_id,
        yes=yes,
        config_path=config,
        console=console,
    )


@projects_app.command(name="cleanup")
def projects_cleanup(
    project_id: str = typer.Argument(
        ...,
        help="Project ID to clean up",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
):
    """
    Clean up deleted files from a project.

    Removes metadata for files that have been deleted from disk.
    """
    projects_cleanup_command(
        project_id=project_id,
        yes=yes,
        config_path=config,
        console=console,
    )


# Add projects command group to main app
app.add_typer(projects_app, name="projects")


# Create skill subcommand group -- audit-playbook skills.
skill_app = typer.Typer(
    name="skill",
    help="List and run security audit skills (operator-authored audit playbooks)",
    no_args_is_help=True,
)


@skill_app.command(name="list")
def skill_list(
    show_patterns: bool = typer.Option(
        False,
        "--show-patterns",
        help="Expand each skill's Step 1 discovery patterns for inspection",
    ),
    skills_dir: Optional[Path] = typer.Option(
        None,
        "--skills-dir",
        help=(
            "Override the skills directory. Bypasses config.skills.enabled "
            "and the FALCONEYE_SKILLS_DIR env var. Useful for one-off inspection."
        ),
        exists=False,   # allow non-existent so we can show a helpful error
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
):
    """
    List available skills discovered under config.skills.directory.

    Prints a table of skill name, category, function, and pattern count.
    With --show-patterns, expands each skill to show the discovery
    patterns extracted from its Step 1 section -- useful for verifying
    that the loader is parsing your skill files as you intended.

    Requires config.skills.enabled: true and a valid skills directory.
    """
    skill_list_command(
        config_path=config,
        console=console,
        show_patterns=show_patterns,
        skills_dir_override=skills_dir,
    )


@skill_app.command(name="lint")
def skill_lint(
    skills_dir: Optional[Path] = typer.Option(
        None,
        "--skills-dir",
        help="Override the skills directory (bypasses config gating)",
        exists=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
):
    """
    Check skill files for problems without running them.

    Reports what the loader would otherwise skip silently: discovery
    regexes that Python can't compile or reads differently from
    grep -E (e.g. BRE-style \\| alternation), Step 1 grep commands the
    loader ignores, broken frontmatter, duplicate names, and skills
    that can't run yet. Exits 1 if any error is found.

    Usage:
        falconeye skill lint
        falconeye skill lint --skills-dir ~/skills
    """
    skill_lint_command(
        config_path=config,
        console=console,
        skills_dir_override=skills_dir,
    )



@skill_app.command(name="run")
def skill_run(
    skill_name: str = typer.Argument(
        ...,
        help="Name of the skill to run (matches SkillDefinition.name)",
    ),
    workspace: Path = typer.Argument(
        ...,
        help="Path to the codebase to audit",
        exists=True,
    ),
    output_format: str = typer.Option(
        "console",
        "--output", "-o",
        help="Output format: console or json",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Write JSON output to this file (requires --output json)",
    ),
    skills_dir: Optional[Path] = typer.Option(
        None,
        "--skills-dir",
        help="Override the skills directory (bypasses config gating)",
        exists=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
    confirm_taint: bool = typer.Option(
        False,
        "--confirm-taint",
        help=(
            "Run call-graph reachability confirmation on the skill's findings "
            "before output (same enrichment as `review --confirm-taint`). "
            "Requires taint to be enabled for at least one language in config."
        ),
    ),
):
    """
    Execute a skill against a workspace.

    Runs the skill's Step 1 grep discovery to locate suspicious code,
    then invokes the configured LLM once per file with discovery hits
    to evaluate the code against the skill's audit procedure. Emits
    findings in SecurityFinding shape (compatible with taint-confirm,
    reporters, and the shape-tolerant JSON loader).

    Requires a runnable skill (function=scan) with at least one
    Step 1 discovery pattern. Skills with function=verify or
    function=triage are rejected with a clear message pointing to
    their respective pipeline stages.

    Usage:
        falconeye skill run injection-prevention Portal
        falconeye skill run injection-prevention Portal --output json --output-file inj.json
        falconeye skill run injection-prevention Portal --skills-dir ~/skills
        falconeye skill run injection-prevention Portal --confirm-taint
    """
    skill_run_command(
        skill_name=skill_name,
        workspace=workspace,
        output_format=output_format,
        output_file=output_file,
        config_path=config,
        skills_dir_override=skills_dir,
        console=console,
        confirm_taint=confirm_taint,
    )



@skill_app.command(name="run-all")
def skill_run_all(
    workspace: Path = typer.Argument(
        ...,
        help="Path to the codebase to audit",
        exists=True,
    ),
    output_format: str = typer.Option(
        "console",
        "--output", "-o",
        help="Output format: console or json",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Write JSON output to this file (requires --output json)",
    ),
    dedupe: bool = typer.Option(
        True,
        "--dedupe/--no-dedupe",
        help=(
            "Collapse findings sharing (file, line, category). "
            "When multiple skills flag the same site, the highest-severity finding survives with tags merged from the dropped duplicates. Disable with --no-dedupe to see raw per-skill output."
        ),
    ),
    skills_dir: Optional[Path] = typer.Option(
        None,
        "--skills-dir",
        help="Override the skills directory (bypasses config gating)",
        exists=False,
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config file",
        exists=True,
    ),
    confirm_taint: bool = typer.Option(
        False,
        "--confirm-taint",
        help=(
            "Run call-graph reachability confirmation on the aggregated findings "
            "(after dedup) before output. Requires taint to be enabled for at "
            "least one language in config."
        ),
    ),
):
    """
    Execute every runnable skill against a workspace.

    Loads every skill from the configured directory, filters to
    those with function=scan and at least one Step 1 discovery
    pattern, runs each in sequence, and aggregates all findings
    into a single SecurityReview. Skipped and errored skills are
    surfaced in the batch summary so operators can see why they
    didn\'t contribute.

    When --dedupe is on (default), findings sharing (file, line,
    category) are collapsed to the highest-severity survivor. The
    survivor accumulates the skill:<name> tags of dropped
    duplicates so cross-skill provenance is visible.

    Usage:
        falconeye skill run-all Portal
        falconeye skill run-all Portal --output json --output-file all.json
        falconeye skill run-all Portal --no-dedupe
        falconeye skill run-all Portal --confirm-taint -o json --output-file all.json
    """
    skill_run_all_command(
        workspace=workspace,
        output_format=output_format,
        output_file=output_file,
        config_path=config,
        skills_dir_override=skills_dir,
        dedupe=dedupe,
        console=console,
        confirm_taint=confirm_taint,
    )


app.add_typer(skill_app, name="skill")



def main():
    """Main entry point."""
    try:
        # Show banner if not in help mode
        if len(sys.argv) > 1 and sys.argv[1] not in ['--help', '-h', '--version']:
            print_banner(console)
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print("\n[bold red]Error:[/bold red]", end=" ")
        console.print(str(e), markup=False)
        sys.exit(1)


if __name__ == "__main__":
    main()