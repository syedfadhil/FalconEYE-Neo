"""CLI commands for project management."""

from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ...infrastructure.di.container import DIContainer


def projects_list_command(config_path: Optional[str], console: Console):
    """
    List all indexed projects.

    Args:
        config_path: Config file path
        console: Rich console
    """
    console.print(Panel.fit(
        "[bold]Indexed Projects[/bold]",
        border_style="blue"
    ))

    try:
        # Create DI container
        container = DIContainer.create(config_path)
        registry = container.index_registry

        # Get all projects
        projects = registry.get_all_projects()

        if not projects:
            console.print("\n[yellow]No indexed projects found.[/yellow]")
            console.print("\nUse 'falconeye index <path>' to index a project.")
            return

        # Create table
        table = Table(
            title=f"\n{len(projects)} Project(s)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Project ID", style="green", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Type", style="cyan")
        table.add_column("Files", justify="right", style="yellow")
        table.add_column("Languages", style="magenta")
        table.add_column("Last Scanned", style="blue")

        # Add rows
        for project in projects:
            languages_str = ", ".join(project.languages[:3])
            if len(project.languages) > 3:
                languages_str += f" +{len(project.languages) - 3}"

            last_scan = project.last_full_scan.strftime("%Y-%m-%d %H:%M")

            table.add_row(
                project.project_id,
                project.project_name,
                project.project_type.value,
                str(project.total_files),
                languages_str,
                last_scan,
            )

        console.print(table)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise


def projects_info_command(
    project_id: str,
    config_path: Optional[str],
    console: Console,
):
    """
    Show detailed information about a project.

    Args:
        project_id: Project ID
        config_path: Config file path
        console: Rich console
    """
    console.print(Panel.fit(
        f"[bold]Project Info: {project_id}[/bold]",
        border_style="blue"
    ))

    try:
        # Create DI container
        container = DIContainer.create(config_path)
        registry = container.index_registry

        # Get project
        project = registry.get_project(project_id)

        if not project:
            console.print(f"\n[red]Project '{project_id}' not found.[/red]")
            console.print("\nUse 'falconeye projects list' to see all projects.")
            return

        # Display project info
        console.print("\n[bold]General Information[/bold]")
        console.print(f"  Project ID: [green]{project.project_id}[/green]")
        console.print(f"  Name: {project.project_name}")
        console.print(f"  Type: [cyan]{project.project_type.value}[/cyan]")
        console.print(f"  Root Path: {project.project_root}")

        if project.git_remote_url:
            console.print(f"  Git Remote: {project.git_remote_url}")

        if project.last_indexed_commit:
            console.print(f"  Last Commit: {project.last_indexed_commit[:12]}")

        console.print("\n[bold]Statistics[/bold]")
        console.print(f"  Total Files: [yellow]{project.total_files}[/yellow]")
        console.print(f"  Total Chunks: [yellow]{project.total_chunks}[/yellow]")
        console.print(f"  Languages: [magenta]{', '.join(project.languages)}[/magenta]")

        console.print("\n[bold]Timestamps[/bold]")
        console.print(f"  Created: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"  Updated: {project.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"  Last Scan: {project.last_full_scan.strftime('%Y-%m-%d %H:%M:%S')}")

        # Get file information
        files = registry.get_all_files(project_id)
        console.print(f"\n[bold]Files ({len(files)})[/bold]")

        if len(files) <= 20:
            # Show all files
            for file_meta in files[:20]:
                status_icon = "[green]+[/green]" if file_meta.status.value == "indexed" else "[red]-[/red]"
                console.print(f"  {status_icon} {file_meta.relative_path}")
        else:
            # Show first 10
            for file_meta in files[:10]:
                status_icon = "[green]+[/green]" if file_meta.status.value == "indexed" else "[red]-[/red]"
                console.print(f"  {status_icon} {file_meta.relative_path}")
            console.print(f"  ... and {len(files) - 10} more files")

        # Get statistics
        stats = registry.get_project_stats(project_id)
        if stats:
            console.print("\n[bold]File Status[/bold]")
            console.print(f"  Indexed: [green]{stats.get('indexed', 0)}[/green]")
            if stats.get('deleted', 0) > 0:
                console.print(f"  Deleted: [red]{stats.get('deleted', 0)}[/red]")
            if stats.get('modified', 0) > 0:
                console.print(f"  Modified: [yellow]{stats.get('modified', 0)}[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise


def projects_delete_command(
    project_id: str,
    yes: bool,
    config_path: Optional[str],
    console: Console,
):
    """
    Delete a project from the index.

    Args:
        project_id: Project ID
        yes: Skip confirmation
        config_path: Config file path
        console: Rich console
    """
    console.print(Panel.fit(
        f"[bold]Delete Project: {project_id}[/bold]",
        border_style="red"
    ))

    try:
        # Create DI container
        container = DIContainer.create(config_path)
        registry = container.index_registry

        # Check if project exists
        project = registry.get_project(project_id)
        if not project:
            console.print(f"\n[red]Project '{project_id}' not found.[/red]")
            return

        # Confirm deletion
        if not yes:
            console.print("\n[yellow]Warning:[/yellow] This will delete all indexed data for:")
            console.print(f"  Project: {project.project_name}")
            console.print(f"  Files: {project.total_files}")
            console.print(f"  Chunks: {project.total_chunks}")
            console.print("\nThis action cannot be undone.")

            import typer
            confirm = typer.confirm("\nAre you sure you want to delete this project?")
            if not confirm:
                console.print("\n[yellow]Deletion cancelled.[/yellow]")
                return

        # Delete project
        console.print(f"\n[yellow]Deleting project '{project_id}'...[/yellow]")
        registry.delete_project(project_id)

        console.print(f"[green]Project '{project_id}' deleted successfully.[/green]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise


def projects_cleanup_command(
    project_id: str,
    yes: bool,
    config_path: Optional[str],
    console: Console,
):
    """
    Clean up deleted files from a project.

    Args:
        project_id: Project ID
        yes: Skip confirmation
        config_path: Config file path
        console: Rich console
    """
    console.print(Panel.fit(
        f"[bold]Cleanup Project: {project_id}[/bold]",
        border_style="yellow"
    ))

    try:
        # Create DI container
        container = DIContainer.create(config_path)
        registry = container.index_registry

        # Check if project exists
        project = registry.get_project(project_id)
        if not project:
            console.print(f"\n[red]Project '{project_id}' not found.[/red]")
            return

        # Get deleted files
        from ...domain.value_objects.project_metadata import FileStatus
        deleted_files = registry.get_files_by_status(project_id, FileStatus.DELETED)

        if not deleted_files:
            console.print("\n[green]No deleted files to clean up.[/green]")
            return

        # Confirm cleanup
        console.print(f"\n[yellow]Found {len(deleted_files)} deleted file(s):[/yellow]")
        for file_meta in deleted_files[:10]:
            console.print(f"  - {file_meta.relative_path}")
        if len(deleted_files) > 10:
            console.print(f"  ... and {len(deleted_files) - 10} more")

        if not yes:
            import typer
            confirm = typer.confirm(f"\nRemove {len(deleted_files)} deleted file(s) from index?")
            if not confirm:
                console.print("\n[yellow]Cleanup cancelled.[/yellow]")
                return

        # Delete files
        console.print(f"\n[yellow]Cleaning up {len(deleted_files)} file(s)...[/yellow]")
        file_paths = [f.file_path for f in deleted_files]
        registry.delete_files_batch(project_id, file_paths)

        console.print(f"[green]Cleaned up {len(deleted_files)} file(s).[/green]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        raise
