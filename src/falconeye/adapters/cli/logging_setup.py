"""
CLI logging setup shared by every long-running command.

Lives in its own small module so the import-light command modules
(commands_skill, commands_taint, commands_taint_confirm) can apply it
without importing commands.py and, with it, the whole review pipeline.
"""

from __future__ import annotations

from pathlib import Path

from ...infrastructure.logging import FalconEyeLogger


def configure_cli_logging(config, verbose: bool) -> None:
    """Apply ``config.logging`` and the ``-v`` flag to the shared logger.

    Logs always go to ``config.logging.file`` (when set). The console
    gets them only with ``-v``, so normal runs show progress and
    summaries rather than log lines. Uses ``configure()``, not
    ``get_instance(...)``: the singleton already exists by the time a
    command runs (the LLM adapters create it during container
    creation), and get_instance ignores arguments after its first call.

    A config without a ``logging`` section (test stubs) is left alone.
    """
    cfg = getattr(config, "logging", None)
    if cfg is None:
        return
    FalconEyeLogger.get_instance().configure(
        level=cfg.level,
        log_file=Path(cfg.file) if cfg.file else None,
        console=verbose,
        rotation=cfg.rotation,
        retention_days=cfg.retention_days,
    )
