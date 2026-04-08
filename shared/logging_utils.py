"""Общая настройка logging и progress bar для CLI-скриптов."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ModuleNotFoundError:
    dist_packages = Path("/usr/lib/python3/dist-packages")
    if dist_packages.exists() and str(dist_packages) not in sys.path:
        sys.path.append(str(dist_packages))
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    else:
        raise


_CONSOLE = Console()
ProgressType = Progress


def get_console() -> Console:
    """Вернуть общий Console для логов и progress bar."""
    return _CONSOLE


def create_progress() -> Progress:
    """Вернуть общий layout progress bar на базе rich."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=get_console(),
        transient=False,
        expand=True,
    )


def configure_logging(level: int = logging.INFO) -> None:
    """Настроить logging проекта и приглушить шумные сторонние логи."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = RichHandler(
        console=get_console(),
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    noisy_default = os.getenv("NL2SQL_NOISY_LOG_LEVEL", "WARNING").upper()
    noisy_level = getattr(logging, noisy_default, logging.WARNING)

    for logger_name in ("httpx", "httpcore", "openai"):
        logging.getLogger(logger_name).setLevel(noisy_level)
