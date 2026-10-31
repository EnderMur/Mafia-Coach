"""Configure application logging sinks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger, Record

_LOG_FORMAT = "{time:YYYY-MM-DDTHH:mm:ss.SSSZZ} | {level} | {name}:{function}:{line} | {message}"


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: str | Path = "logs",
    rotation: str = "10 MB",
    retention_days: int = 30,
) -> None:
    """Configure application and reasoning file sinks.

    Records marked with the reasoning channel are written only to reasoning.log.
    """
    logger.remove()
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger.add(
        directory / "app.log",
        level=level,
        rotation=rotation,
        retention=f"{retention_days} days",
        encoding="utf-8",
        enqueue=True,
        format=_LOG_FORMAT,
        filter=_exclude_reasoning,
    )
    logger.add(
        directory / "reasoning.log",
        level="DEBUG",
        rotation=rotation,
        retention=f"{retention_days} days",
        encoding="utf-8",
        enqueue=True,
        format=_LOG_FORMAT,
        filter=_only_reasoning,
    )


def get_logger(name: str) -> Logger:
    """Return a loguru logger bound with a component name."""
    return logger.bind(name=name)


def _exclude_reasoning(record: Record) -> bool:
    """Keep reasoning-channel records out of the general application log."""
    return bool(record["extra"].get("channel") != "reasoning")


def _only_reasoning(record: Record) -> bool:
    """Keep only reasoning-channel records in the reasoning log."""
    return bool(record["extra"].get("channel") == "reasoning")
