"""Tests for file-based application logging."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from infra.logging.setup import get_logger, setup_logging


@pytest.fixture
def logger_isolated() -> Iterator[None]:
    """Remove global loguru sinks before and after each test."""
    logger.remove()
    try:
        yield
    finally:
        logger.complete()
        logger.remove()


def test_setup_creates_directory_and_both_files(logger_isolated: None, tmp_path: Path) -> None:
    """The first application log creates both configured sink files."""
    log_dir = tmp_path / "nested" / "logs"
    setup_logging(log_dir=log_dir)
    get_logger("setup_test").info("initial record")
    logger.complete()
    assert (log_dir / "app.log").is_file()
    assert (log_dir / "reasoning.log").is_file()


def test_component_info_record_is_written_to_app_log(logger_isolated: None, tmp_path: Path) -> None:
    """Component loggers write INFO messages to the application sink."""
    setup_logging(log_dir=tmp_path)
    get_logger("mod").info("application marker")
    logger.complete()
    assert "application marker" in (tmp_path / "app.log").read_text(encoding="utf-8")


def test_reasoning_debug_record_is_written_to_reasoning_log(
    logger_isolated: None, tmp_path: Path
) -> None:
    """Reasoning-channel DEBUG records reach the dedicated sink."""
    setup_logging(log_dir=tmp_path)
    get_logger("mod").bind(channel="reasoning").debug("reasoning marker")
    logger.complete()
    assert "reasoning marker" in (tmp_path / "reasoning.log").read_text(encoding="utf-8")


def test_reasoning_records_are_excluded_from_app_log(logger_isolated: None, tmp_path: Path) -> None:
    """Reasoning-channel records are routed only to reasoning.log."""
    setup_logging(log_dir=tmp_path)
    get_logger("mod").bind(channel="reasoning").debug("private reasoning marker")
    logger.complete()
    assert "private reasoning marker" not in (tmp_path / "app.log").read_text(encoding="utf-8")
