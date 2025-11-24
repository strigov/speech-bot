"""Logging configuration using structlog."""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, only console logging is used.
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add file handler if log_dir is provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "transcriber.log",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if log_dir is None else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)
