"""
Centralized logging configuration for AFML (tqdm‑aware version).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

# Optional tqdm support – fallback gracefully if not installed
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback sink that writes directly to stderr
    class _FallbackSink:
        def write(self, message: str) -> None:
            sys.stderr.write(message)
            sys.stderr.flush()


# Sentinel and handler IDs
_CONFIGURED = False
_CONSOLE_HANDLER_ID: int | None = None
_FILE_HANDLER_ID: int | None = None

# Console format (unchanged)
_CONSOLE_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# File format (unchanged)
_FILE_FMT = "{message}"


class _TqdmSink:
    """Loguru sink that writes through tqdm.write() for clean progress bars."""

    def write(self, message: str) -> None:
        if TQDM_AVAILABLE:
            tqdm.write(message, end="")
        else:
            # Fallback when tqdm is not installed
            sys.stderr.write(message)
            sys.stderr.flush()


def configure_logging() -> None:
    """Set up loguru handlers once – console uses tqdm.write() automatically."""
    global _CONFIGURED, _CONSOLE_HANDLER_ID, _FILE_HANDLER_ID

    if _CONFIGURED:
        return

    # Remove default handler
    logger.remove()

    level = os.getenv("AFML_LOG_LEVEL", "INFO").upper()

    # --- Console handler (tqdm‑aware) ---
    _CONSOLE_HANDLER_ID = logger.add(
        _TqdmSink(),
        format=_CONSOLE_FMT,
        level=level,
        colorize=True,  # ANSI colors work in tqdm.write() when supported
        enqueue=False,  # synchronous for notebooks / REPLs
    )

    # --- File handler (unchanged) ---
    log_dir_env = os.getenv("AFML_LOG_DIR")
    if log_dir_env == "":
        _FILE_HANDLER_ID = None
    else:
        if log_dir_env is not None:
            log_dir = Path(log_dir_env)
        else:
            cache_env = os.getenv("AFML_CACHE")
            if cache_env:
                log_dir = Path(cache_env) / "logs"
            else:
                log_dir = Path("logs")

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            _FILE_HANDLER_ID = logger.add(
                str(log_dir / "afml_{time:YYYYMMDD}.jsonl"),
                format=_FILE_FMT,
                serialize=True,
                level="DEBUG",
                rotation="500 MB",
                retention="30 days",
                compression="zip",
                enqueue=True,  # async file writes
            )
        except OSError as exc:
            # Fallback: warn on console but continue without file logging
            logger.warning(f"Could not create log directory {log_dir}: {exc}")
            _FILE_HANDLER_ID = None

    _CONFIGURED = True


def reconfigure_logging(
    level: str = "INFO",
    enable_file: bool = True,
) -> None:
    """Change console log level at runtime, keeping tqdm awareness."""
    global _CONSOLE_HANDLER_ID, _FILE_HANDLER_ID

    # Replace console handler with new level (still using _TqdmSink)
    if _CONSOLE_HANDLER_ID is not None:
        logger.remove(_CONSOLE_HANDLER_ID)

    _CONSOLE_HANDLER_ID = logger.add(
        _TqdmSink(),
        format=_CONSOLE_FMT,
        level=level.upper(),
        colorize=True,
        enqueue=False,
    )

    # Optionally disable file handler
    if not enable_file and _FILE_HANDLER_ID is not None:
        logger.remove(_FILE_HANDLER_ID)
        _FILE_HANDLER_ID = None


# Auto‑configure on import
configure_logging()
