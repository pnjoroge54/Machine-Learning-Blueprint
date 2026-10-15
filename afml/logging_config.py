"""
Single loguru configuration for afml.

Rules (AFML Studio plan, Section 5.6):

1. Only ``configure_logging`` calls ``logger.remove()`` or adds global sinks.
   Every other module calls ``logger`` and nothing else.
2. ``configure_logging`` installs an intercept handler on the root standard-library
   logger and redirects Python warnings, so Optuna, Streamlit, joblib and
   ``warnings.warn`` output all land in loguru.
3. Spawned processes inherit no sinks, so each worker subprocess calls
   ``configure_logging`` at start.
4. Progress is a log record carrying ``stage`` and ``progress`` in its ``extra``
   fields (see ``log_progress``), not a print or a separate callback.
"""

from __future__ import annotations

import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

_configured = False


class InterceptHandler(logging.Handler):
    """Route standard-library logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller that issued the logging call, skipping logging internals
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _tqdm_aware_write(message: str) -> None:
    try:
        from tqdm import tqdm

        tqdm.write(message, end="", file=sys.stderr)
    except Exception:
        sys.stderr.write(message)


def configure_logging(
    level: str = "INFO",
    jsonl_path: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> None:
    """
    Configure loguru once for the whole process.

    Args:
        level: Minimum level for the console sink.
        jsonl_path: Optional path for a serialized (JSONL) file sink at DEBUG level.
        console: Add a tqdm-aware console sink on stderr.
    """
    global _configured

    logger.remove()
    if console:
        logger.add(_tqdm_aware_write, level=level, format=CONSOLE_FORMAT, colorize=True)
    if jsonl_path is not None:
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(jsonl_path), level="DEBUG", serialize=True, enqueue=False)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    _configured = True


def is_configured() -> bool:
    return _configured


def log_progress(stage: str, progress: float, message: str = "", **extra) -> None:
    """
    Emit a progress record. ``progress`` is a fraction in [0, 1].

    The studio UI reads these records from a run's JSONL sink, so the same stream
    serves both the log view and the progress display.
    """
    logger.bind(stage=stage, progress=float(progress), **extra).info(
        message or f"{stage}: {progress:.0%}"
    )
