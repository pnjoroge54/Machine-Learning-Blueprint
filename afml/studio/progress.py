"""Progress records for studio runs."""

from loguru import logger


def log_progress(stage: str, progress: float, message: str = "", **extra) -> None:
    """
    Emit a progress record. ``progress`` is a fraction in [0, 1].

    Progress is a log record carrying ``stage`` and ``progress`` in its ``extra``
    fields, not a print or a separate callback. The studio UI reads these records
    from a run's JSONL sink, so one stream serves both the log view and the
    progress display.
    """
    logger.bind(stage=stage, progress=float(progress), **extra).info(
        message or f"{stage}: {progress:.0%}"
    )
