"""Where the studio keeps its ledger and run folders."""

from __future__ import annotations

import os
from pathlib import Path


def studio_home() -> Path:
    """
    ``AFML_STUDIO_HOME`` if set, else ``$AFML_DATA_DIR/studio``, else
    ``~/afml_data/studio``.
    """
    env = os.environ.get("AFML_STUDIO_HOME")
    if env:
        home = Path(env)
    else:
        data_dir = os.environ.get("AFML_DATA_DIR")
        home = (Path(data_dir) if data_dir else Path.home() / "afml_data") / "studio"
    home.mkdir(parents=True, exist_ok=True)
    return home


def ledger_path() -> Path:
    return studio_home() / "ledger.sqlite"


def runs_dir() -> Path:
    d = studio_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def repo_root() -> Path:
    """The checkout that contains the ``afml`` package."""
    return Path(__file__).resolve().parents[2]
