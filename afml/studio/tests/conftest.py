import os
import tempfile
from pathlib import Path

# afml decides its cache directory at import time, so point it at a throwaway
# directory before anything imports afml.
_TMP = Path(tempfile.mkdtemp(prefix="afml_studio_tests_"))
os.environ.setdefault("AFML_CACHE", str(_TMP / "cache"))
os.environ["AFML_STUDIO_SKIP_CONDA_EXPORT"] = "1"
os.environ.setdefault("AFML_STUDIO_SKIP_PREWARM", "1")

import pytest  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh studio home (ledger and run folders) per test."""
    monkeypatch.setenv("AFML_STUDIO_HOME", str(tmp_path / "studio"))
    from afml.studio.ledger import get_ledger

    return get_ledger()


def base_config(stage="features", symbol="SYNTH", end="2024-01-31", **overrides):
    cfg = {
        "stage": stage,
        "data": {
            "symbol": symbol,
            "source": "synthetic",
            "source_params": {"seed": 3, "freq": "15min", "signal_strength": 0.05},
            "bar_type": "time",
            "bar_size": "M15",
            "start_date": "2024-01-01",
            "end_date": end,
        },
        "labels": {
            "strategy": {"name": "bollinger", "params": {"window": 20, "num_std": 1.5}},
            "target": {"name": "daily_vol", "params": {"lookback": 50}},
            "max_holding_period": {"hours": 12},
        },
        "features": {
            "steps": [
                {"name": "log_returns", "params": {"lags": [1, 4]}},
                {"name": "volatility", "params": {"window": 10}},
            ]
        },
        "importance": {
            "methods": ["mdi", "mda"],
            "n_splits": 3,
            "estimator": {"name": "random_forest", "params": {"n_estimators": 20, "n_jobs": 1}},
        },
    }
    cfg.update(overrides)
    return cfg
