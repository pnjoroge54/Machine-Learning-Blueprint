"""
High-level Backtest Cache using the unified system.
"""

import hashlib
import cloudpickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from .unified_cache import CACHE_DIRS, cacheable


@dataclass
class BacktestMetadata:
    strategy_name: str
    parameters: Dict[str, Any]
    data_hash: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    run_id: str


class BacktestCache:
    """Simplified backtest caching layer."""

    def __init__(self):
        self.base_dir = CACHE_DIRS["backtest"]
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @cacheable(time_aware=True, auto_versioning=True)
    def run_backtest(self, strategy_name: str, parameters: Dict[str, Any], data: pd.DataFrame, **kwargs):
        """Example cached backtest entry point."""
        logger.info(f"Executing backtest for {strategy_name} with params {parameters}")
        # Replace with your actual backtest computation
        return {"metrics": {}, "trades": pd.DataFrame(), "equity_curve": pd.Series()}

    def save_result(self, strategy_name: str, parameters: Dict, data: pd.DataFrame, result: Dict):
        """Manual save with metadata."""
        data_hash = hashlib.md5(str(data.shape).encode() + str(data.columns).encode()).hexdigest()[:12]
        run_id = hashlib.md5(f"{strategy_name}_{parameters}_{data_hash}".encode()).hexdigest()[:16]

        path = self.base_dir / f"backtest_{run_id}.pkl"
        metadata = BacktestMetadata(
            strategy_name=strategy_name,
            parameters=parameters,
            data_hash=data_hash,
            start_date=data.index[0] if isinstance(data.index, pd.DatetimeIndex) else None,
            end_date=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None,
            run_id=run_id,
        )
        with open(path, "wb") as f:
            pickle.dump({"metadata": metadata, "result": result}, f)
        logger.info(f"Saved backtest result: {path}")
        return run_id


backtest_cache = BacktestCache()
