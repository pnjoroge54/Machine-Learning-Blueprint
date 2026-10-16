"""
Tests for strategies.trading_strategies (Bollinger and MA-crossover signals) and
cross_validation.trial_tracker.StrategyTrialTracker.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ..cross_validation.trial_tracker import StrategyTrialTracker
from ..strategies.trading_strategies import BollingerStrategy, MACrossoverStrategy

# --------------------------------------------------------------------------- strategies


def _bars(close):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h")
    return pd.DataFrame({"close": np.asarray(close, dtype=float)}, index=idx)


def test_bollinger_is_mean_reverting():
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 0.1, 120)
    close[80] = 103.0  # spike far above the upper band
    close[100] = 97.0  # drop far below the lower band
    signals = BollingerStrategy(window=20, std=2.0).generate_signals(_bars(close))
    assert signals.iloc[80] == -1  # sell the spike
    assert signals.iloc[100] == 1  # buy the drop
    assert set(signals.unique()) <= {-1, 0, 1}
    assert (signals.iloc[:19] == 0).all()  # bands undefined before the window fills


def test_ma_crossover_follows_the_trend():
    close = np.r_[np.linspace(100, 90, 60), np.linspace(90, 110, 60)]
    signals = MACrossoverStrategy(fast_window=5, slow_window=20).generate_signals(_bars(close))
    assert signals.iloc[50] == -1  # falling: fast below slow
    assert signals.iloc[-1] == 1  # rising: fast above slow
    assert set(signals.unique()) <= {-1, 0, 1}


def test_strategy_names_encode_parameters():
    a = BollingerStrategy(window=20, std=2.0).get_strategy_name()
    b = BollingerStrategy(window=30, std=2.0).get_strategy_name()
    assert a != b and a.startswith("Bollinger")
    assert BollingerStrategy().get_objective() == "mean_reversion"


# --------------------------------------------------------------------------- trial tracker


@pytest.fixture
def tracker(tmp_path):
    return StrategyTrialTracker(project_name="unit", storage_dir=str(tmp_path))


def test_tracker_counts_every_trial_and_unique_configs(tracker):
    assert tracker.get_trial_count() == 0
    tracker.log_trial("ma 10/30", {"fast": 10, "slow": 30}, ["ret", "vol"], sharpe_ratio=0.8)
    tracker.log_trial("ma 10/30 rerun", {"slow": 30, "fast": 10}, ["vol", "ret"], sharpe_ratio=0.7)
    n = tracker.log_trial("ma 20/50", {"fast": 20, "slow": 50}, ["ret"], sharpe_ratio=1.1)
    assert n == 3
    assert tracker.get_trial_count() == 3  # reruns still count toward N
    assert tracker.get_trial_count(unique_only=True) == 2  # order of params/features ignored


def test_tracker_persists_and_reloads(tmp_path):
    t1 = StrategyTrialTracker("persist", storage_dir=str(tmp_path))
    t1.log_trial("a", {"x": 1}, ["f"], sharpe_ratio=0.5)
    t2 = StrategyTrialTracker("persist", storage_dir=str(tmp_path))
    assert t2.get_trial_count() == 1
    assert json.loads((tmp_path / "persist_trials.json").read_text())[0]["description"] == "a"


def test_tracker_summary_and_dsr_export(tracker, tmp_path):
    assert tracker.get_summary()["total_trials"] == 0
    tracker.log_trial("a", {"x": 1}, ["f"], sharpe_ratio=0.5)
    tracker.log_trial("b", {"x": 2}, ["f"], sharpe_ratio=None)
    tracker.log_trial("c", {"x": 3}, ["f"], sharpe_ratio=1.5)
    summary = tracker.get_summary()
    assert summary["total_trials"] == 3 and summary["unique_configs"] == 3
    assert summary["best_sharpe"] == 1.5
    out = tmp_path / "dsr.json"
    export = tracker.export_for_dsr(str(out))
    assert export["N"] == 3 and export["N_unique"] == 3
    assert export["sharpe_ratios"] == [0.5, 1.5]
    assert json.loads(out.read_text())["N"] == 3
