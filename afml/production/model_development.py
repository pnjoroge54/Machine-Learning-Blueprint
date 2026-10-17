"""
model_development.py
--------------------
Production model development pipeline with optional Optuna HPO integration.

This module serves as the central orchestrator for the Machine Learning Blueprint,
integrating high-performance data processing, advanced labeling techniques,
and rigorous cross-validation protocols designed for non-IID financial data.

The pipeline now supports two training paths controlled by
`model_params['use_optuna']`:

    False (default): clf_hyper_fit_cached via GridSearchCV / RandomizedSearchCV.
    True:            optimize_trading_model via Optuna + HyperbandPruner.

When use_optuna=True the following changes apply:
  - Weight computation for HPO is handled internally by _WeightedEstimator;
    get_optimal_sample_weight is still run for meta-features and reporting.
  - train_model dispatches to _train_model_optuna.
  - self.study is populated with the completed Optuna study for visualization.

When calibrate=True (run() parameter):
  - CalibratorCV is fitted after train_model(), wrapping best_model.
  - The calibrator uses PurgedKFold with the same n_splits and pct_embargo
    as the HPO step to prevent temporal leakage.
  - Adoption is conditional: calibration is deployed only when the nested-CV
    diagnostic (calibration_diagnostic, default True) shows an out-of-sample
    Brier improvement that survives the safety gates in _calibration_improves
    (OOF support, ranking preserved, non-degenerate map). Otherwise the
    uncalibrated model is kept, self.calibrator_ is set back to None, and
    self.calibration_rejected_reason_ records why.
  - When adopted, self.calibrator_ holds the fitted CalibratorCV and
    best_model.predict_proba() returns calibrated probabilities directly —
    no additional step required.
  - ONNX export unwraps the calibrator and exports the inner estimator only.
    Apply calibrator_.calibrator_.predict() as a post-processing step on
    the ONNX model's raw probabilities in the deployed environment.
  - A hard-decision threshold is automatically tuned on OOF probabilities
    (calibrated when adopted, raw when calibration is rejected) and stored in
    self.decision_threshold_. This threshold is consumed by
    prepare_meta_labeling_inputs() and the calibration report.

Architecture Overview:
This pipeline represents a production-grade implementation of the "Advances in
Financial Machine Learning" (AFML) framework. It orchestrates the complex
interaction between labeling, sample weighting, and cross-validation while
maintaining rigorous data integrity through time-aware caching.

Key AFML Methodologies Implemented:
----------------------------------
1. Triple-Barrier Method (TBM):
   Moves beyond fixed-horizon labeling by utilizing dynamic profit-taking,
   stop-loss, and time-exhaustion barriers. This captures the path-dependency
   essential for realistic trading strategy modeling.

2. Sample Weighting & Time Decay:
   Addresses the issue of overlapping outcomes in financial time series.
   The pipeline searches for optimal weights using:
     - Uniqueness (tW): Weights inverse to the concurrency of labels.
     - Return (w): Weights based on the absolute magnitude of the price move.
     - Time Decay: Both linear and exponential decay to prioritize recent data.

3. Purged & Embargoed Cross-Validation:
   Prevents information leakage by removing training observations that overlap
   with the test set (purging) and adding a buffer following the test set
   (embargo) to account for serial correlation.

4. Meta-Feature Engineering:
   The pipeline calculates rolling performance metrics (Accuracy, Precision,
   Recall, F1) using Numba-accelerated functions. These "self-referential"
   metrics are fed back into the model, allowing it to adapt to changing
   market regimes and its own recent performance.

Pipeline Workflow:
-----------------
1. Data Loading: Fetches tick data and constructs specialized bars.
2. Feature Engineering: Generates primary indicators and time-based features.
3. Label Generation: Applies the Triple-Barrier Method to define 'bin' targets.
4. Weight Optimization: Evaluates multiple weighting schemes to find the best
   fit for the current market environment.
5. Meta-Feature Integration: Joins rolling performance metrics to the feature set.
6. Training/HPO: Executes either Scikit-learn or Optuna-based hyperparameter
   optimization with Purged-KFold validation.
7. Calibration (optional): Wraps best_model in CalibratorCV to correct
   systematic overconfidence before position sizing. Automatically tunes a
   hard-decision threshold on calibrated OOF probabilities.
8. Reporting: Generates HTML summaries and hyperparameter importance reports,
   now including threshold diagnostic tables.

Meta-Labeling Flow:
------------------
Primary models (is_primary=True or auto-detected) can hand off to a secondary
pipeline via prepare_meta_labeling_inputs(), which returns the events DataFrame
annotated with the primary model's predicted side column. Pass the result
directly as the `events` argument when constructing the secondary pipeline.

When strategy=None the pipeline uses every bar as a potential entry point with
no directional side signal, producing a symmetric triple-barrier label space
(bin ∈ {-1, 0, 1}). This is the correct default for a purely ML-driven pipeline
with no pre-defined entry logic.
"""

import html
import inspect
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from loguru import logger
from sklearn import clone
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from ..cache import (
    IncrementalBarCache,
    cacheable,
    get_cache_monitor,
    log_data_access,
    print_contamination_report,
)
from ..cross_validation.cross_validation import PurgedKFold, ml_cross_val_score
from ..data_structures import _INFO_BAR_TYPES, calculate_ticks_per_period, make_bars
from ..ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from ..features.meta_label_features import add_meta_label_features
from ..features.trading_session import get_time_features
from ..labeling.triple_barrier import add_vertical_barrier, get_event_weights, triple_barrier_labels
from ..mt5.tick_data_loader import tick_data_loader as loader
from ..sample_weights.optimized_attribution import get_weights_by_time_decay_optimized
from ..strategies.signal_processing import get_entries
from ..strategies.trading_strategies import BaseStrategy
from ..util.pipelines import MyPipeline, make_custom_pipeline, set_pipeline_params

# Bagging classifiers subsample X to NumPy internally, so their inner estimators
# are fitted without feature names. Passing a named DataFrame at inference time
# then triggers a benign "X has feature names, but ... was fitted without feature
# names" warning. Column order is preserved, so predictions are unaffected.
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but .* was fitted without feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", "All-NaN slice encountered")


# ============================================================================
# Cached data helpers
# ============================================================================

bar_cache = IncrementalBarCache()


def load_and_prepare_training_data(
    symbol: str,
    start_date: str,
    end_date: str,
    account_name: str,
    bar_type: str,
    bar_size: int | str = 100,
    price: str = "mid_price",
    path: Path | None = None,
    # --- Information bar calibration (preferred entry point) ---
    target_timeframe: str | None = None,
    ewm_span: int = 100,
    # --- Escape hatches (mutually exclusive with target_timeframe) ---
    exp_ticks_init: float | None = None,
    exp_imbalance_init: float | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load tick data and construct OHLC bars for training or analysis.

    This wrapper encapsulates the full pipeline: tick data retrieval,
    calibration of information bar parameters, bar construction, and
    audit logging. It surfaces the target-timeframe API as the default
    entry point for information bars, reserving raw parameter control
    for advanced users.

    Parameters
    ----------
    symbol : str
        Trading instrument symbol (e.g., 'EURUSD', 'AAPL', 'BTCUSD').
    start_date : str
        Training window start date, ISO format ('YYYY-MM-DD').
    end_date : str
        Training window end date, ISO format ('YYYY-MM-DD').
    account_name : str
        MT5 account identifier for tick data source (e.g., 'live', 'icmarkets').
    bar_type : str
        Specifies the bar sampling method. Valid values:

        Standard bars (clock or tick-based):
            'tick'    : fixed number of ticks per bar
            'time'    : fixed time interval per bar
            'volume'  : fixed cumulative volume per bar
            'dollar'  : fixed cumulative dollar per bar

        Information bars (information-driven sampling):
            'tick_imbalance'   : sample on cumulative tick imbalance
            'volume_imbalance' : sample on cumulative volume imbalance
            'dollar_imbalance' : sample on cumulative dollar imbalance
            'tick_runs'        : sample on dominant directional run (ticks)
            'volume_runs'      : sample on dominant directional run (volume)
            'dollar_runs'      : sample on dominant directional run (dollar)

    bar_size : int or str, default=100
        Standard bars only; ignored for information bars.

            For 'time' bars: MT5 timeframe string (e.g., 'M5', 'M15', 'H1').
            For 'tick', 'volume', 'dollar' bars: numeric count.
            For 'tick' bars, a timeframe string is auto-converted via
            calculate_ticks_per_period.

    price : str, default='mid_price'
        Price column used for OHLC construction. One of:
            'mid_price' : (bid + ask) / 2
            'bid'       : bid prices
            'ask'       : ask prices
            'bid_ask'   : separate OHLC for bid and ask (not supported
                          for information bars)

    path : Path, optional
        Override the loader's default tick data directory. If None,
        uses the configured path.

    Information Bar Calibration
    ===========================

    For information bar types ('*_imbalance', '*_runs'), calibration of the
    initial threshold parameters is required. Choose exactly one path:

    Path A: Auto-calibration (recommended)
    ------
    Pass ``target_timeframe`` (e.g., 'M15'). The function derives E_0[T]
    and E_0[|imbalance|] from the tick data so that bars close at roughly
    the target clock-time cadence on average.

    The realized cadence fluctuates with volatility and information flow
    — it is an anchor, not a hard guarantee. After the first ~``ewm_span``
    bars, the threshold becomes data-driven and independent of initial values.

    Example:

        >>> bars = load_and_prepare_training_data(
        ...     symbol="EURUSD",
        ...     start_date="2022-01-01",
        ...     end_date="2023-12-31",
        ...     account_name="live",
        ...     bar_type="dollar_imbalance",
        ...     target_timeframe="M15",
        ... )

    Path B: Manual calibration (advanced)
    ------
    Pass ``exp_ticks_init`` and optionally ``exp_imbalance_init``.
    Both are the initial values for the AFML threshold:

        |θ_T| ≥ E_0[T] · |E_0[imbalance per tick]|

    If ``exp_imbalance_init`` is omitted, it defaults to 0.1.

    Example:

        >>> bars = load_and_prepare_training_data(
        ...     symbol="SPX",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     account_name="icmarkets",
        ...     bar_type="tick_imbalance",
        ...     exp_ticks_init=5000,
        ...     exp_imbalance_init=0.02,
        ... )

    Constraint: Passing both ``target_timeframe`` and ``exp_ticks_init``
    raises ``ValueError``. The mutual exclusivity enforces a clear contract:
    either specify the desired cadence (Path A) or the threshold parameters
    directly (Path B), not both.

    target_timeframe : str, optional
        Information bars only. MT5 timeframe string representing the target
        bar cadence. Examples: 'M1', 'M5', 'M15', 'H1', 'D1'.
        Mutually exclusive with ``exp_ticks_init``.

    ewm_span : int, default=20
        Information bars only. Exponential-weighted-moving average span
        (in bars) controlling how quickly the threshold adapts after each
        bar closes. Smaller values (5–10) adapt faster to changing tick
        rates; larger values (30–50) provide more stable transient behavior.

        The adaptation primarily affects the first ``ewm_span`` bars;
        after that, the threshold converges to its data-driven equilibrium.

    exp_ticks_init : int, float, or None
        Information bars only. Initial E_0[T] (expected ticks per bar)
        for manual calibration. Must be a positive number; typical range
        is 100–10,000 depending on instrument tick rate and target cadence.

        Mutually exclusive with ``target_timeframe``. If neither
        ``target_timeframe`` nor ``exp_ticks_init`` is provided for an
        information bar type, raises ``ValueError``.

    exp_imbalance_init : float, optional
        Information bars only. Initial E_0[|imbalance per tick|] for
        manual calibration.

        For tick imbalance/runs bars: represents expected directional bias,
        bounded above by 1.0 (since b_t ∈ {-1, +1}). Typical range: 0.01–0.1.
        For volume/dollar variants: scaled by average tick volume or
        dollar value, so ranges are higher. See make_bars docstring.

        If ``exp_ticks_init`` is provided but this is omitted, defaults
        to 0.1. Ignored if ``target_timeframe`` is provided.

    verbose : bool, default=True
        Log bar construction diagnostics, bar/tick counts, and DataFrame
        structure. Set to False to suppress logging for batch processing.

    Returns
    -------
    pd.DataFrame
        OHLC bars indexed by bar-close time (DatetimeIndex, UTC).
        Columns (in order):

            open, high, low, close    : float64, OHLC prices
            spread                    : float64, mean bid-ask spread per bar
            spread_bps                : float64, mean spread in basis points
            tick_volume               : float64, number of ticks per bar
            volume                    : float64 (if present in source), sum
                                        of cumulative tick volumes per bar
            tick_num                  : int64 (always), 1-based global tick
                                        index at which the bar closed

    Raises
    ------
    ValueError
        If ``bar_type`` is information but both ``target_timeframe`` and
        ``exp_ticks_init`` are provided. Pass one or the other.

    ValueError
        If ``bar_type`` is information but neither ``target_timeframe`` nor
        ``exp_ticks_init`` is provided. At least one is required.

    KeyError
        If ``bar_type`` requires 'volume' column (volume_* or dollar_* variants)
        but tick data lacks it.

    TypeError
        If tick DataFrame cannot be indexed by datetime.

    Notes
    -----
    1. **Caching & reproducibility**: Tick data is loaded once and cached;
       repeated calls with identical (symbol, date range, account) retrieve
       from cache. The cache is time-aware to prevent forward-looking bias
       in walk-forward backtests.

    2. **Audit logging**: Each call logs the dataset name, date range, shape,
       and purpose ('train', 'val', 'test') to an access log. This trail
       enables contamination detection if the same data is used in multiple
       roles.

    3. **EWM transient**: The initial E_0[T] and E_0[|imbalance|] only
       influence bar closure for the first ~``ewm_span`` bars. After that,
       the threshold is entirely determined by the empirical tick dynamics.
       This is why auto-calibration is safe — poor initial guesses only
       cause a brief transient, not systematic miscalibration.

    4. **Information bars are adaptive**: Unlike standard bars, which have
       a fixed closure rule, information bars' closure thresholds adapt
       per-bar. This makes them robust to changing market microstructure
       but also means the realized bar count will fluctuate even with
       identical initial parameters across different time windows.

    Examples
    --------
    **Standard tick bars (no calibration needed):**

    >>> bars = load_and_prepare_training_data(
    ...     symbol="AAPL",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     account_name="tws",
    ...     bar_type="tick",
    ...     bar_size=500,
    ... )

    **Dollar imbalance bars, auto-calibrated to M15 cadence:**

    >>> bars = load_and_prepare_training_data(
    ...     symbol="EURUSD",
    ...     start_date="2022-01-01",
    ...     end_date="2023-12-31",
    ...     account_name="live",
    ...     bar_type="dollar_imbalance",
    ...     target_timeframe="M15",
    ... )

    **Tick imbalance bars, manual calibration:**

    >>> bars = load_and_prepare_training_data(
    ...     symbol="BTC/USDT",
    ...     start_date="2023-06-01",
    ...     end_date="2023-12-31",
    ...     account_name="binance",
    ...     bar_type="tick_imbalance",
    ...     exp_ticks_init=300,
    ...     exp_imbalance_init=0.01,
    ... )

    **Volume runs bars with custom EWM adaptation:**

    >>> bars = load_and_prepare_training_data(
    ...     symbol="ES",
    ...     start_date="2024-01-01",
    ...     end_date="2024-03-31",
    ...     account_name="icmarkets",
    ...     bar_type="volume_runs",
    ...     target_timeframe="M5",
    ...     ewm_span=30,  # slower adaptation
    ... )
    """
    # -----------------------------------------------------------------------
    # Load tick data
    # -----------------------------------------------------------------------

    if path is not None:
        loader.path = Path(path)

    tick_df = loader.get_tick_data(symbol, start_date, end_date, account_name)

    # -----------------------------------------------------------------------
    # Resolve bar_size for standard bars (e.g., 'M5' → numeric tick count)
    # -----------------------------------------------------------------------

    if bar_type == "tick" and isinstance(bar_size, str):
        bar_size = calculate_ticks_per_period(tick_df, bar_size)

    # -----------------------------------------------------------------------
    # Validate calibration parameters for information bars
    # -----------------------------------------------------------------------

    is_info_bar = bar_type in _INFO_BAR_TYPES

    if is_info_bar:
        raw_params_given = (exp_ticks_init is not None) or (exp_imbalance_init is not None)

        if target_timeframe is not None and raw_params_given:
            raise ValueError(
                f"For information bar type '{bar_type}': pass either "
                f"'target_timeframe' (recommended, auto-calibrated) OR "
                f"'exp_ticks_init'/'exp_imbalance_init' (advanced, manual), "
                f"not both. See docstring for examples."
            )

        if target_timeframe is None and exp_ticks_init is None:
            raise ValueError(
                f"Information bar type '{bar_type}' requires calibration. "
                f"Pass either 'target_timeframe' (e.g., 'M15') for "
                f"auto-calibration or 'exp_ticks_init' for manual control. "
                f"See docstring for examples."
            )

        bar_size = (
            target_timeframe if target_timeframe is not None else bar_size
        )  # For info bars, bar_size is effectively the target_timeframe for logging

    if not _INFO_BAR_TYPES.intersection({bar_type}):
        target_timeframe = None
        ewm_span = None
        exp_ticks_init = None
        exp_imbalance_init = None

    # -----------------------------------------------------------------------
    # Construct bars via the caching layer
    # -----------------------------------------------------------------------

    # data = bar_cache.get_or_extend(
    data = _make_bars(
        tick_df=tick_df,
        bar_type=bar_type,
        bar_size=bar_size,
        price=price,
        target_timeframe=target_timeframe,
        ewm_span=ewm_span,
        exp_ticks_init=exp_ticks_init,
        exp_imbalance_init=exp_imbalance_init,
        # verbose=True,
    )

    # -----------------------------------------------------------------------
    # Audit log for contamination tracking
    # -----------------------------------------------------------------------

    log_data_access(
        dataset_name=f"{symbol}_{bar_type}_{bar_size}_{price}".lower(),
        start_date=data.index[0],
        end_date=data.index[-1],
        purpose="train",
        data_shape=data.shape,
    )

    if verbose:
        logger.info(
            f"Loaded {data.shape[0]:,} {bar_type} bars for {symbol} "
            f"({start_date} to {end_date}, {price})"
        )

    return data


@cacheable(time_aware=True)
def _make_bars(
    tick_df,
    bar_type,
    bar_size,
    price,
    target_timeframe,
    ewm_span,
    exp_ticks_init,
    exp_imbalance_init,
):
    return make_bars(
        tick_df=tick_df,
        bar_type=bar_type,
        bar_size=bar_size,
        price=price,
        verbose=True,
        target_timeframe=target_timeframe,
        ewm_span=ewm_span,
        exp_ticks_init=exp_ticks_init,
        exp_imbalance_init=exp_imbalance_init,
    )


@cacheable(time_aware=True)
def create_feature_engineering_pipeline(
    data: pd.DataFrame, ticks: pd.DataFrame, feature_config: dict, data_config: dict
) -> pd.DataFrame:
    """
    Compute engineered features with caching.

    Parameters
    ----------
    data : pd.DataFrame
        Input bar data.
    ticks : pd.DataFrame
        Input tick data.
    feature_config : dict
        Feature configuration.
        Expected keys:
        - func : callable
            Function that computes features from a DataFrame.
        - params : dict
            Parameters passed to `func`.
    data_config:
        Data configuration.
        Expected keys:
        - bar_size : str
            Bar size using MT5 naming conventions, e.g., M1, H1, D1.
        - bar_type : str
            Bar type should be one of "time", "tick", "volume", "dollar"

    Returns
    -------
    pd.DataFrame
        Feature matrix.

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    func = feature_config["func"]
    sig = inspect.signature(func)
    func_params = feature_config["params"].copy()
    key = "ticks"
    if key in sig.parameters.keys() and key not in func_params.keys():
        func_params[key] = ticks
    func_feat = func(data, **func_params)

    ## Determine if the symbol is a forex pair to apply appropriate time features.
    major_currencies = ["USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"]
    is_forex = any(currency in data_config.get("symbol", "") for currency in major_currencies)

    time_feat = get_time_features(
        data,
        timeframe=data_config["bar_size"],
        bar_type=data_config["bar_type"],
        forex=is_forex,
    )

    # micro_feat = compute_all_microfeatures(
    #     data, ticks, include_bar_features=(1 if "volume" in ticks.columns else 0)
    # )
    # trend_feat = (
    #     trend_scanning_labels(data["close"], span=(5, 50), lookforward=False)
    #     .drop(columns=["t1", "ret", "bin"])
    #     .rename(columns=lambda x: f"trend_{x}")
    # )
    # , micro_feat.ffill().bfill(), trend_feat
    features = func_feat.join([time_feat], how="left").dropna(how="all")
    return features


@cacheable(time_aware=True)
def generate_events_triple_barrier(
    data: pd.DataFrame,
    strategy: BaseStrategy | None,
    target_config: dict,
    profit_target: float = 1,
    stop_loss: float = 1,
    max_holding_period: dict[str, int] = dict(days=1),
    min_ret: float = 0.0,
    vertical_barrier_zero: bool = True,
    filter_as_series: bool = True,
) -> pd.DataFrame:
    """
    Generate trading events using the triple-barrier method.

    Parameters
    ----------
    data : pd.DataFrame
        Price bars with 'close' column.
    strategy : BaseStrategy or None
        Strategy instance implementing `generate_signals()`. When None, every
        bar in `data` is treated as a potential entry point and no directional
        side signal is applied, producing a symmetric label space
        (bin ∈ {-1, 0, 1}).
    target_config : dict
        Volatility target configuration.
        - func: Volatility target function
        - params: Function parameters
    profit_target : float, default=1
        Profit-taking threshold multiplier.
    stop_loss : float, default=1
        Stop-loss threshold multiplier.
    max_holding_period : dict, default={'days': 1}
        Maximum holding period for vertical barrier.
    min_ret : float, default=0.0
        Minimum return threshold.
    vertical_barrier_zero : bool, default=True
        Set label to zero if vertical barrier is reached.
    filter_as_series : bool, default=True
        Pass volatility threshold as series instead of scalar.

    Returns
    -------
    pd.DataFrame
        Event labels with columns:
        - 'bin' : {-1, 0, 1} for primary; {0, 1} for secondary (meta-label)
        - 't1'  : vertical barrier timestamps
        - 'w'   : sample weights
        - 'tW'  : uniqueness weights
        - 'side': directional signal (only present for secondary models)

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    data_dict = dict(
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        df=data,
        data=data,
        prices=data,
    )
    close = data["close"]
    target_func = target_config["func"]
    target_params = target_config["params"].copy()

    sig = inspect.signature(target_func)
    for key in sig.parameters.keys():
        if key not in target_params:
            target_params[key] = data_dict.get(key)

    try:
        target = target_func(**target_params)
    except Exception as e:
        logger.error(f"Target function failed: {e}")
        raise

    # ── Entry generation ──────────────────────────────────────────────────────
    if strategy is None:
        # No pre-defined signal: use every bar as a candidate entry.
        # side=None → symmetric triple-barrier (bin ∈ {-1, 0, 1}).
        side = None
        t_events = close.index
    else:
        if filter_as_series is None:
            side = strategy.generate_signals(data)
            t_events = side[side != 0].index
        else:
            filter_threshold = target if filter_as_series else target.mean()
            side, t_events = get_entries(strategy, data, filter_threshold)

    vb = add_vertical_barrier(t_events, close, **max_holding_period)
    events = triple_barrier_labels(
        close,
        target,
        t_events,
        vertical_barrier_times=vb,
        side_prediction=side,
        pt_sl=[profit_target, stop_loss],
        min_ret=min_ret,
        min_pct=0.05,
        vertical_barrier_zero=vertical_barrier_zero,
        drop=True,
        verbose=False,
    )
    events = get_event_weights(events, close)
    return events


# ============================================================================
# Pipeline Components
# ============================================================================


@cacheable(time_aware=True)
def _add_meta_label_features(events, prices, sample_weight):
    features = add_meta_label_features(events, prices, sample_weight)
    return features


@cacheable(time_aware=True)
def best_weighting_scheme(
    classifier,
    X,
    y,
    cv_gen,
    scoring,
    sample_weight_train,
    sample_weight_score,
    scheme=None,
    best_score=0,
    best_scheme=None,
    cv_results=pd.DataFrame(),
):
    scores = ml_cross_val_score(
        classifier,
        X,
        y,
        cv_gen,
        sample_weight_train=sample_weight_train,
        sample_weight_score=sample_weight_score,
        scoring=scoring,
    )
    score = scores.mean()
    cv_results[scheme] = scores

    if not np.isinf(score) and score > best_score:
        best_score = score
        best_scheme = scheme

    return best_score, best_scheme, cv_results


def get_optimal_sample_weight(
    data_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    features: pd.DataFrame,
    n_splits: int = 5,
    linear: bool = None,
    decay_factors: list | np.ndarray = np.arange(0.1, 1, 0.1),
) -> pd.Series:
    """
    Search-based optimization for sample weighting schemes.

    Financial Rationale:
    Financial observations are rarely IID. This function conducts a systematic
    search to find the weighting scheme that yields the highest cross-validated
    performance, effectively 'de-noising' the dataset.

    Evaluated Schemes:
    1. Uniqueness (tW): Weights samples based on how little they overlap with
       other concurrent labels.
    2. Return (w): Weights samples by the absolute log-return of the outcome.
    3. Time-Decay: Applies linear or exponential decay to prioritize recent
       market structure over distant history.

    Parameters
    ----------
    data_index: pd.DatetimeIndex
        Price data index.
    events : pd.DataFrame
        Event labels with uniqueness weights.
    features: pd.DataFrame
        Training features.
    n_splits : int, optional
        Number of cross-validation splits (default: 5).
    linear : bool, optional
        Default is None, which searches both linear and exponential time-decay.
        If True, use linear time-decay, if False, exponential.
    decay_factors: Union[list, np.ndarray]
        Time-decay factors to apply to best sample weight.

    Returns
    -------
    weights : pd.Series
        Computed sample weights.
    cv_results : dict
        Cross-validation results.
    """
    valid_index = features.dropna().index.intersection(events.index)
    cont = events.loc[valid_index]
    X = features.loc[valid_index]
    y = cont["bin"]

    classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            criterion="entropy",
            max_depth=4,
            max_features="sqrt",
            class_weight="balanced",
        ),
        n_estimators=200,
        random_state=42,
        max_samples=events["tW"].mean(),  # Use mean uniqueness as subsample size for bagging
        n_jobs=max(1, os.cpu_count() - 1),
    )

    cv_gen = PurgedKFold(n_splits=n_splits, t1=cont["t1"], pct_embargo=0.02)

    weights = {
        "unweighted": pd.Series(1.0, index=cont.index),
        "uniqueness": cont["tW"],
    }

    best_score, best_scheme = -np.inf, None
    cv_results = pd.DataFrame()
    meta_label = set(y.unique()) == {0, 1}
    scoring = "neg_log_loss"

    if meta_label:
        scoring = "f1"
    else:
        weights["return"] = cont["w"]

    pbar1 = tqdm(
        weights.items(),
        desc="Analyzing weighting schemes",
        total=len(weights),
        mininterval=1.0,
        leave=False,
        file=sys.stdout,
    )
    for scheme, weight in pbar1:
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier),
            X,
            y,
            cv_gen,
            scoring,
            weight,
            cont["w"],
            scheme,
            best_score,
            best_scheme,
            cv_results,
        )
        pbar1.set_postfix(
            {"scheme": scheme, "scoring": scoring, "score": f"{cv_results[scheme].mean():.4f}"}
        )

    best_weight = weights["unweighted"] if best_scheme is None else weights[best_scheme]
    linear_search = [1, 0] if linear is None else ([1] if linear else [0])

    time_decay_weights = {}
    for decay in decay_factors:
        for lin in linear_search:
            decay_vec = get_weights_by_time_decay_optimized(
                triple_barrier_events=cont,
                close_index=data_index,
                last_weight=decay,
                linear=lin,
                av_uniqueness=cont["tW"],
            )
            scheme = f"{best_scheme}_{'linear' if lin else 'exp'}_{decay}"
            time_decay_weights[scheme] = best_weight * decay_vec

    pbar2 = tqdm(
        time_decay_weights.items(),
        desc="Analyzing time-decay",
        total=len(time_decay_weights),
        mininterval=1.0,
        leave=False,
        file=sys.stdout,
    )
    for scheme, weight in pbar2:
        best_score, best_scheme, cv_results = best_weighting_scheme(
            clone(classifier),
            X,
            y,
            cv_gen,
            scoring,
            weight,
            cont["w"],
            scheme,
            best_score,
            best_scheme,
            cv_results,
        )
        pbar2.set_postfix(
            {"scheme": scheme, "scoring": scoring, "score": f"{cv_results[scheme].mean():.4f}"}
        )

    print(pformat({"best": best_scheme, "scoring": scoring, "score": round(float(best_score), 4)}))

    weights.update(time_decay_weights)
    cv_results_dict = {
        "best_score": float(best_score),
        "cv_results": cv_results,
        "scoring": scoring,
        "best_scheme": best_scheme,
    }
    return weights[best_scheme], cv_results_dict


def _fit_calibrator(
    X, y, sample_weight, t1, n_splits, pct_embargo, base_pipeline, method="isotonic"
):
    """Fit a deployment CalibratorCV, cached on the *content* of its inputs.

    Defined at module scope (not nested inside ``fit_calibrated_model``) on
    purpose. The previous nested version captured the freshly-built ``cv``
    object in its closure, and the @cacheable key generator hashes non-primitive
    closure cells by ``id()`` — a new memory address on every call — so two
    invocations with identical inputs produced different keys and the cache
    never hit. Passing the CV protocol (``t1``, ``n_splits``, ``pct_embargo``)
    and ``base_pipeline`` as arguments makes the key a deterministic function of
    the inputs (and correctly sensitive to the CV configuration), so the result
    is reused across calls and across process restarts.
    """
    from ..calibration.calibration import CalibratorCV

    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    return CalibratorCV(estimator=clone(base_pipeline), cv=cv, method=method).fit(
        X, y, sample_weight
    )


def _calibrated_oof_probabilities(calibrator) -> tuple[np.ndarray, np.ndarray]:
    """Return the valid-mask and calibrated probabilities for a fitted CalibratorCV."""
    raw_all = np.asarray(calibrator.oof_probs_, dtype=float)
    valid = ~np.isnan(raw_all)
    raw = raw_all[valid]

    if calibrator.method == "isotonic":
        calibrated = np.clip(calibrator.calibrator_.predict(raw), 0.0, 1.0)
    else:
        calibrated = calibrator.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]

    return valid, calibrated


def _oof_brier_summary(calibrator, X, y, cv) -> dict[str, float]:
    """Compute overall and per-fold std Brier scores from calibrated OOF probabilities."""
    valid, calibrated = _calibrated_oof_probabilities(calibrator)
    y_valid = y.to_numpy()[valid]
    squared_error = np.full(len(y), np.nan, dtype=float)
    squared_error[valid] = (calibrated - y_valid) ** 2

    fold_scores = []
    for _, test in cv.split(X):
        fold_error = squared_error[test]
        fold_error = fold_error[~np.isnan(fold_error)]
        if len(fold_error):
            fold_scores.append(float(np.mean(fold_error)))

    return {
        "mean": float(np.mean(squared_error[valid])),
        "std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
    }


@cacheable(time_aware=True)
def fit_calibrated_model(
    base_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    sample_weight_score: pd.Series,
    t1: pd.Series,
    n_splits: int,
    pct_embargo: float,
    calibration_diagnostic: bool = False,
    method: str = "isotonic",
    n_jobs: int | None = None,
):
    """
    Fit a CalibratorCV and (optionally) compute the raw-vs-calibrated CV
    diagnostic, with caching.

    Both expensive operations — the deployment calibrator fit and the nested-CV
    diagnostic — are keyed on the calibration inputs (base estimator
    hyperparameters, feature/label/weight content, CV protocol and the
    diagnostic flag) so an identical configuration reuses the persisted result
    instead of refitting.  See ``ModelDevelopmentPipeline.calibrate_model`` for
    how the returned objects are wired into the pipeline.

    Parameters
    ----------
    base_pipeline : estimator
        Unfitted base pipeline (already wrapped via ``make_custom_pipeline``) to
        be cloned for the calibrator and the per-fold diagnostic.
    X : pd.DataFrame
        Preprocessed feature matrix aligned to ``y``.
    y : pd.Series
        Binary labels.
    sample_weight : pd.Series
        Training sample weights.
    sample_weight_score : pd.Series
        Scoring sample weights (used only by the diagnostic).
    t1 : pd.Series
        Event end times for PurgedKFold.
    n_splits : int
        Number of CV folds.
    pct_embargo : float
        Embargo fraction for PurgedKFold.
    calibration_diagnostic : bool, optional
        When True, run the costly nested-CV raw-vs-calibrated Brier diagnostic
        (default False).
    method : {"isotonic", "platt"}, optional
        Calibration map for both the deployed calibrator and the diagnostic
        (default "isotonic"). "platt" (sigmoid) is a 2-parameter fit that is
        far more stable on small or overlapping OOF sets.
    n_jobs : int or None, optional
        Parallel jobs for the diagnostic's outer-CV fold loop (default None =
        serial). Passed to analyze_calibrated_cross_val_scores; only matters when
        calibration_diagnostic is True. Does not affect the deployment fit.

    Returns
    -------
    calibrator : CalibratorCV
        The fitted deployment calibrator.
    oof_brier_summary : dict
        Mean and fold standard deviation of the calibrated out-of-fold Brier
        score. Keys are ``mean`` and ``std``.
    calibration_scores : pd.DataFrame or None
        Per-fold raw-vs-calibrated CV scores when the diagnostic is enabled,
        else None.  A failed diagnostic is non-fatal and yields None.
    """
    from ..calibration.calibration import analyze_calibrated_cross_val_scores

    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)

    calibrator = _fit_calibrator(
        X,
        y,
        sample_weight,
        t1=t1,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        base_pipeline=base_pipeline,
        method=method,
    )

    oof_brier_summary = _oof_brier_summary(calibrator, X, y, cv)

    calibration_scores = None
    if calibration_diagnostic:
        try:
            _, calibration_scores, _ = analyze_calibrated_cross_val_scores(
                base_estimator=clone(base_pipeline),
                X=X,
                y=y,
                cv_gen=cv,
                sample_weight_train=sample_weight,
                sample_weight_score=sample_weight_score,
                calibrator_cv=cv,
                method=method,
                n_jobs=n_jobs,
            )
        except Exception as e:
            logger.warning(f"Calibration diagnostic (analyze_calibrated_*) skipped: {e}")

    return calibrator, oof_brier_summary, calibration_scores


# ============================================================================
# ModelDevelopmentPipeline
# ============================================================================


class ModelDevelopmentPipeline:
    """
    The central production controller for Model Training and HPO.

    This class encapsulates the state of the model development lifecycle,
    ensuring that hyperparameters, feature names, and evaluation metrics
    are kept in sync with the physical artifacts saved to disk.

    Core Responsibilities:
    - Pre-processing: Removes constant and duplicate features to reduce
      model variance and training time.
    - Backend Switching: Transparently toggles between Scikit-learn (Grid)
      and Optuna (Bayesian) optimization based on 'use_optuna' config.
    - Artifact Management: Automatically organizes models, parquet data,
      and HTML reports into a versioned directory structure. The model
      filename is prefixed with the bagging wrapper type (sbag/bag/plain)
      so artifacts are immediately distinguishable in the file system.
    - Calibration (optional): Wraps the trained model in CalibratorCV
      to correct systematic overconfidence before downstream bet sizing.
    - Analysis: Triggers feature importance calculation and automated
      contamination reports after every successful run.
    - Threshold Tuning: Automatically selects an optimal hard-decision
      threshold on calibrated OOF probabilities. The threshold is used
      for meta-labeling side generation and is documented in the
      calibration report.

    Meta-Labeling:
    - Primary models (is_primary=True or auto-detected) can hand off to a
      secondary pipeline via prepare_meta_labeling_inputs().
    - strategy=None instructs the pipeline to use every bar as a candidate
      entry point with no directional signal (symmetric barriers).
    """

    def __init__(
        self,
        data_config: dict,
        feature_config: dict,
        target_config: dict,
        label_config: dict,
        model_params: dict,
        strategy: BaseStrategy | None = None,
        is_primary: bool | None = None,
        base_dir: str = "Models",
    ):
        """
        Initialize the pipeline with configuration parameters.

        Parameters
        ----------
        data_config : dict
            Bar construction configuration.
            - symbol : str
            - start_date : str  ('YYYY-MM-DD')
            - end_date : str    ('YYYY-MM-DD')
            - account_name : str
            - bar_type : str    ('tick', 'volume', 'time')
            - bar_size : str    (e.g. 'M1', 'M5')
            - price : str       ('bid', 'ask', 'mid_price', 'bid_ask')
            - path : Union[str, Path] = None
        strategy : BaseStrategy or None, optional
            Signal-generating strategy.  When None every bar is treated as a
            potential entry point and no directional side signal is applied,
            producing a symmetric label space (bin ∈ {-1, 0, 1}).  Pass a
            concrete strategy only when entry timing or direction comes from a
            rule-based signal (e.g. moving-average crossover for meta-labeling).
        feature_config : dict
            - func: Feature engineering function
            - params: Function parameters
        target_config : dict
            - func: Volatility target function
            - params: Function parameters
        label_config : dict
            - profit_target : float
            - stop_loss : float
            - max_holding_period : dict
            - min_ret : float
            - vertical_barrier_zero : bool
            - filter_as_series : bool
        model_params : dict
            - pipe_clf : BaseEstimator or Pipeline or MyPipeline
            - param_grid : dict or list of dicts
            - n_splits : int, default=5
            - bagging_n_estimators : int, default=0
            - bagging_sequential : bool, default=False
            - bagging_max_samples : float or int, default=1.0
            - bagging_max_features : float or int, default=1.0
            - rnd_search_iter : int, default=0
            - n_jobs : int, default=-1
            - pct_embargo : float, default=0.02
            - random_state : int or None, default=None
            - use_optuna : bool, default=False
            - n_trials : int  (Optuna only)
            - timeout : int   (Optuna only)
            - pruner_type : str, default='hyperband'
            - verbose : int, default=0
        is_primary : bool or None, optional
            Explicit role override.
            - True  → treat as primary model regardless of events content.
            - False → treat as secondary (meta-labeling) model.
            - None  → auto-detect: primary if 'side' not in events.columns
                       (default, preserves original behaviour).
        base_dir : str
            Root directory for saved artifacts.
        """
        from .file_manager import ModelFileManager

        if data_config["target_timeframe"] is not None:
            if data_config["bar_type"] in _INFO_BAR_TYPES:
                # For info bars, bar_size is effectively the target_timeframe
                # for logging and feature engineering purposes,
                # even though the actual bar construction is adaptive.
                data_config["bar_size"] = data_config["target_timeframe"]
            else:
                data_config["target_timeframe"] = None

        self.data_config = data_config
        self.symbol = data_config["symbol"]
        self.train_start = data_config["start_date"]
        self.train_end = data_config["end_date"]
        self.strategy = strategy
        self.feature_config = feature_config
        self.label_config = label_config
        self.target_config = target_config
        self.account_name = data_config.get("account_name", "default")
        self.pipeline_version = "4.4"
        self.model_params = model_params

        # Explicit is_primary override; None means auto-detect in generate_labels.
        self._is_primary_override = is_primary

        self.config = data_config.copy()
        self.config.pop("path", None)
        self.config["training_start"] = self.config.pop("start_date")
        self.config["training_end"] = self.config.pop("end_date")
        self.config["strategy"] = (
            strategy.get_strategy_name() if strategy is not None else "ml_driven"
        )
        self.config["feature_func"] = feature_config["func"].__name__
        self.config["feature_params"] = feature_config["params"]
        self.config["target_func"] = target_config["func"].__name__
        self.config["target_params"] = target_config["params"]
        self.config.update(label_config)

        # ── State ─────────────────────────────────────────────────────────────
        self.bar_data = None
        self.features = None
        self.events = None
        self.sample_weight = None
        self.best_weighting_scheme = None
        self.meta_features = None
        self.preprocessed_features = None
        self.preprocessor = None
        self.dropped_columns_ = None
        self.best_model = None
        self.calibrator_ = None
        self.calibration_scores_ = None
        self.calibration_rejected_reason_ = None
        self.decision_threshold_ = 0.5
        self.threshold_objective_ = None
        self.threshold_diagnostic_ = None
        self.threshold_summary_ = None
        self.cv_results = None
        self.weight_cv_results = None
        self.feature_importance = None
        self.metrics = None
        self.study = None
        self.is_primary = None  # resolved in generate_labels()
        self.display = None
        self.calibrate = None

        # ── Model type: bagging wrapper prefix + base estimator token ─────────
        # The wrapper prefix is computed here — it depends only on model_params
        # and is therefore stable for the lifetime of this instance. It is
        # prepended to the base estimator token so the model filename in the
        # file system immediately identifies the ensemble strategy used.
        #
        # Examples:  sbag_rf  |  bag_rf  |  plain_rf
        #
        if isinstance(model_params["pipe_clf"], Pipeline):
            _base_model = model_params["pipe_clf"].steps[-1][1]
        else:
            _base_model = model_params["pipe_clf"]

        self.model_type = f"{self._bagging_wrapper}_{get_model_type(_base_model)}"

        self.file_manager = ModelFileManager(base_dir)
        self.file_paths = self.file_manager.setup_model_directory(self.config, self.model_type)

        self.decay_factors = [0.001, 0.1, 0.25, 0.5, 0.75, 0.9]

        self.completed_steps = {
            "data_loading": False,
            "feature_engineering": False,
            "label_generation": False,
            "weight_computation": False,
            "meta_features": False,
            "model_training": False,
            "calibration": False,
            "analysis": False,
        }

        self.log_file = self.file_paths["logs"] / "pipeline.log"
        self._setup_logging()
        self.n_splits = model_params.get("n_splits", 5)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _bagging_wrapper(self) -> str:
        """
        Canonical three-state label for the bagging wrapper in use.

        Computed purely from model_params so it is available from __init__
        onward.  Used as a prefix on model_type (and therefore on the saved
        model filename) so artifacts are immediately distinguishable without
        opening them.

        Returns
        -------
        str
            'sbag'  — SequentiallyBootstrappedBaggingClassifier
            'bag'   — standard BaggingClassifier
            'plain' — no bagging; base estimator used directly
        """
        sequential = self.model_params.get("bagging_sequential", False)
        n_estimators = self.model_params.get("bagging_n_estimators", 0)
        if sequential and n_estimators > 0:
            return "sbag"
        if n_estimators > 0:
            return "bag"
        return "plain"

    # ── Logging ───────────────────────────────────────────────────────────────

    def _setup_logging(self):
        self.logger = logger.bind(context=self.__class__.__name__)
        # Optional: add per-run file sink (removed when pipeline is GC'd)
        self._log_handler_id = logger.add(
            self.file_paths["logs"] / "pipeline.log",
            level="DEBUG",
            rotation="50 MB",
            filter=lambda record: record["extra"].get("context") == "ModelDevelopmentPipeline",
        )

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(
        self,
        generate_reports: bool = True,
        cache_reports: bool = False,
        save: bool = True,
        export_onnx: bool = False,
        calibrate: bool = False,
        display: bool = False,
        continue_study: bool = False,
        verbose: bool = True,
    ) -> tuple:
        """
        Run the complete model development pipeline with integrated reporting.

        Parameters
        ----------
        generate_reports : bool, optional
            Generate analysis reports (default: True).
        cache_reports : bool, optional
            Display cache performance reports (default: False).
        save : bool, optional
            Save model and artifacts (default: True).
        export_onnx : bool, optional
            Export model to ONNX format (default: False).
        calibrate : bool, optional
            Fit CalibratorCV on OOF predictions after training. When True, the
            calibrator is adopted (self.best_model is replaced, so downstream
            best_model.predict_proba() returns calibrated probabilities) only if
            it passes the out-of-sample adoption gate (_calibration_improves);
            otherwise the uncalibrated model is kept and
            self.calibration_rejected_reason_ records why.  Default: False.
        display : bool, optional
            Display the hyperparameter analysis report inline in a Jupyter
            notebook using IPython.display.  Default: False.
        verbose : bool, optional
            Print progress information (default: True).

        Returns
        -------
        tuple
            (best_model, feature_columns, metrics, config)
        """
        time0 = time.time()
        use_optuna = self.model_params.get("use_optuna", False)

        if verbose:
            print("\n" + "=" * 70)
            print(
                f"PRODUCTION MODEL DEVELOPMENT PIPELINE "
                f"(Backend: {'Optuna' if use_optuna else 'sklearn'})"
            )
            print("=" * 70)

        try:
            self.load_training_data()
            self.engineer_features()
            self.generate_labels()
            self.compute_sample_weights()
            self.add_meta_features()
            self.preprocess_features()
            self.train_model(continue_study)

            # ── Calibration (optional) ────────────────────────────────────────
            self.calibrate = calibrate
            if calibrate:
                self.calibrate_model()

            self.analyze_features()
            self._compile_metrics()

            if generate_reports:
                self._generate_analysis_reports(display=display)
            if cache_reports:
                self._display_cache_reports()
            if (save or export_onnx) and self.best_model is not None:
                self.export_onnx = export_onnx
                try:
                    self._save_all_artifacts()
                except Exception as e:
                    self.logger.error(f"Failed to save artifacts: {e}")
                    raise

            if verbose:
                elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace(
                    "0 days ", ""
                )
                print(f"\n✓ Completed in {elapsed}")

            return self.best_model, self._get_feature_names(), self.metrics, self.config

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def load_training_data(self):
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        ticks = loader.get_tick_data(
            self.data_config["symbol"],
            self.data_config["start_date"],
            self.data_config["end_date"],
            self.data_config["account_name"],
        )
        self.features = create_feature_engineering_pipeline(
            self.bar_data, ticks, self.feature_config, self.data_config
        )
        self.logger.info(f"features: ({self.features.shape[0]:,}, {self.features.shape[1]:,})")
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        """
        Apply the triple-barrier method and resolve the model's role.

        Role resolution order:
        1. If ``_is_primary_override`` was set at construction, use it.
        2. Otherwise auto-detect: primary when 'side' is absent from events.
        """
        self.events = generate_events_triple_barrier(
            self.bar_data,
            self.strategy,
            self.target_config,
            **self.label_config,
        )

        if self._is_primary_override is not None:
            self.is_primary = self._is_primary_override
        else:
            self.is_primary = "side" not in self.events.columns

        self.config["model_role"] = "primary" if self.is_primary else "secondary"
        self.logger.info(
            f"Model role: {self.config['model_role']} | "
            f"Label space: {np.unique(self.events['bin']).tolist()}"
        )
        self.logger.info(f"Average uniqueness: ({self.events['tW'].mean():.4f})")
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
            self.bar_data.index,
            self.events,
            self.features,
            self.n_splits,
            None,
            self.decay_factors,
        )
        self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
        self.logger.info(
            f"best_weighting_scheme: {self.best_weighting_scheme} ({self.weight_cv_results['scoring']}={self.weight_cv_results['best_score']:.4f})"
        )
        self.completed_steps["weight_computation"] = True

    def add_meta_features(self):
        """
        Append rolling performance meta-features for secondary models.

        Primary models skip this step — they have no prior predictions to
        reference.  Secondary (meta-labeling) models receive rolling accuracy,
        precision, recall, and F1 metrics derived from the primary model's
        label outcomes, allowing the secondary to adapt to changing regimes.
        """
        if self.is_primary:
            self.meta_features = pd.DataFrame(index=self.events.index)
            self.logger.info("Primary model — rolling meta-features skipped.")
        else:
            self.meta_features = _add_meta_label_features(
                events=self.events,
                prices=self.bar_data.loc[self.events.index],
                sample_weight=self.sample_weight,
            )
        self.completed_steps["meta_features"] = True

    def preprocess_features(self):
        if self.meta_features.empty:
            combined = self.features
        else:
            combined = self.features.join(self.meta_features, how="inner")

        # Drop columns that are entirely NaN *before* the row-wise dropna.
        # Otherwise a single all-NaN column (e.g. a directional meta-feature for
        # a side that never trades in a one-sided model) drops every row and
        # leaves a (0, n) frame, which DropConstantFeatures rejects.
        all_nan_cols = combined.columns[combined.isna().all()].tolist()
        if all_nan_cols:
            self.logger.warning(
                f"Dropping {len(all_nan_cols)} all-NaN feature column(s) before "
                f"preprocessing: {all_nan_cols}"
            )
            combined = combined.drop(columns=all_nan_cols)

        combined = combined.dropna()
        if combined.empty:
            raise ValueError(
                "No samples remain after dropping rows with NaN features. "
                f"Input had {self.features.shape[0]} rows; check feature warmup "
                "windows and meta-feature alignment."
            )

        self.preprocessor = Pipeline(
            [
                ("dcf", DropConstantFeatures()),
                ("ddf", DropDuplicateFeatures()),
            ]
        )
        self.preprocessed_features = self.preprocessor.fit_transform(combined)

        # Capture which columns each selector removed so they can be surfaced in
        # logs and the calibration report (feature_engine exposes
        # ``features_to_drop_`` on each fitted selector).
        constant_dropped = list(
            getattr(self.preprocessor.named_steps["dcf"], "features_to_drop_", [])
        )
        duplicate_dropped = list(
            getattr(self.preprocessor.named_steps["ddf"], "features_to_drop_", [])
        )
        self.dropped_columns_ = {
            "constant": constant_dropped,
            "duplicate": duplicate_dropped,
        }

        self.logger.info(
            f"Preprocessed features: {self.preprocessed_features.shape} "
            f"(dropped {combined.shape[0] - self.preprocessed_features.shape[0]:,} rows, "
            f"{combined.shape[1] - self.preprocessed_features.shape[1]} columns)"
        )
        if constant_dropped:
            self.logger.info(
                f"Dropped {len(constant_dropped)} constant column(s): \n{pformat(constant_dropped)}"
            )
        if duplicate_dropped:
            self.logger.info(
                f"Dropped {len(duplicate_dropped)} duplicate column(s): \n{pformat(duplicate_dropped)}"
            )
        self.events = self.events.loc[self.preprocessed_features.index]

    def train_model(self, continue_study):
        """
        Dispatch to the appropriate HPO backend.

        The Optuna study optimises the base estimator's hyperparameters only.
        The bagging wrapper (if any) is applied post-HPO using the tuned base
        estimator.  Because bagging is not part of the optimisation loop the
        wrapper type does not affect the study name or config hash — a single
        study's trials are valid priors regardless of which wrapper is
        subsequently applied.

        Post-dispatch (both paths):
            The fitted preprocessor (DropConstant + DropDuplicate) is prepended
            to best_model so that sklearn inference is fully self-contained.
            NOTE: for ONNX export the preprocessor step is stripped before
            conversion.  Apply self.preprocessor.transform() as a standalone
            step before passing data to the deployed ONNX model.
        """
        self.model_params["pipe_clf"] = make_custom_pipeline(self.model_params["pipe_clf"])
        pipe = clone(self.model_params["pipe_clf"])

        bagging_n_estimators = self.model_params.get("bagging_n_estimators", 0)
        bagging_sequential = self.model_params.get("bagging_sequential", False)
        # avgU-as-max_samples (AFML §6.2) only applies to the standard
        # BaggingClassifier; the sequential wrapper draws full-size samples and
        # corrects for overlap internally, so skip the auto-set there.
        if bagging_n_estimators > 0 and not bagging_sequential:
            if self.model_params.get("bagging_max_samples") is None:
                av_uniqueness = float(self.events["tW"].mean().round(2))
                self.model_params["bagging_max_samples"] = av_uniqueness
                self.logger.info(
                    f"bagging_max_samples set to average uniqueness ({av_uniqueness:.4f})"
                )

        self.model_params["pipe_clf"] = pipe

        if self.model_params.get("use_optuna", False):
            self._train_model_optuna(continue_study)
        else:
            self._train_model_sklearn()

        self.best_model = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                *self.best_model.steps,
            ]
        )
        self.best_model = set_pipeline_params(
            self.best_model, n_jobs=self.model_params.get("n_jobs", -1)
        )
        self.completed_steps["model_training"] = True

    def calibrate_model(self) -> None:
        """
        Fit CalibratorCV wrapping best_model on the full training data, then tune
        the hard-decision threshold from calibrated out-of-fold probabilities.
        """
        X = self.preprocessed_features.loc[self.events.index]
        y = self.events["bin"]
        sample_weight = self.sample_weight.loc[self.events.index]
        sample_weight_score = self.events["w"].loc[self.events.index]
        pct_embargo = self.model_params.get("pct_embargo", 0.01)
        # The diagnostic defaults to True: it is the only out-of-sample evidence
        # the adoption gate (_calibration_improves) can trust, so calibration is
        # not deployed without it. Explicitly setting it False keeps the
        # uncalibrated model (the gate has nothing to validate against).
        calibration_diagnostic = self.model_params.get("calibration_diagnostic", True)

        # Calibration map: "isotonic" (default, non-parametric) or "platt"
        # (sigmoid; 2-parameter, far more stable on small/overlapping OOF sets).
        # Accept "sigmoid" as a sklearn-style alias for "platt". The same method
        # is used for both the deployed calibrator and the per-fold diagnostic so
        # the report reflects what is shipped.
        calibration_method = self.model_params.get("calibration_method", "isotonic")
        if calibration_method == "sigmoid":
            calibration_method = "platt"

        # Parallelism for the (optional) nested-CV diagnostic's outer folds.
        # Prefer an explicit calibration_n_jobs, else fall back to the pipeline's
        # n_jobs. Only the diagnostic uses this; the deployment fit is unaffected.
        calibration_n_jobs = self.model_params.get(
            "calibration_n_jobs", self.model_params.get("n_jobs", None)
        )

        base_pipeline = make_custom_pipeline(self.best_model)

        self.calibrator_, oof_brier_summary, self.calibration_scores_ = fit_calibrated_model(
            base_pipeline=base_pipeline,
            X=X,
            y=y,
            sample_weight=sample_weight,
            sample_weight_score=sample_weight_score,
            t1=self.events["t1"],
            n_splits=self.n_splits,
            pct_embargo=pct_embargo,
            calibration_diagnostic=calibration_diagnostic,
            method=calibration_method,
            n_jobs=calibration_n_jobs,
        )

        if self.calibration_scores_ is not None:
            raw_brier = self.calibration_scores_.loc["raw_brier", "mean"]
            raw_brier_std = self.calibration_scores_.loc["raw_brier", "std"]
            cal_brier = self.calibration_scores_.loc["cal_brier", "mean"]
            cal_brier_std = self.calibration_scores_.loc["cal_brier", "std"]

            scoring = "accuracy" if self.is_primary else "f1"
            cv_label = (
                f"[{self._bagging_wrapper}] CV {scoring + ' ' * (len('Brier') - len(scoring))}"
            )
            raw_score_mean = self.calibration_scores_.loc[f"raw_{scoring}", "mean"]
            raw_score_std = self.calibration_scores_.loc[f"raw_{scoring}", "std"]
            cal_score_mean = self.calibration_scores_.loc[f"cal_{scoring}", "mean"]
            cal_score_std = self.calibration_scores_.loc[f"cal_{scoring}", "std"]

            self.logger.info(
                f"\n[{self._bagging_wrapper}] CV Brier — raw: (μ={raw_brier:.4f}, σ={raw_brier_std:.4f}), "
                f"calibrated: (μ={cal_brier:.4f}, σ={cal_brier_std:.4f}) \n"
                f"{cv_label} — raw: (μ={raw_score_mean:.4f}, σ={raw_score_std:.4f}), "
                f"calibrated: (μ={cal_score_mean:.4f}, σ={cal_score_std:.4f})"
            )

        # ── Adoption gate ─────────────────────────────────────────────────────
        # Only deploy the calibrated model when the nested-CV diagnostic shows an
        # out-of-sample improvement that survives the safety gates; otherwise keep
        # the simpler uncalibrated model. See _calibration_improves for the rules.
        adopt, reason = self._calibration_improves()
        self.calibration_rejected_reason_ = None if adopt else reason

        if adopt:
            self.best_model = self.calibrator_
            self._tune_decision_threshold()
            self.logger.info(
                f"\nCalibration adopted — {reason}.\n"
                "CalibratorCV OOF Brier score: "
                f"(μ={oof_brier_summary['mean']:.4f}, σ={oof_brier_summary['std']:.4f})"
            )
        else:
            # Keep the uncalibrated best_model. Tune the decision threshold on the
            # raw OOF probabilities (the data already exists and is strictly better
            # than the 0.5 default), then discard the calibrator so every downstream
            # consumer (ONNX export, badges, metrics, report) treats this run as
            # uncalibrated.
            self._tune_decision_threshold(use_calibrated=False)
            self.logger.warning(
                f"\nCalibration rejected — {reason}. Keeping uncalibrated model; "
                "decision threshold tuned on raw OOF probabilities."
            )
            self.calibrator_ = None

        self.completed_steps["calibration"] = True

    # ── Meta-labeling handoff ─────────────────────────────────────────────────

    def prepare_meta_labeling_inputs(self) -> pd.DataFrame:
        """
        Produce events annotated with the primary model's predicted side.

        The side signal is derived from calibrated probabilities when calibration
        is enabled, using the tuned decision threshold if available.
        """
        if self.best_model is None:
            raise RuntimeError(
                "Pipeline must complete training before calling prepare_meta_labeling_inputs()."
            )
        if not self.is_primary:
            raise RuntimeError("prepare_meta_labeling_inputs() is only valid for primary models.")

        X = self.preprocessed_features.loc[self.events.index]
        proba = self.best_model.predict_proba(X)

        threshold = getattr(self, "decision_threshold_", 0.5)
        side = np.where(proba[:, 1] >= threshold, 1, -1)

        meta_events = self.events.copy()
        meta_events["side"] = side
        meta_events["primary_prob"] = proba[:, 1]
        meta_events["primary_threshold"] = threshold
        return meta_events

    # ── Private training backends ─────────────────────────────────────────────

    def _train_model_sklearn(self):
        from ..cross_validation.hyper_fit import clf_hyper_fit_cached

        bagging_sequential = self.model_params.get("bagging_sequential", False)
        bagging_n = self.model_params.get("bagging_n_estimators", 0)
        sample_weight_train = self.sample_weight.loc[self.events.index]
        sample_weight_score = self.events["w"].loc[sample_weight_train.index]

        included = inspect.signature(clf_hyper_fit_cached).parameters.keys()
        params = {k: v for k, v in self.model_params.items() if k in included}

        if bagging_sequential and bagging_n > 0:
            params["bagging_n_estimators"] = 0
            tuned_pipeline, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],
                **params,
                sample_weight_train=sample_weight_train,
                sample_weight_score=sample_weight_score,
            )
            self.best_model = self._apply_sequential_bagging(
                self.preprocessed_features,
                self.events["bin"],
                tuned_pipeline,
                sample_weight=sample_weight_train,
            )
        else:
            self.best_model, self.cv_results = clf_hyper_fit_cached(
                features=self.preprocessed_features,
                labels=self.events["bin"],
                t1=self.events["t1"],
                **params,
                sample_weight_train=sample_weight_train,
                sample_weight_score=sample_weight_score,
            )

    def _train_model_optuna(self, continue_study):
        from ..cross_validation.optuna_hyper_fit import (
            check_for_overfitting,
            optimize_trading_model,
            pretty_print_trial,
        )

        X, y = self.preprocessed_features, self.events["bin"]
        base_clf = self.model_params["pipe_clf"].steps[-1][1]
        metric = "f1" if set(y.unique()) == {0, 1} else "neg_log_loss"

        included = inspect.signature(optimize_trading_model).parameters.keys()
        opt_params = {"metric": metric, "continue_study": continue_study}
        for k, v in self.model_params.items():
            if k == "param_grid":
                opt_params["param_distributions"] = v
            elif k in included:
                opt_params[k] = v

        # ── Study name ────────────────────────────────────────────────────────
        # Tokens that change the Optuna optimization surface:
        #   strategy  — determines entry set and label distribution
        #   symbol    — different instruments have different dynamics
        #   bar_type/size — sampling frequency changes autocorrelation structure
        #   role      — primary (ternary) vs secondary (binary) label space
        #   config hash — catches all remaining surface dimensions (CV protocol,
        #                 search space shape, barrier params, target function)
        #
        # The bagging wrapper is intentionally omitted: HPO optimises the base
        # estimator only; bagging is applied post-study.  A study's trials are
        # valid priors regardless of which wrapper is subsequently applied.
        _role = "pri" if self.is_primary else "sec"
        _config_hash = self._get_study_config_hash(metric=metric)

        opt_params["study_name"] = (
            f"{self.config['strategy']}"
            f"_{self.symbol}"
            f"_{self.data_config.get('bar_type', 'unk')}"
            f"{self.data_config.get('bar_size', 'unk')}"
            f"_{_role}"
            f"_s{_config_hash}"
        )

        db_path: Path = self.file_paths["db_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        opt_params["db_path"] = f"sqlite:///{db_path.resolve()}"
        opt_params["reports_path"] = self.file_paths["reports"] / "trials"
        callbacks = [check_for_overfitting, pretty_print_trial]

        try:
            from .optuna_dashboard import launch_optuna_dashboard

            launch_optuna_dashboard(storage=opt_params["db_path"], timeout=90, force_restart=True)
        except Exception as e:
            self.logger.error(e)

        self.study, cv_results_df = optimize_trading_model(
            classifier=base_clf,
            X=X,
            y=y,
            events=self.events,
            data_index=self.bar_data.index,
            refit=True,
            callbacks=callbacks,
            **opt_params,
        )

        self.logger.info(
            f"Optuna complete.\nBest score: {self.study.best_value:.4f}"
            f"\nBest params: {pformat(self.study.best_params)}"
        )

        best_estimator = make_custom_pipeline(self.study.best_estimator_.base_estimator)
        bagging_sequential = self.model_params.get("bagging_sequential", False)
        bagging_n_estimators = self.model_params.get("bagging_n_estimators", 0)
        bagging_max_samples = self.model_params.get("bagging_max_samples", 1.0)
        bagging_max_features = self.model_params.get("bagging_max_features", 1.0)
        n_jobs = self.model_params.get("n_jobs", -1)
        random_state = self.model_params.get("random_state", None)

        if bagging_sequential and bagging_n_estimators > 0:
            self.best_model = self._apply_sequential_bagging(
                X,
                y,
                best_estimator,
                sample_weight=self.study.best_estimator_.sample_weight_,
            )
        elif bagging_n_estimators > 0:
            time0 = time.time()
            base_est = set_pipeline_params(best_estimator, n_jobs=1)
            bag = BaggingClassifier(
                estimator=MyPipeline(base_est.steps),
                n_estimators=int(bagging_n_estimators),
                max_samples=bagging_max_samples,
                max_features=bagging_max_features,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            bag.fit(X, y, sample_weight=self.study.best_estimator_.sample_weight_)
            self.best_model = Pipeline([("bag", bag)])
            elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace(
                "0 days ", ""
            )
            self.logger.info(f"\n✓ BaggingClassifier fitted in {elapsed}")
        else:
            self.best_model = best_estimator

        pruner_type = self.model_params.get("pruner_type", "hyperband")
        self.cv_results = {
            "best_params": self.study.best_params,
            "best_score": self.study.best_value,
            "cv_results": cv_results_df,
            "scoring": metric,
            "search_method": "optuna",
            "pruner_type": pruner_type,
            "n_trials_completed": len([t for t in self.study.trials if t.state.name == "COMPLETE"]),
            "n_trials_pruned": len([t for t in self.study.trials if t.state.name == "PRUNED"]),
        }

    # ── Config hash ───────────────────────────────────────────────────────────

    @staticmethod
    def _canonicalize_search_space(param_grid: dict) -> list:
        """
        Produce a stable, JSON-serialisable representation of the search space
        that captures each parameter's *value space* (choices / ranges), not
        just its key.

        The Optuna study name embeds a hash of this representation.  Two studies
        may only be resumed into one another when their value spaces match
        exactly — otherwise ``trial.suggest_categorical`` on a changed choice
        list raises ``CategoricalDistribution does not support dynamic value
        space``, and changed numeric ranges silently corrupt the TPE surrogate.

        The value types mirror those dispatched in
        ``FinancialModelSuggester.suggest_and_apply``: lists (categorical),
        scipy.stats frozen distributions, ``range``, and scalar constants.
        """

        def encode(v):
            # scipy.stats frozen distribution (has .ppf and underlying .dist)
            if hasattr(v, "ppf") and hasattr(v, "dist"):
                return {
                    "kind": "scipy",
                    "name": v.dist.name,
                    "args": [repr(a) for a in v.args],
                    "kwds": {k: repr(v.kwds[k]) for k in sorted(v.kwds)},
                }
            if isinstance(v, range):
                return {"kind": "range", "start": v.start, "stop": v.stop, "step": v.step}
            if isinstance(v, (list, tuple)):
                return {"kind": "categorical", "choices": [repr(c) for c in v]}
            return {"kind": "const", "value": repr(v)}

        return [[k, encode(param_grid[k])] for k in sorted(param_grid)]

    def _get_study_config_hash(self, metric: str = "") -> str:
        """
        Return an 8-character SHA-256 prefix uniquely identifying the
        combination of parameters that determines the Optuna study's
        optimisation surface.

        Resuming an existing study is only correct when the optimisation
        surface is identical to the one those trials explored.  Every
        dimension that, if changed, should produce a fresh study is included.

        Covered dimensions
        ------------------
        model
            Base classifier type and constructor parameters.
        search
            Canonicalised search space — keys *and* their choices/ranges.
            Adding or removing a hyperparameter changes the dimensionality of
            the space; changing a categorical's choices or a numeric range
            changes the value space.  Either case must fork the study,
            otherwise resuming an existing study raises Optuna's
            "CategoricalDistribution does not support dynamic value space"
            (or silently corrupts the TPE surrogate for numeric params).
        cv
            n_splits and pct_embargo.
        metric
            'f1' (binary) or 'neg_log_loss' (ternary).
        role
            'primary' vs 'secondary'.
        label
            All label_config fields.
        target
            Volatility target function name and parameters.

        Note: bagging configuration is intentionally excluded.  HPO optimises
        the base estimator only; the wrapper applied afterwards does not change
        the optimisation surface.

        Parameters
        ----------
        metric : str
            Scoring metric passed to optimize_trading_model.

        Returns
        -------
        str
            8-character lowercase hex string (SHA-256 prefix).
        """
        import hashlib

        # ── 1. Base classifier ────────────────────────────────────────────────
        pipe = self.model_params["pipe_clf"]
        base_clf = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

        # ── 2. Search space shape *and* value space ───────────────────────────
        param_grid = self.model_params.get("param_grid", {})
        search_space = self._canonicalize_search_space(param_grid)

        # ── 3. CV protocol ────────────────────────────────────────────────────
        cv_config = {
            "n_splits": self.model_params.get("n_splits", 5),
            "pct_embargo": self.model_params.get("pct_embargo", 0.01),
        }

        # ── 4. Combine and hash ───────────────────────────────────────────────
        combined = {
            "model": type(base_clf).__name__,
            "search": search_space,
            "cv": cv_config,
            "metric": metric,
            "role": "primary" if self.is_primary else "secondary",
            "config": self.config,
        }

        digest = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()

        return digest[:8]

    # ── Bagging helpers ───────────────────────────────────────────────────────

    def _apply_sequential_bagging(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tuned_pipeline,
        sample_weight: pd.Series = None,
    ) -> Pipeline:
        """
        Wrap a tuned base pipeline in SequentiallyBootstrappedBaggingClassifier.

        After fitting, the trained estimators are transferred to a standard
        BaggingClassifier shell so that inference (predict / predict_proba) is
        available without requiring the events index at deployment time.
        """
        time0 = time.time()
        self.logger.info("\nSequential bootstrap is being fitted...")

        bagging_n = self.model_params.get("bagging_n_estimators", 0)
        # Sequential bootstrap already corrects for non-IID overlap inside the
        # draw (the indicator matrix lowers the probability of redundant
        # samples), so we draw a full-size sample. The avgU-as-max_samples
        # remedy (AFML §6.2) applies only to the standard BaggingClassifier
        # path, which still reads bagging_max_samples upstream.
        bagging_samples = 1.0
        bagging_feats = self.model_params.get("bagging_max_features", 1.0)
        random_state = self.model_params.get("random_state", 1)

        base_est = set_pipeline_params(
            tuned_pipeline, n_jobs=1, max_samples=self.events["tW"].mean(), random_state=None
        )

        bag = apply_seq_bootstrap(
            X=X,
            y=y,
            estimator=MyPipeline(base_est.steps),
            n_estimators=int(bagging_n),
            max_samples=bagging_samples,
            max_features=bagging_feats,
            samples_info_sets=self.events["t1"],
            price_bars_index=self.bar_data.index,
            random_state=random_state,
            sample_weight=sample_weight,
        )

        # Keep the genuine SequentiallyBootstrappedBaggingClassifier in the
        # pipeline (do NOT down-cast to a standard BaggingClassifier shell).
        # Inference (predict / predict_proba) is inherited from BaggingClassifier
        # and needs only estimators_, so the price bar index is not required at
        # deployment time.  Retaining the real class is what lets calibration
        # (CalibratorCV / analyze_calibrated_cross_val_scores) detect the
        # sequential sampler via its samples_info_sets and reproduce it per
        # fold; converting to a shell silently reverts refits to standard
        # bootstrapping.  Downstream artifact saving and ONNX export already
        # special-case SequentiallyBootstrappedBaggingClassifier.
        elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace("0 days ", "")
        self.logger.info(f"\n✓ Sequential bootstrap fitted in {elapsed}")

        return Pipeline([("seq_bag", bag)])

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze_features(self):
        from .weighted_estimator import _WeightedEstimator

        clf = self.best_model
        # Unwrap only when calibration was actually adopted (best_model is the
        # CalibratorCV). On the rejected-calibration path self.calibrate is still
        # True but best_model is the plain pipeline, which has no estimator_.
        if self.calibrator_ is not None:
            clf = clf.estimator_

        if hasattr(clf, "steps"):
            clf = clf.steps[-1][1]

        feat_names = self._get_feature_names()

        if isinstance(clf, (SequentiallyBootstrappedBaggingClassifier, BaggingClassifier)):
            importances = np.mean(
                [est.steps[-1][1].feature_importances_ for est in clf.estimators_], axis=0
            )
        elif isinstance(clf, _WeightedEstimator):
            try:
                importances = clf.base_estimator.feature_importances_
            except Exception as e:
                importances = np.zeros(len(feat_names))
                self.logger.error(e)
        else:
            try:
                importances = clf.feature_importances_
            except Exception as e:
                importances = np.zeros(len(feat_names))
                self.logger.error(e)

        self.feature_importance = pd.DataFrame(
            {
                "feature": feat_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        self.completed_steps["analysis"] = True

    def _compile_metrics(self):
        self.metrics = {
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
            "training_samples": len(self.bar_data),
            "feature_count": len(self._get_feature_names()),
            "best_weighting_scheme": self.best_weighting_scheme,
            "average_uniqueness": self.events["tW"].mean(),
            "completed_steps": self.completed_steps,
            "calibrated": self.calibrator_ is not None,
            "calibration_rejected_reason": self.calibration_rejected_reason_,
            "bagging_wrapper": self._bagging_wrapper,
            "model_role": self.config.get("model_role"),
        }

        if self.calibrator_ is not None:
            try:
                from ..calibration.calibration import calibration_report

                y_true, raw, calibrated = self._calibration_oof_arrays()
                report_df = calibration_report(y_true, raw, calibrated)
                self.metrics["calibration"] = report_df["value"].to_dict()
            except Exception as e:
                self.logger.warning(f"Calibration metrics compilation failed: {e}")

    def _get_feature_names(self):
        if self.preprocessed_features is None:
            return []
        return self.preprocessed_features.columns.tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _seq_bagging_to_standard(
        sb: SequentiallyBootstrappedBaggingClassifier,
    ) -> BaggingClassifier:
        """
        Down-cast a fitted SequentiallyBootstrappedBaggingClassifier to a
        standard, fitted BaggingClassifier.

        At inference the two are identical — predict / predict_proba are
        inherited unchanged from BaggingClassifier, and the sequential sampler
        only ever influences which rows the (already-fitted) base estimators saw
        during training.  sklearn-onnx ships a shape calculator/converter for
        BaggingClassifier but not for the custom subclass, so this conversion is
        applied only when preparing the model for ONNX export.  The live,
        in-memory model keeps the real class so calibration can reproduce the
        sequential bootstrap per fold.
        """
        bag = BaggingClassifier(
            estimator=sb.estimator,
            n_estimators=len(sb.estimators_),
            max_features=sb.max_features,
            bootstrap=sb.bootstrap,
            bootstrap_features=sb.bootstrap_features,
            random_state=sb.random_state,
            n_jobs=sb.n_jobs,
        )
        # Transfer the fitted state required for inference.
        bag.estimators_ = sb.estimators_
        bag.estimators_features_ = sb.estimators_features_
        bag.classes_ = sb.classes_
        bag.n_classes_ = sb.n_classes_
        bag.n_features_in_ = sb.n_features_in_
        if hasattr(sb, "estimator_"):
            bag.estimator_ = sb.estimator_
        return bag

    @staticmethod
    def _convert_mypipeline_for_onnx(pipeline: Pipeline) -> None:
        """
        Recursively replace every MyPipeline instance inside a fitted sklearn
        Pipeline with a standard sklearn Pipeline in-place, and down-cast any
        SequentiallyBootstrappedBaggingClassifier to a standard BaggingClassifier
        so sklearn-onnx can convert it.
        """
        for i, (name, step) in enumerate(pipeline.steps):
            # Down-cast the sequential bootstrap classifier first so the branch
            # below handles its base-estimator MyPipeline conversion uniformly.
            if isinstance(step, SequentiallyBootstrappedBaggingClassifier):
                step = ModelDevelopmentPipeline._seq_bagging_to_standard(step)
                pipeline.steps[i] = (name, step)

            if isinstance(step, MyPipeline):
                pipeline.steps[i] = (name, Pipeline(step.steps))
            elif isinstance(step, BaggingClassifier):
                if isinstance(step.estimator, MyPipeline):
                    step.estimator = Pipeline(step.estimator.steps)
                if hasattr(step, "estimators_"):
                    step.estimators_ = [
                        Pipeline(e.steps) if isinstance(e, MyPipeline) else e
                        for e in step.estimators_
                    ]

    def _save_all_artifacts(self):
        metadata = {
            "strategy": self.config["strategy"],
            "feature_names": self._get_feature_names(),
            "use_optuna": self.model_params.get("use_optuna", False),
            "pipeline_version": self.pipeline_version,
            "calibrated": self.calibrator_ is not None,
            "bagging_wrapper": self._bagging_wrapper,
            "model_role": self.config.get("model_role"),
        }
        self.file_manager.save_model(self.best_model, metadata)

        if self.strategy is not None:
            self.file_manager.save_object(self.strategy, "strategy")

        self.file_manager.save_dataframe(self.preprocessed_features, "features")
        self.file_manager.save_dataframe(self.events, "events")

        if self.sample_weight is not None:
            self.file_manager.save_dataframe(self.sample_weight.to_frame("weights"), "weights")

        self.file_manager.save_object(self.metrics, "metrics")

        if self.feature_config is not None:
            self.file_manager.save_object(self.feature_config, "feature_config")

        feature_names = self._get_feature_names()
        if feature_names:
            self.file_manager.save_object(feature_names, "feature_names")

        if self.preprocessor is not None:
            self.file_manager.save_object(self.preprocessor, "preprocessor")

        if self.target_config is not None:
            self.file_manager.save_object(self.target_config, "target_config")

        if self.calibrator_ is not None:
            self.file_manager.save_object(self.calibrator_.calibrator_, "calibrator")

        if self.export_onnx:
            if self.calibrator_ is not None:
                onnx_source = Pipeline(self.calibrator_.estimator_.steps[1:])
            else:
                onnx_source = Pipeline(self.best_model.steps[1:])

            self._convert_mypipeline_for_onnx(onnx_source)
            self.file_manager.save_model_as_onnx(onnx_source, self._get_feature_names(), metadata)

        self.logger.info(f"Saved all artifacts to {self.file_paths['base_dir']}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _report_context(self) -> dict:
        """Identifying fields shared by every generated report.

        Bundles the symbol, training date range and bar size/type so each
        report header states exactly which dataset and period it describes.
        ``side`` is populated only when the model was trained through
        BidAskLongShortPipeline (which sets ``price`` to ``bid``/``ask``), so
        the long and short report sets remain distinguishable.
        """
        bar_type = self.data_config.get("bar_type", "")
        bar_size = self.data_config.get("bar_size", "")
        bar_label = f"{bar_size} {bar_type}".strip() or "bars"
        price = self.data_config.get("price")
        side = None
        if price == "ask":
            side = "Long (Ask)"
        elif price == "bid":
            side = "Short (Bid)"
        return {
            "symbol": self.symbol,
            "bar_label": bar_label,
            "start_date": self.train_start,
            "end_date": self.train_end,
            "side": side,
        }

    def _report_subtitle(self, sep: str = " &nbsp;|&nbsp; ") -> str:
        """One-line dataset descriptor for HTML report headers."""
        ctx = self._report_context()
        parts = [
            ctx["symbol"],
            f"{ctx['bar_label']} bars",
            f"{ctx['start_date']} to {ctx['end_date']}",
        ]
        if ctx["side"]:
            parts.append(ctx["side"])
        return sep.join(parts)

    def _report_title_suffix(self) -> str:
        """Compact ``symbol | bars | dates [| side]`` suffix for <title>/suptitles."""
        return self._report_subtitle(sep=" | ").replace("&nbsp;", "")

    def _generate_analysis_reports(self, display: bool = False):
        """
        Generates hyperparameter analysis, importance plots, and HTML summary.

        Any column in cv_results["cv_results"] that is 100% NaN is dropped
        before calling generate_complete_hyperparameter_report to prevent the
        matplotlib "autodetected range of [nan, nan]" error when param_grid
        contains heterogeneous search spaces.
        """
        from ..cross_validation.hyper_fit_analysis import generate_complete_hyperparameter_report

        try:
            if self.cv_results and "cv_results" in self.cv_results:
                cv_df = pd.DataFrame(self.cv_results["cv_results"])
                nan_cols = cv_df.columns[cv_df.isna().all()].tolist()
                if nan_cols:
                    cv_df = cv_df.dropna(axis=1, how="all")
                    self.logger.info(
                        f"Hyperparameter report: dropped {len(nan_cols)} fully-NaN "
                        f"column(s): {nan_cols}"
                    )
                generate_complete_hyperparameter_report(
                    cv_results=cv_df,
                    strategy_config=self.config,
                    output_dir=self.file_paths["reports"],
                    display_in_notebook=display,
                )

            self._generate_training_summary_html()
            if self.study is not None:
                self._generate_optuna_report(display=display)
            if self.calibrator_ is not None:
                self._generate_calibration_report()

        except Exception as e:
            self.logger.warning(f"Report generation failed: {e}")

    def _generate_optuna_report(self, display: bool = False):
        """
        Save a self-contained HTML report with Optuna visualisation plots.
        """
        import matplotlib.pyplot as plt
        import optuna.visualization as vis
        from plotly.io import to_html

        report_path = self.file_paths["reports"] / "optuna_study_report.html"
        study = self.study

        n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
        n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])

        plot_specs = [
            (
                "Optimization History",
                "Objective value per trial. Dashed line shows the running best.",
                lambda: vis.plot_optimization_history(study),
            ),
            (
                "Fold Scores per Trial (Intermediate Values)",
                "Each line is one trial. Trials that end early were pruned.",
                lambda: vis.plot_intermediate_values(study),
            ),
            (
                "Hyperparameter Importances",
                "fANOVA estimate of each parameter's contribution to score variance.",
                lambda: vis.plot_param_importances(study),
            ),
            (
                "Parallel Coordinates",
                "One line per completed trial, coloured by score.",
                lambda: vis.plot_parallel_coordinate(study),
            ),
        ]

        html_parts = [
            f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Optuna Study Report — {self._report_title_suffix()}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a;
               color: #f1f5f9; padding: 40px; line-height: 1.6; }}
        h1   {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
        h2   {{ color: #7dd3fc; margin-top: 48px; }}
        p.caption {{ color: #64748b; font-size: 0.88rem; margin-top: 4px; }}
        .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 32px; }}
        .meta span {{ color: #22c55e; font-weight: 600; }}
        .plot {{ margin-bottom: 56px; }}
    </style>
</head>
<body>
<h1>Optuna Study Report</h1>
<p class="meta">
    {self._report_subtitle()}<br>
    Study: <b>{study.study_name}</b><br>
    Completed: <span>{n_complete}</span> trials &nbsp;|&nbsp;
    Pruned: <span style="color:#f97316">{n_pruned}</span> trials &nbsp;|&nbsp;
    Best score: <span>{study.best_value:.4f}</span>
</p>"""
        ]

        for title, caption, plot_fn in plot_specs:
            try:
                fig = plot_fn()
                html = to_html(fig, full_html=False, include_plotlyjs="cdn")
                html_parts.append(
                    f'<div class="plot"><h2>{title}</h2>'
                    f'<p class="caption">{caption}</p>{html}</div>'
                )
            except Exception as e:
                self.logger.warning(f"Optuna plot '{title}' failed: {e}")

        try:
            from ..cross_validation.optuna_hyper_fit import plot_model_vs_baseline

            original_backend = plt.get_backend()
            plt.switch_backend("agg")
            with plt.ioff():
                plot_model_vs_baseline(
                    study,
                    self.events["bin"],
                    self.events,
                    show=display,
                )
                plt.gcf().suptitle(
                    f"Model vs Baseline — {self._report_title_suffix()}",
                    fontsize=12,
                )
                baseline_path = self.file_paths["reports"] / "optuna_baseline_comparison.png"
                plt.savefig(
                    baseline_path,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor="#0f172a",
                    edgecolor="none",
                )
                plt.close("all")
            plt.switch_backend(original_backend)
            self.logger.info(f"Baseline comparison plot saved: {baseline_path}")
        except Exception as e:
            self.logger.warning(f"Baseline plot failed: {e}")

        html_parts.append("</body></html>")

        report_path.write_text("\n".join(html_parts), encoding="utf-8")
        self.logger.info(f"Optuna study report saved: {report_path}")

    def _generate_training_summary_html(self):
        """Constructs a comprehensive HTML training report."""
        try:
            report_path = self.file_paths["reports"] / "training_summary.html"

            best_score = self.cv_results.get("best_score", 0)
            search_method = "Optuna" if self.study is not None else "Scikit-Learn"
            calibrated = self.calibrator_ is not None

            # ── Calibration badge (three states) ──────────────────────────────
            # adopted        → calibrator passed the out-of-sample gate and is
            #                  deployed (green, links to the reliability report).
            # evaluated·rejected → calibration was fitted but the gate kept the
            #                  uncalibrated model; show why (amber).
            # not attempted  → calibrate=False (grey).
            calibration_rejected = (
                bool(self.calibrate)
                and not calibrated
                and self.calibration_rejected_reason_ is not None
            )
            if calibrated:
                cal_badge_class, cal_badge_text = "badge-on", "CalibratorCV"
                cal_badge_extra = (
                    '<br><a href="calibration_report.html" '
                    'style="color:#38bdf8;font-size:0.8rem;">View reliability report</a>'
                )
            elif calibration_rejected:
                cal_badge_class, cal_badge_text = "badge-warn", "Evaluated &middot; rejected"
                cal_badge_extra = (
                    '<br><span class="label" style="color:#fbbf24;font-size:0.8rem;">'
                    f"{html.escape(self.calibration_rejected_reason_)}</span>"
                )
            else:
                cal_badge_class, cal_badge_text = "badge-off", "No calibration"
                cal_badge_extra = ""

            # ── Dropped-columns card (preprocessing) ──────────────────────────
            # Columns removed by the preprocessing pipeline (DropConstantFeatures
            # → DropDuplicateFeatures). Surfaced here so it is always visible,
            # independent of whether calibration ran.
            dropped_section = ""
            if self.dropped_columns_ is not None:
                constant_dropped = self.dropped_columns_.get("constant", [])
                duplicate_dropped = self.dropped_columns_.get("duplicate", [])
                if constant_dropped or duplicate_dropped:
                    drop_rows = []
                    if constant_dropped:
                        drop_rows.append(
                            f"<tr><td>Constant</td><td>{len(constant_dropped)}</td>"
                            f"<td>{', '.join(map(str, constant_dropped))}</td></tr>"
                        )
                    if duplicate_dropped:
                        drop_rows.append(
                            f"<tr><td>Duplicate</td><td>{len(duplicate_dropped)}</td>"
                            f"<td>{', '.join(map(str, duplicate_dropped))}</td></tr>"
                        )
                    dropped_body = (
                        "<table><thead><tr><th>Reason</th><th>Count</th>"
                        "<th>Columns</th></tr></thead>"
                        f"<tbody>{''.join(drop_rows)}</tbody></table>"
                    )
                else:
                    dropped_body = '<p class="label">No columns were dropped.</p>'
                dropped_section = f"""
                    <div class="card">
                        <h2>Dropped Features (Preprocessing)</h2>
                        <p class="label">Columns removed by the preprocessing pipeline
                           (DropConstantFeatures &rarr; DropDuplicateFeatures) before training.</p>
                        {dropped_body}
                    </div>"""

            # ── Hyperparameter report link ────────────────────────────────────
            # Render the markdown report produced earlier in
            # _generate_analysis_reports (hyperparameter_analysis_report.md) into
            # a standalone HTML page and expose it as a link, mirroring the
            # calibration "View reliability report" link. Absent when no
            # cv_results were available; failures are non-fatal.
            hyperparameter_link = ""
            md_path = self.file_paths["reports"] / "hyperparameter_analysis_report.md"
            if md_path.exists():
                try:
                    html_name = self._render_hyperparameter_report_html(md_path)
                    hyperparameter_link = (
                        f'<br><a href="{html_name}" '
                        'style="color:#38bdf8;font-size:0.8rem;">'
                        "View hyperparameter report</a>"
                    )
                except Exception as e:
                    self.logger.warning(f"Hyperparameter report link generation failed: {e}")

            # ── Interactive panels ────────────────────────────────────────────
            # Both reuse data already computed during the run and degrade to ""
            # when their inputs are absent (e.g. calibration not attempted).
            calibration_decision_section = self._calibration_decision_html()
            threshold_section = self._threshold_curve_html()

            html_content = f"""
            <html>
            <head>
                <title>Training Report - {self._report_title_suffix()}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background-color: #0f172a; color: #f1f5f9; padding: 40px;
                            line-height: 1.6; }}
                    .container {{ max-width: 900px; margin: auto; }}
                    h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
                    .card {{ background-color: #1e293b; border-radius: 12px; padding: 24px;
                             margin-bottom: 24px; border: 1px solid #334155; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #334155; }}
                    th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase;
                          font-size: 0.8rem; }}
                    .metric {{ font-size: 1.5rem; font-weight: 700; color: #22c55e; }}
                    .label {{ color: #94a3b8; font-size: 0.9rem; }}
                    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
                              font-size: 0.78rem; font-weight: 600; }}
                    .badge-on  {{ background: #14532d; color: #4ade80; }}
                    .badge-off {{ background: #1e293b; color: #64748b; border: 1px solid #334155; }}
                    .badge-warn {{ background: #422006; color: #fbbf24; border: 1px solid #92400e; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Training Summary: {self.symbol}</h1>
                    <p class="label">{self._report_subtitle()}</p>
                    <p class="label">Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

                    <div class="card">
                        <h2>Performance Snapshot</h2>
                        <table>
                            <tr>
                                <td>
                                    <span class="label">
                                        Primary Metric ({self.cv_results.get("scoring", "F1")})
                                    </span><br>
                                    <span class="metric">{best_score:.4f}</span>
                                </td>
                                <td>
                                    <span class="label">Backend</span><br>
                                    <strong>{search_method}</strong>
                                    {hyperparameter_link}
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Training Samples</span><br>
                                    <strong>{len(self.events)}</strong>
                                </td>
                                <td>
                                    <span class="label">Average Uniqueness</span><br>
                                    <strong>{self.events["tW"].mean():.4f}</strong>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Model Role</span><br>
                                    <strong>{self.config.get("model_role", "N/A").capitalize()}</strong>
                                </td>
                                <td>
                                    <span class="label">Bagging Wrapper</span><br>
                                    <strong>{self._bagging_wrapper}</strong>
                                </td>
                            </tr>
                            <tr>
                                <td>
                                    <span class="label">Calibrated</span><br>
                                    <span class="badge {cal_badge_class}">
                                        {cal_badge_text}
                                    </span>
                                    {cal_badge_extra}
                                </td>
                                <td></td>
                            </tr>
                        </table>
                    </div>

                    <div class="card">
                        <h2>Weighting Logic</h2>
                        <p><strong>Selected Scheme:</strong>
                           {self.best_weighting_scheme or "Standard/Time-Decay"}</p>
                        <p class="label">
                            Weights were optimized via Purged-KFold to minimize
                            serial correlation leakage.
                        </p>
                    </div>
                    {dropped_section}
                </div>
            </body>
            </html>
            """

            report_path.write_text(html_content, encoding="utf-8")
            self.logger.info(f"Generated HTML summary report: {report_path}")

        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")

    def _render_hyperparameter_report_html(self, md_path: Path) -> str:
        """
        Render the hyperparameter analysis markdown into a standalone, themed
        HTML page next to it and return the HTML file name.

        The page reuses the dark report theme so the link from the training
        summary opens a properly rendered report (matching calibration_report.html),
        rather than raw markdown. Returns the file name for use as a relative href.
        """
        import markdown as _markdown

        body = _markdown.markdown(
            md_path.read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code", "sane_lists"],
        )
        html_path = md_path.with_suffix(".html")
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hyperparameter Report - {self._report_title_suffix()}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f172a; color: #f1f5f9; padding: 40px;
                line-height: 1.6; }}
        .container {{ max-width: 900px; margin: auto; }}
        h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
        h2 {{ color: #7dd3fc; margin-top: 28px; }}
        h3 {{ color: #7dd3fc; margin-top: 20px; font-size: 1.05rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 600; }}
        code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px;
                font-size: 0.85rem; }}
        pre {{ background: #1e293b; padding: 14px; border-radius: 8px;
               overflow-x: auto; border: 1px solid #334155; }}
        pre code {{ background: none; padding: 0; }}
        blockquote {{ border-left: 3px solid #334155; margin: 12px 0;
                      padding-left: 16px; color: #94a3b8; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="container">
        <p><a href="training_summary.html">&larr; Back to training summary</a></p>
        {body}
    </div>
</body>
</html>"""
        html_path.write_text(html_doc, encoding="utf-8")
        self.logger.info(f"Generated hyperparameter HTML report: {html_path}")
        return html_path.name

    def _calibration_oof_arrays(self):
        """
        Recover aligned OOF labels, raw probabilities, and calibrated probabilities.

        This method is intentionally pure: it does not tune thresholds, update model
        state, log decisions, or mutate report artifacts.
        """
        if self.calibrator_ is None:
            raise ValueError("No fitted calibrator available.")

        raw_all = np.asarray(self.calibrator_.oof_probs_, dtype=float)
        y_all = self.events["bin"].to_numpy()
        valid = ~np.isnan(raw_all)

        raw = raw_all[valid]
        y_true = y_all[valid]

        calibrator = self.calibrator_.calibrator_
        if self.calibrator_.method == "isotonic":
            calibrated = np.clip(calibrator.predict(raw), 0.0, 1.0)
        else:
            calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

        return y_true, raw, calibrated

    # Adoption-gate tolerances (see _calibration_improves).
    _CAL_AUC_TOL = 0.01  # max allowed AUC drop; monotone maps should preserve ranking.
    _CAL_MIN_SPREAD_RATIO = 0.25  # calibrated std must stay ≥ this fraction of raw std.
    _CAL_MIN_OOF = {"isotonic": 200, "platt": 50}  # min valid OOF obs per method.

    def _calibration_improves(self) -> tuple[bool, str]:
        """
        Decide whether the fitted ``CalibratorCV`` should be adopted as ``best_model``.

        Calibration is deployed only when the nested-CV diagnostic shows an
        out-of-sample Brier improvement beyond noise *and* a set of safety gates
        pass. Any failure keeps the simpler, uncalibrated model. The order is:

        0. **Out-of-sample evidence is mandatory.** Without ``calibration_scores_``
           (the nested-CV raw-vs-calibrated diagnostic) there is nothing to
           validate the calibration map against, so calibration is not adopted.
           This is also how an explicit ``calibration_diagnostic=False`` resolves.
        1. **OOF support.** Isotonic needs ~200 valid OOF observations (Platt ~50)
           for the map to be well-estimated; below that it is unreliable.
        2. **Ranking preserved.** Isotonic/Platt are monotone, so AUC must be
           essentially unchanged. A real drop signals a pathological map.
        3. **Non-degenerate map.** The calibrated probabilities must retain spread;
           an isotonic step function that collapses toward the base rate destroys
           bet sizing.
        4. **Benefit.** Out-of-sample Brier must improve by at least half a
           standard error, not merely on average.

        Returns
        -------
        (adopt, reason) : tuple[bool, str]
            ``adopt`` is True only when every gate passes; ``reason`` describes the
            deciding factor either way (for logging and ``calibration_rejected_reason_``).
        """
        cal = self.calibrator_
        if cal is None:
            return False, "no calibrator fitted"

        # Gate 0: out-of-sample evidence is mandatory.
        if self.calibration_scores_ is None:
            return False, "diagnostic unavailable; cannot validate calibration out-of-sample"

        # Gate 1: OOF support adequacy.
        valid = ~np.isnan(np.asarray(cal.oof_probs_, dtype=float))
        n_valid = int(valid.sum())
        min_oof = self._CAL_MIN_OOF.get(cal.method, 200)
        if n_valid < min_oof:
            return False, (
                f"insufficient OOF support ({n_valid} < {min_oof} for method='{cal.method}')"
            )

        y_true, raw, calibrated = self._calibration_oof_arrays()

        # Gate 2: discrimination (ranking) must be preserved.
        if len(np.unique(y_true)) >= 2:
            raw_auc = roc_auc_score(y_true, raw)
            cal_auc = roc_auc_score(y_true, calibrated)
            if cal_auc < raw_auc - self._CAL_AUC_TOL:
                return False, (
                    f"calibration degraded ranking AUC "
                    f"({cal_auc:.4f} < {raw_auc:.4f} − {self._CAL_AUC_TOL})"
                )

        # Gate 3: non-degenerate map (calibrated probabilities retain spread).
        raw_std = float(np.std(raw))
        cal_std = float(np.std(calibrated))
        if raw_std > 0 and cal_std < self._CAL_MIN_SPREAD_RATIO * raw_std:
            return False, (
                f"calibration collapsed probability spread "
                f"(std {cal_std:.4f} < {self._CAL_MIN_SPREAD_RATIO} × {raw_std:.4f})"
            )

        # Gate 4: out-of-sample Brier must improve beyond a half standard error.
        raw_brier = self.calibration_scores_.loc["raw_brier", "mean"]
        cal_brier = self.calibration_scores_.loc["cal_brier", "mean"]
        raw_brier_std = self.calibration_scores_.loc["raw_brier", "std"]
        cal_brier_std = self.calibration_scores_.loc["cal_brier", "std"]
        margin = 0.5 * max(raw_brier_std, cal_brier_std) / np.sqrt(self.n_splits)
        if cal_brier > raw_brier - margin:
            return False, (
                f"out-of-sample Brier not improved beyond noise "
                f"(cal={cal_brier:.4f}, raw={raw_brier:.4f}, margin={margin:.4f})"
            )

        improvement = raw_brier - cal_brier
        return True, (
            f"out-of-sample Brier improved by {improvement:.4f} "
            f"(cal={cal_brier:.4f} < raw={raw_brier:.4f})"
        )

    def _tune_decision_threshold(self, use_calibrated: bool = True) -> None:
        """
        Select a hard-decision threshold from out-of-fold probabilities.

        The default objective is weighted F1. The method stores both the selected
        threshold and a full threshold diagnostic table for reports.

        Parameters
        ----------
        use_calibrated : bool, default True
            Tune on the calibrated OOF probabilities (the deployed model when
            calibration is adopted). When False — used on the rejected-calibration
            fallback path — tune on the raw OOF probabilities instead, since the
            uncalibrated model is what will actually be deployed.
        """
        if self.calibrator_ is None:
            return

        y_true, raw, calibrated = self._calibration_oof_arrays()
        probs = calibrated if use_calibrated else raw
        valid = ~np.isnan(np.asarray(self.calibrator_.oof_probs_, dtype=float))
        sample_weight = self.sample_weight.loc[self.events.index].to_numpy()[valid]

        threshold_objective = "f1" if set(self.events["bin"].unique()) == {1, 0} else "neg_log_loss"
        objective = self.model_params.get("threshold_objective", threshold_objective)
        thresholds = np.unique(
            np.concatenate([np.quantile(probs, np.linspace(0.01, 0.99, 99)), [0.5]])
        )

        rows = []
        for threshold in thresholds:
            y_pred = (probs >= threshold).astype(int)

            precision = precision_score(
                y_true, y_pred, sample_weight=sample_weight, zero_division=0
            )
            recall = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
            f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1], sample_weight=sample_weight
            ).ravel()

            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "positive_rate": float(y_pred.mean()),
                    "n_positive": int(y_pred.sum()),
                    "tn": float(tn),
                    "fp": float(fp),
                    "fn": float(fn),
                    "tp": float(tp),
                }
            )

        diagnostic = pd.DataFrame(rows)
        best_idx = diagnostic[objective].idxmax()

        self.decision_threshold_ = float(diagnostic.loc[best_idx, "threshold"])
        self.threshold_objective_ = objective
        self.threshold_diagnostic_ = diagnostic

        default_idx = diagnostic.index[np.isclose(diagnostic["threshold"].to_numpy(), 0.5)][0]
        self.threshold_summary_ = {
            "objective": objective,
            "selected_threshold": self.decision_threshold_,
            "default_threshold": 0.5,
            "selected_score": float(diagnostic.loc[best_idx, objective]),
            "default_score": float(diagnostic.loc[default_idx, objective]),
            "selected_positive_rate": float(diagnostic.loc[best_idx, "positive_rate"]),
            "default_positive_rate": float(diagnostic.loc[default_idx, "positive_rate"]),
            "selected_n_positive": int(diagnostic.loc[best_idx, "n_positive"]),
            "default_n_positive": int(diagnostic.loc[default_idx, "n_positive"]),
        }

        self.logger.info(
            f"\nDecision threshold selected: {self.decision_threshold_:.4f} "
            f"({objective}={self.threshold_summary_['selected_score']:.4f}; "
            f"default {objective}={self.threshold_summary_['default_score']:.4f})"
        )

    def _generate_calibration_report(self):
        """
        Save a self-contained HTML calibration report with a raw-vs-calibrated
        reliability diagram and a Brier/ECE/MCE metrics table.

        Only produced when a CalibratorCV has been fitted (``calibrate=True``),
        since the raw and calibrated out-of-fold probabilities required for the
        reliability curves exist only then.  When the opt-in per-fold diagnostic
        is enabled (``model_params["calibration_diagnostic"]=True``), the
        per-fold raw-vs-calibrated CV scores are appended as a second table.

        Additionally, a decision threshold diagnostic table is included when
        threshold tuning has been performed.

        A failure here is non-fatal: it is logged as a warning and leaves the
        rest of the report generation unaffected.
        """
        if self.calibrator_ is None:
            return

        try:
            import base64
            from io import BytesIO

            import matplotlib.pyplot as plt

            from ..calibration.calibration import calibration_report, plot_reliability_with_ci

            y_true, raw, calibrated = self._calibration_oof_arrays()
            report_df = calibration_report(y_true, raw, calibrated)

            # ── Reliability diagram (raw vs calibrated) ───────────────────────
            original_backend = plt.get_backend()
            plt.switch_backend("agg")
            png_path = self.file_paths["reports"] / "calibration_reliability.png"
            with plt.ioff():
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                plot_reliability_with_ci(y_true, raw, ax=axes[0], title="Raw (uncalibrated)")
                plot_reliability_with_ci(y_true, calibrated, ax=axes[1], title="Calibrated")
                fig.suptitle(
                    f"Reliability — {self._report_title_suffix()}",
                    fontsize=13,
                    color="#f1f5f9",
                )
                fig.tight_layout()
                fig.savefig(
                    png_path,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor="#0f172a",
                    edgecolor="none",
                )
                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    dpi=150,
                    bbox_inches="tight",
                    facecolor="#0f172a",
                    edgecolor="none",
                )
                plt.close("all")
            plt.switch_backend(original_backend)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")

            # ── Metrics tables ────────────────────────────────────────────────
            # One row per metric (Brier, ECE, MCE) with original / calibrated /
            # improvement columns. All three are lower-is-better, so improvement
            # = original − calibrated; a positive value means calibration helped.
            # The improvement column is color-coded for at-a-glance reading,
            # mirroring the per-fold CV diagnostic table below.
            vals = report_df["value"].to_dict()
            metric_labels = {"brier": "Brier", "ece": "ECE", "mce": "MCE"}
            metric_rows = []
            for key, label in metric_labels.items():
                original = vals.get(f"original_{key}")
                calibrated = vals.get(f"calibrated_{key}")
                if original is None or calibrated is None:
                    continue
                improvement = original - calibrated
                if improvement == 0:
                    cls, arrow = "neutral", "—"
                elif improvement > 0:
                    cls, arrow = "better", "▼"  # error decreased
                else:
                    cls, arrow = "worse", "▲"  # error increased
                # Last column is metric-specific. For Brier we report the Brier
                # Skill Score (raw → calibrated) instead of a raw % change: BSS
                # is referenced to the climatology base rate, so it has a
                # meaningful zero and is safe to read as skill. For ECE/MCE,
                # which are bounded errors with a true zero, a plain % reduction
                # is fine. (We never percentage-ize Neg Log Loss — see the
                # per-fold diagnostic table below.)
                if key == "brier":
                    bss_raw = vals.get("original_bss")
                    bss_cal = vals.get("calibrated_bss")
                    if bss_raw is not None and bss_cal is not None:
                        rel_cell = f"BSS {bss_raw:.3f} &rarr; {bss_cal:.3f}"
                    else:
                        rel_cell = "—"
                else:
                    pct = (improvement / abs(original) * 100) if original else None
                    rel_cell = f"{pct:+.1f}%" if pct is not None else "—"
                metric_rows.append(
                    f"<tr><td>{label} &darr;</td>"
                    f"<td>{original:.4f}</td>"
                    f"<td>{calibrated:.4f}</td>"
                    f'<td class="{cls}">{arrow} {improvement:+.4f}</td>'
                    f'<td class="{cls}">{rel_cell}</td></tr>'
                )
            metrics_table = (
                '<table class="cal-table diag-table" border="0"><thead><tr>'
                "<th>Metric</th><th>Original</th><th>Calibrated</th>"
                "<th>Improvement</th><th>Skill / % Reduction</th>"
                f"</tr></thead><tbody>{''.join(metric_rows)}</tbody></table>"
            )

            # ── ROC-AUC / discrimination banner ───────────────────────────────
            # Monotonic calibrators (Platt, isotonic) preserve ranking, so a
            # near-zero raw-vs-calibrated AUC change means any accuracy/precision/
            # recall shift is purely a 0.5-threshold artifact, not a loss of
            # discrimination. When that holds, point the reader at the decision
            # threshold rather than implying the calibrated model is broken.
            auc_raw = vals.get("original_auc")
            auc_cal = vals.get("calibrated_auc")
            auc_note = ""
            if (
                auc_raw is not None
                and auc_cal is not None
                and not (pd.isna(auc_raw) or pd.isna(auc_cal))
            ):
                d_auc = auc_cal - auc_raw
                threshold_pointer = (
                    " See the Decision Threshold Diagnostic below."
                    if self.threshold_summary_ is not None
                    else " Re-tune the decision threshold rather than using 0.5."
                )
                if abs(d_auc) < 1e-6:
                    auc_note = (
                        f'<p class="label">ROC-AUC: {auc_raw:.4f} (raw) &rarr; '
                        f"{auc_cal:.4f} (calibrated), &Delta; &approx; 0. Calibration "
                        "preserved ranking, so any shift in Accuracy / Precision / "
                        "Recall / F1 is a 0.5-threshold artifact, not a change in "
                        f"discrimination.{threshold_pointer}</p>"
                    )
                else:
                    auc_note = (
                        f'<p class="label">ROC-AUC: {auc_raw:.4f} (raw) &rarr; '
                        f"{auc_cal:.4f} (calibrated), &Delta; = {d_auc:+.4f}.</p>"
                    )

            diagnostic_section = ""
            if self.calibration_scores_ is not None:
                # Pivot the flat raw_*/cal_* index into one row per metric so the
                # raw and calibrated columns sit side-by-side. The Δ column is
                # direction-aware (Brier and Neg Log Loss are lower-is-better) and
                # color-coded so improvement vs degradation reads at a glance.
                cs = self.calibration_scores_
                metric_display = {
                    "accuracy": "Accuracy",
                    "pwa": "PWA",
                    "neg_log_loss": "Neg Log Loss",
                    "brier": "Brier",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                }
                # Only Brier is a loss here; neg_log_loss is already negated
                # (sklearn convention) so, like the rest, higher is better.
                lower_better = {"brier"}

                diag_rows = []
                for metric, label in metric_display.items():
                    raw_key, cal_key = f"raw_{metric}", f"cal_{metric}"
                    if raw_key not in cs.index or cal_key not in cs.index:
                        continue
                    raw_mean, raw_std = cs.loc[raw_key, "mean"], cs.loc[raw_key, "std"]
                    cal_mean, cal_std = cs.loc[cal_key, "mean"], cs.loc[cal_key, "std"]
                    delta = cal_mean - raw_mean
                    if metric in lower_better:
                        improved = delta < 0
                        label = f"{label} ↓"  # mark lower-is-better
                    else:
                        improved = delta > 0
                    if delta == 0:
                        cls, arrow = "neutral", "—"
                    elif improved:
                        cls, arrow = "better", ("▲" if delta > 0 else "▼")
                    else:
                        cls, arrow = "worse", ("▲" if delta > 0 else "▼")
                    # Neg Log Loss lives in log space, where a percentage change
                    # is meaningless. Report the relative likelihood instead:
                    # exp(Δ) is how likely the calibrated model is to have
                    # produced the data relative to the raw model (sklearn's
                    # neg_log_loss sign convention makes this exactly exp(delta)).
                    delta_extra = ""
                    if metric == "neg_log_loss":
                        rel_likelihood = float(np.exp(delta))
                        delta_extra = (
                            f'<br><span class="label">{rel_likelihood:.2f}&times; likelihood</span>'
                        )
                    diag_rows.append(
                        f"<tr><td>{label}</td>"
                        f"<td>{raw_mean:.4f} &plusmn; {raw_std:.4f}</td>"
                        f"<td>{cal_mean:.4f} &plusmn; {cal_std:.4f}</td>"
                        f'<td class="{cls}">{arrow} {delta:+.4f}{delta_extra}</td></tr>'
                    )
                diag_table = (
                    '<table class="cal-table diag-table" border="0"><thead><tr>'
                    "<th>Metric</th><th>Raw (mean &plusmn; std)</th>"
                    "<th>Calibrated (mean &plusmn; std)</th><th>&Delta; (cal &minus; raw)</th>"
                    f"</tr></thead><tbody>{''.join(diag_rows)}</tbody></table>"
                )
                diagnostic_section = f"""
                    <div class="card">
                        <h2>Per-fold CV Diagnostic (raw vs calibrated)</h2>
                        <p class="label">Mean &plusmn; std across PurgedKFold folds from
                           analyze_calibrated_cross_val_scores. &Delta; is calibrated &minus; raw:
                           <span class="better">green</span> = calibration improved the metric,
                           <span class="worse">red</span> = degraded. Metrics marked &darr; are
                           lower-is-better. For Neg Log Loss the relative likelihood
                           exp(&Delta;) is shown &mdash; how likely the calibrated model is to
                           have produced the data versus the raw model (e.g. 0.91&times; means
                           ~9% less likely).</p>
                        {diag_table}
                    </div>"""

            # ── Threshold diagnostic section ──────────────────────────────────
            threshold_section = ""
            if self.threshold_summary_ is not None and self.threshold_diagnostic_ is not None:
                summary = self.threshold_summary_
                threshold_table = pd.DataFrame(
                    {
                        "default_0_5": {
                            "threshold": summary["default_threshold"],
                            summary["objective"]: summary["default_score"],
                            "positive_rate": summary["default_positive_rate"],
                            "n_positive": summary["default_n_positive"],
                        },
                        "selected": {
                            "threshold": summary["selected_threshold"],
                            summary["objective"]: summary["selected_score"],
                            "positive_rate": summary["selected_positive_rate"],
                            "n_positive": summary["selected_n_positive"],
                        },
                    }
                ).to_html(classes="cal-table", border=0, float_format=lambda v: f"{v:.4f}")

                threshold_section = f"""
                    <div class="card">
                        <h2>Decision Threshold Diagnostic</h2>
                        <p class="label">Threshold selected from calibrated OOF probabilities.
                           Objective: {summary["objective"]}. This affects hard labels and
                           meta-labeling side generation, not probability calibration itself.</p>
                        {threshold_table}
                    </div>"""

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Calibration Report - {self._report_title_suffix()}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f172a; color: #f1f5f9; padding: 40px;
                line-height: 1.6; }}
        .container {{ max-width: 1100px; margin: auto; }}
        h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
        h2 {{ color: #7dd3fc; margin-top: 8px; }}
        .card {{ background-color: #1e293b; border-radius: 12px; padding: 24px;
                 margin-bottom: 24px; border: 1px solid #334155; }}
        table.cal-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        table.cal-table th, table.cal-table td {{ text-align: left; padding: 10px;
                 border-bottom: 1px solid #334155; }}
        table.cal-table th {{ color: #94a3b8; font-weight: 600; }}
        .label {{ color: #94a3b8; font-size: 0.9rem; }}
        .better {{ color: #4ade80; font-weight: 600; }}
        .worse {{ color: #f87171; font-weight: 600; }}
        .neutral {{ color: #94a3b8; }}
        table.diag-table td:nth-child(n+2),
        table.diag-table th:nth-child(n+2) {{ text-align: right;
                 font-variant-numeric: tabular-nums; }}
        img {{ width: 100%; max-width: 1100px; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Calibration Report: {self._report_subtitle()}</h1>
        <p class="label">Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
           &nbsp;|&nbsp; Method: {self.calibrator_.method}
           &nbsp;|&nbsp; OOF samples: {len(y_true)}</p>

        <div class="card">
            <h2>Reliability Diagram</h2>
            <p class="label">Predicted probability vs observed frequency with bootstrap
               95% confidence intervals. Points on the dashed diagonal are perfectly
               calibrated.</p>
            <img src="data:image/png;base64,{img_b64}" alt="Reliability diagram">
        </div>

        <div class="card">
            <h2>Calibration Metrics</h2>
            <p class="label">Lower Brier, Expected Calibration Error (ECE) and
               Maximum Calibration Error (MCE) indicate better calibration (all
               marked &darr;). Improvement is original &minus; calibrated:
               <span class="better">green &#9660;</span> = calibration reduced the
               error, <span class="worse">red &#9650;</span> = increased it. The last
               column reports the Brier Skill Score (higher = more skill vs the
               base-rate climatology) for Brier, and the % error reduction for ECE/MCE.</p>
            {metrics_table}
            {auc_note}
        </div>
        {threshold_section}
        {diagnostic_section}
    </div>
</body>
</html>"""

            report_path = self.file_paths["reports"] / "calibration_report.html"
            report_path.write_text(html_content, encoding="utf-8")
            self.logger.info(f"Generated calibration report: {report_path}")

        except Exception as e:
            self.logger.warning(f"Calibration report generation failed: {e}")

    def _display_cache_reports(self):
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE REPORT")
        print("=" * 70)
        monitor = get_cache_monitor()
        if monitor:
            monitor.print_report()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def check_contamination(self):
        print("\n" + "=" * 70)
        print("DATA CONTAMINATION CHECK")
        print("=" * 70)
        print_contamination_report()

    def get_data_summary(self) -> pd.DataFrame:
        summary_data = []
        components = [
            ("bar_data", self.bar_data),
            ("features", self.features),
            ("preprocessed_features", self.preprocessed_features),
            ("events", self.events),
            ("meta_features", self.meta_features),
            ("sample_weight", self.sample_weight),
        ]
        for name, data in components:
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    shape, dtype, columns = data.shape, "DataFrame", f"{len(data.columns)} cols"
                elif isinstance(data, pd.Series):
                    shape, dtype, columns = (len(data),), "Series", "N/A"
                else:
                    shape, dtype, columns = "N/A", type(data).__name__, "N/A"
                summary_data.append(
                    {
                        "Component": name,
                        "Type": dtype,
                        "Rows": shape[0] if isinstance(shape, tuple) else shape,
                        "Columns": (
                            shape[1] if isinstance(shape, tuple) and len(shape) > 1 else columns
                        ),
                        "Memory (MB)": (
                            data.memory_usage(deep=True).sum() / (1024**2)
                            if hasattr(data, "memory_usage")
                            else "N/A"
                        ),
                    }
                )
        return pd.DataFrame(summary_data)


# ============================================================================
# Helpers
# ============================================================================


def get_model_type(model) -> str:
    types = {
        "RandomForestClassifier": "rf",
        "SequentiallyBootstrappedBaggingClassifier": "seq_rf",
        "DecisionTreeClassifier": "dt",
    }
    name = type(model).__name__
    return types.get(name, name.replace("Classifier", "").lower())


def is_tree(estimator) -> bool:
    return isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier))


@cacheable(time_aware=True)
def apply_seq_bootstrap(
    X,
    y,
    estimator,
    n_estimators,
    max_samples,
    max_features,
    samples_info_sets,
    price_bars_index,
    random_state,
    sample_weight,
):
    bag = SequentiallyBootstrappedBaggingClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        samples_info_sets=samples_info_sets,
        price_bars_index=price_bars_index,
        random_state=random_state,
    )

    if sample_weight is not None:
        bag.fit(X, y, sample_weight=sample_weight)
    else:
        bag.fit(X, y)

    return bag
