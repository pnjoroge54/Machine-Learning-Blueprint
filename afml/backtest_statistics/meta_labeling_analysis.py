"""
Comprehensive analysis and visualization suite for meta-labeling performance evaluation.
This module provides detailed analysis tools including statistical tests, visual comparisons,
and reporting functionality.
"""

from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import classification_report
from tqdm import tqdm

from afml.cache.unified_cache_system import cacheable
from afml.cross_validation.combinatorial import CombinatorialPurgedCV
from afml.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from afml.labeling.triple_barrier import add_vertical_barrier, triple_barrier_labels
from afml.sample_weights.optimized_attribution import get_weights_by_time_decay_optimized
from afml.strategies.trading_strategies import BaseStrategy
from afml.util.volatility import get_daily_vol

from ..bet_sizing.bet_sizing import bet_size_budget, bet_size_probability, bet_size_reserve
from ..cross_validation.cross_validation import PurgedSplit
from ..production.model_development import (
    calculate_rolling_metrics,
    create_feature_engineering_pipeline,
    generate_events_triple_barrier,
    load_and_prepare_training_data,
)
from .performance_analysis import calculate_performance_metrics


def evaluate_meta_labeling_performance(
    events: pd.DataFrame,
    meta_probabilities: pd.Series,
    close: pd.Series,
    confidence_threshold: float = 0.5,
    trading_days_per_year: int = 252,
    trading_hours_per_day: int = 24,
    strategy_name: str = "Strategy",
    bet_sizing: str = None,
    **kwargs,
) -> dict:
    """
    Evaluates and compares the performance of a primary strategy against a
    meta-labeled version of that strategy.

    This function simulates two strategies:
    1.  The primary strategy, which takes all signals.
    2.  The meta-labeled strategy, which filters trades based on a confidence
        threshold and sizes them according to the meta-model's probability.

    Args:
        events: A DataFrame of trade events that contains at least the columns 't1' and 'side'.
            - index: Event start times
            - t1: Event end times, i.e., the time of first barrier touch
            - side: Trade direction
        meta_probabilities: A Series or array of probabilities from the meta-model.
        close: A Series of prices that cover the period encapsulated in events.
        confidence_threshold: The minimum probability required to take a trade.
        trading_days_per_year: The number of trading days in a year.
        trading_hours_per_day: The number of trading hours per day.
        strategy_name: The name of the strategy for reporting.
        bet_sizing: One of None, "budget", "probability", "reserve", "dynamic"
        kwargs: Bet-sizing arguments for "reserve" method that do not relate to events data.
            Expected keys:
            - fit_runs : int
                Number of runs to execute when trying to fit the distribution.
            - epsilon : float
                Error tolerance.
            - factor : int
                Lambda factor from equations.
            - variant : int
                 The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using first 5 moments.
            - max_iter : int
                Maximum number of iterations after which to terminate loop.
            - num_workers : int
                Number of CPU cores to use for multiprocessing execution, set to -1 to use all CPU cores. Default is 1.
            - return_parameters : bool
                 If True, function also returns a dictionary of the fited mixture parameters.

    Returns:
        A dictionary containing the performance metrics for both strategies,
        their return series, and other comparison metadata.
    """
    # Calculate base returns (price changes without side)
    events = events.dropna(subset=["t1"])
    t1 = events["t1"]
    side = events["side"]
    all_dates = events.index.union(other=t1.array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # Base returns (price movements)
    base_returns = prices.loc[t1.array].array / prices.loc[events.index] - 1

    # Primary strategy: apply side to get directional returns
    primary_returns = pd.Series(base_returns * side.values, index=events.index)
    data_index = close.loc[: t1.iloc[-1]].index

    # Filter trades based on confidence threshold
    aligned_probs = meta_probabilities.reindex(events.index, fill_value=0.5)
    confident_trades = aligned_probs > confidence_threshold
    meta_prob = aligned_probs[confident_trades]
    meta_events = events[confident_trades]
    meta_side = meta_events["side"]
    meta_t1 = meta_events["t1"]

    # --- Bet Sizing Logic ---
    if bet_sizing is None:
        bets = meta_side.copy()
        bet_sizing = "fixed"
    elif bet_sizing == "probability":
        bets = bet_size_probability(
            meta_events, meta_prob, num_classes=2, pred=meta_side, **kwargs
        )
    elif bet_sizing == "budget":
        result = bet_size_budget(meta_t1, meta_side)
        bets = result["bet_size"]
    elif bet_sizing == "reserve":
        result = bet_size_reserve(meta_t1, meta_side, **kwargs)
        bets = result["bet_size"]
    else:
        raise ValueError(f"Unknown bet_sizing method: {bet_sizing}")

    msg = f"Bet Sizing Method: {bet_sizing.title()} | Confidence Threshold: {confidence_threshold}"
    msg = msg + f"\n{kwargs}" if kwargs else msg
    logger.info(msg)

    # Apply bet sizes to base returns for filtered trades
    meta_base_returns = base_returns[confident_trades]

    meta_returns = (meta_base_returns * bets).dropna()

    # --- Performance Calculation ---
    # Don't pass positions - calculate trade stats separately for events
    primary_metrics = calculate_performance_metrics(
        primary_returns,
        data_index,
        positions=None,  # Don't pass positions for event-based data
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )

    meta_metrics = calculate_performance_metrics(
        meta_returns,
        data_index,
        positions=None,  # Don't pass positions for event-based data
        trading_days_per_year=trading_days_per_year,
        trading_hours_per_day=trading_hours_per_day,
    )

    # --- Add Event-Specific Metrics Manually ---
    # Calculate trade duration directly from events
    primary_metrics["avg_trade_duration"] = (events["t1"] - events.index).mean().round("1s")
    meta_metrics["avg_trade_duration"] = (
        (meta_events["t1"] - meta_events.index).mean().round("1s") if not meta_returns.empty else 0
    )

    # Calculate bet frequency
    primary_metrics["bet_frequency"] = len(events)
    meta_metrics["bet_frequency"] = len(meta_events)

    total_periods = len(data_index)
    periods_per_year = trading_days_per_year  # Simplified

    primary_metrics["bets_per_year"] = int(
        len(events) * (periods_per_year / total_periods) if total_periods > 0 else 0
    )
    meta_metrics["bets_per_year"] = int(
        len(meta_events) * (periods_per_year / total_periods)
        if total_periods > 0
        else 0
    )

    # --- Meta-Specific Metrics ---
    total_signals = len(events)
    filtered_signals = len(meta_events)

    meta_metrics["signal_filter_rate"] = (
        1 - (filtered_signals / total_signals) if total_signals > 0 else 0
    )

    if len(meta_returns) > 1:
        duration_days = (meta_returns.index[-1] - meta_returns.index[0]).days
        actual_trades_per_year = (
            len(meta_returns) * (365.25 / duration_days)
            if duration_days > 0
            else len(meta_returns)
        )
    else:
        actual_trades_per_year = 1

    meta_metrics["actual_trades_per_year"] = int(actual_trades_per_year)

    return {
        "strategy_name": strategy_name,
        "primary_metrics": primary_metrics,
        "meta_metrics": meta_metrics,
        "primary_returns": primary_returns,
        "meta_returns": meta_returns,
        "total_primary_signals": total_signals,
        "filtered_signals": filtered_signals,
        "bet_sizing": bet_sizing,
        "confidence_threshold": confidence_threshold,
    }


def calculate_risk_adjusted_metrics(results: dict, threshold: float) -> pd.DataFrame:
    """
    Calculate advanced risk-adjusted performance metrics.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance

    Returns:
        DataFrame with risk-adjusted metrics
    """
    primary = results["primary_metrics"]
    meta = results["meta_metrics"]

    def omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio (probability weighted ratio of gains vs losses)"""
        if returns.empty:
            return 0

        # Calculate excess returns over threshold
        excess_returns = returns - threshold

        # Split into gains and losses
        gains = excess_returns[excess_returns > 0]
        losses = excess_returns[excess_returns < 0]

        # Omega ratio = (Probability-weighted gains) / (Probability-weighted losses)
        expected_gains = gains.sum() / len(returns)  # Average gain per period
        expected_losses = abs(losses.sum()) / len(returns)  # Average loss per period (absolute)

        return expected_gains / expected_losses if expected_losses != 0 else np.inf

    def tail_ratio(returns: pd.Series) -> float:
        """Ratio of 95th percentile to 5th percentile"""
        if returns.empty:
            return 0
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else np.inf

    primary_returns = results["primary_returns"]
    meta_returns = results["meta_returns"]

    metrics = pd.DataFrame(
        {
            "Primary": [
                omega_ratio(primary_returns, threshold),
                tail_ratio(primary_returns),
                (
                    primary["sharpe_ratio"] / primary["max_drawdown"]
                    if primary["max_drawdown"] > 0
                    else 0
                ),
                (
                    primary["win_rate"] * primary["avg_win"] / abs(primary["avg_loss"])
                    if primary["avg_loss"] != 0
                    else 0
                ),
                (primary["expectancy"] / primary["volatility"] if primary["volatility"] > 0 else 0),
            ],
            "Meta": [
                omega_ratio(meta_returns, threshold),
                tail_ratio(meta_returns),
                (meta["sharpe_ratio"] / meta["max_drawdown"] if meta["max_drawdown"] > 0 else 0),
                (
                    meta["win_rate"] * meta["avg_win"] / abs(meta["avg_loss"])
                    if meta["avg_loss"] != 0
                    else 0
                ),
                (meta["expectancy"] / meta["volatility"] if meta["volatility"] > 0 else 0),
            ],
        },
        index=[
            "Omega Ratio",
            "Tail Ratio",
            "Sharpe/MaxDD",
            "Win-Loss Efficiency",
            "Expectancy/Vol",
        ],
    )

    metrics["Better"] = metrics.apply(
        lambda row: "Meta" if row["Meta"] > row["Primary"] else "Primary", axis=1
    )

    return metrics


def analyze_signal_quality(results: dict) -> Dict:
    """
    Analyze the quality and distribution of filtered signals, with both actual and normalized sizing.

    Args:
        results: Dictionary from evaluate_meta_labeling_performance

    Returns:
        Dictionary with signal quality metrics
    """
    primary_returns = results["primary_returns"]
    meta_returns = results["meta_returns"]

    # Align returns for comparison (same timestamps)
    common_idx = primary_returns.index.intersection(meta_returns.index)
    primary_common = primary_returns.loc[common_idx]
    meta_common = meta_returns.loc[common_idx]

    # Filtered signals = those rejected by meta-labeling
    filtered_mask = ~primary_returns.index.isin(meta_returns.index)
    filtered_returns = primary_returns[filtered_mask]

    # --- Core counts ---
    total_signals = len(primary_returns)
    accepted_signals = len(meta_returns)
    rejected_signals = len(filtered_returns)

    # --- Precision/Recall framing ---
    precision = (meta_returns > 0).mean() if accepted_signals > 0 else 0
    total_winners = (primary_returns > 0).sum()
    recall = ((meta_returns > 0).sum() / total_winners) if total_winners > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # --- Quality metrics ---
    analysis = {
        "total_signals": total_signals,
        "accepted_signals": accepted_signals,
        "rejected_signals": rejected_signals,
        "filter_rate": rejected_signals / total_signals if total_signals > 0 else 0,
        # Accepted signals quality
        "accepted_win_rate": precision,
        "accepted_avg_return": meta_returns.mean() if accepted_signals > 0 else 0,
        "accepted_sharpe": (
            meta_returns.mean() / meta_returns.std() * np.sqrt(252)
            if accepted_signals > 1 and meta_returns.std() > 0
            else 0
        ),
        # Rejected signals quality
        "rejected_win_rate": ((filtered_returns > 0).mean() if rejected_signals > 0 else 0),
        "rejected_avg_return": filtered_returns.mean() if rejected_signals > 0 else 0,
        "rejected_sharpe": (
            filtered_returns.mean() / filtered_returns.std() * np.sqrt(252)
            if rejected_signals > 1 and filtered_returns.std() > 0
            else 0
        ),
        # Filter effectiveness
        "avoided_losses": (
            (filtered_returns < 0).sum() / rejected_signals if rejected_signals > 0 else 0
        ),
        # Classification-style framing
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    # --- Statistical test: accepted vs rejected ---
    if rejected_signals > 0 and accepted_signals > 0:
        t_stat, p_value = stats.ttest_ind(meta_returns, filtered_returns, equal_var=False)
        analysis["ttest_pvalue"] = p_value
        analysis["significantly_better"] = p_value < 0.05
    else:
        analysis["ttest_pvalue"] = np.nan
        analysis["significantly_better"] = False

    # --- Comparative stats on aligned signals ---
    if len(common_idx) > 0:
        # Actual sizing
        analysis["aligned_primary_mean_actual"] = primary_common.mean()
        analysis["aligned_meta_mean_actual"] = meta_common.mean()
        analysis["aligned_diff_actual"] = meta_common.mean() - primary_common.mean()

        # Equal sizing normalization (force same notional per trade)
        primary_equal = (
            primary_common / primary_common.abs().mean()
            if primary_common.abs().mean() != 0
            else primary_common
        )
        meta_equal = (
            meta_common / meta_common.abs().mean() if meta_common.abs().mean() != 0 else meta_common
        )

        analysis["aligned_primary_mean_equal"] = primary_equal.mean()
        analysis["aligned_meta_mean_equal"] = meta_equal.mean()
        analysis["aligned_diff_equal"] = meta_equal.mean() - primary_equal.mean()

    return analysis


def generate_test_events_triple_barrier(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    target_lookback: int,
    profit_target: float = 1,
    stop_loss: float = 1,
    max_holding_period: Dict[str, int] = dict(num_bars=100),
    min_ret: float = 0.0,
    vertical_barrier_zero: bool = True,
) -> pd.DataFrame:
    """
    Generate trading events using the triple-barrier method.

    Parameters
    ----------
    data : pd.DataFrame
        Price bars with 'close' column.
    strategy : BaseStrategy
        Strategy instance implementing `generate_signals()`.
    target_lookback : int
        Lookback window for volatility estimation.
    profit_target : float, default=1
        Profit-taking threshold multiplier.
    stop_loss : float, default=1
        Stop-loss threshold multiplier.
    max_holding_period : dict, default={'num_bars': 100}
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
        - 'bin' : {-1, 0, 1} classification
        - 't1'  : vertical barrier timestamps
        - 'w'   : sample weights
        - 'tW'  : uniqueness weights

    Notes
    -----
    - Prevents data leakage via time-aware caching.
    """
    # Compute barriers
    close = data["close"]
    target = get_daily_vol(close, target_lookback)
    side = strategy.generate_signals(data)
    t_events = side[side != 0].index
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
    return events


@cacheable()
def get_validation_metrics(
    test_start: Union[str, pd.Timestamp, datetime],
    test_end: Union[str, pd.Timestamp, datetime],
    strategy: BaseStrategy,
    model: BaseEstimator,
    config: dict,
    target_config: dict,
    feature_config: dict,
    feature_names: list,
    bet_sizing: str = None,
    confidence_threshold: float = 0.5,
    **kwargs,
) -> dict:
    if config["bar_type"] == "tick":
        bar_size = int(config["tick_bar_size"])
    else:
        bar_size = config["bar_size"]

    df = load_and_prepare_training_data(
        symbol=config["symbol"],
        start_date=test_start,
        end_date=test_end,
        account_name=config["account_name"],
        bar_type=config["bar_type"],
        bar_size=bar_size,
        price=config["price"],
    )

    try:
        on_crossover = strategy.on_crossover()
    except Exception:
        on_crossover = True

    events = generate_events_triple_barrier(
        df,
        strategy=strategy,
        target_config=target_config,
        profit_target=config["profit_target"],
        stop_loss=config["stop_loss"],
        min_ret=0,
        max_holding_period=config["max_holding_period"],
        vertical_barrier_zero=False,
        filter_as_series=None,
        on_crossover=on_crossover,
    )

    data_config = {
        "account_name": config["account_name"],
        "bar_type": config["bar_type"],
        "bar_size": config["bar_size"],
        "price": config["price"],
    }
    features = create_feature_engineering_pipeline(df, feature_config, data_config)
    sample_weight = pd.Series(np.ones(len(events)), index=events.index)
    meta_features = calculate_rolling_metrics(events, sample_weight)
    features = features.join(meta_features).dropna()

    events = events.loc[features.index]

    X = features[feature_names]
    y = events["bin"]

    validate, test = PurgedSplit(events["t1"], test_size_pct=0.5).split(X)
    X_val, X_test = X.iloc[validate], X.iloc[test]
    y_val, y_test = y.iloc[validate], y.iloc[test]
    events_val = events.iloc[validate]
    df_val = df.loc[: events_val.t1[-1]]

    prob = pd.Series(model.predict_proba(X_val)[:, 1], index=X_val.index, name="prob")
    pred = (prob > confidence_threshold).astype(int)

    validation_metrics = evaluate_meta_labeling_performance(
        events=events_val,
        meta_probabilities=prob,
        close=df_val["close"],
        confidence_threshold=confidence_threshold,
        strategy_name=config["strategy"],
        bet_sizing=bet_sizing,
        **kwargs,
    )

    validation_metrics.update(
        dict(
            symbol=config["symbol"],
            bar_type=config["bar_type"],
            bar_size=bar_size,
            min_ret=config["min_ret"],
            strategy_config=dict(
                strategy=config["strategy"],
                account_name=config["account_name"],
                symbol=config["symbol"],
                bar_type=config["bar_type"],
                bar_size=bar_size,
                price=config["price"],
                profit_target=config["profit_target"],
                stop_loss=config["stop_loss"],
                max_holding_period=config["max_holding_period"],
                min_ret=config["min_ret"],
                feature_func=feature_config["func"].__name__,
                feature_params=feature_config["params"],
                target_func=target_config["func"].__name__,
                target_params=target_config["params"],
                bet_sizing=validation_metrics["bet_sizing"].title(),
                confidence_threshold=confidence_threshold,
            ),
            classification_report=classification_report(y_val, pred),
            data=df,
            X_test=X_test,
            y_test=y_test,
            events_test=events.iloc[test],
        )
    )

    logger.info(f"{X_test.index[0]} - {X_test.index[-1]} held out for final testing")
    return validation_metrics


# noinspection PyPep8Naming
@cacheable()
def meta_labeling_cpcv_analysis(
    test_start: Union[str, pd.Timestamp, datetime],
    test_end: Union[str, pd.Timestamp, datetime],
    strategy: BaseStrategy,
    classifier: ClassifierMixin,
    n_splits: int,
    config: dict,
    target_config: dict,
    feature_config: dict,
    feature_names: list,
    weighting_scheme: str,
    bet_sizing: str = None,
    confidence_threshold: float = 0.5,
    **kwargs,
):
    # pylint: disable=invalid-name
    # pylint: disable=comparison-with-callable
    """
    Run purged/embargoed cross-validation for a classifier and return per-fold scores.

    This implements the evaluation pattern from López de Prado (Advances in Financial Machine Learning,
    snippet 7.4) but requires the caller to provide a CV generator (e.g., PurgedKFold).

    Behavior summary
    - Trains the provided classifier on each train split and scores on the corresponding test split.
    - Supports passing separate sample weights for training and scoring.
    - Special-cases `SequentiallyBootstrappedBaggingClassifier`: clones the classifier per fold and
      aligns its samples_info_sets with the train indices; disables internal OOB scoring during CV.
    - Accepts `scoring` as either a string key (mapped to a function) or a callable metric. For
      probability-based scorers (log_loss, probability_weighted_accuracy) the function expects
      probability inputs from `predict_proba`. For label-based scorers the function expects discrete
      predictions from `predict`.

    Parameters
    ----------
    classifier : ClassifierMixin
        A scikit-learn compatible classifier instance (must implement fit/predict and optionally
        predict_proba).
    X : pd.DataFrame
        Feature matrix indexed consistently with y and (for SequentiallyBootstrappedBaggingClassifier)
        with classifier.samples_info_sets.
    y : pd.Series
        Target labels aligned with X (index used to align samples_info_sets when required).
    events : pd.DataFrame
        Triple-barrier events
    cv_gen : BaseCrossValidator
        Cross-validation generator instance with a split(X, y) method (e.g., PurgedKFold).
    sample_weight : Array-like, optional (default=None)
        Per-sample weights used when calling classifier.fit on the train split. If None, all ones
        are used (no weighting).
    sample_weight_score : Array-like, optional (default=None)
        Per-sample weights used when calling the scoring function on the test split. If None, all ones
        are used.
    scoring : str or callable, optional (default=log_loss)
        - If a string, one of the supported keys: "neg_log_loss", "accuracy", "f1", "pwa".
          "neg_log_loss" maps to sklearn.metrics.log_loss and is returned as positive (the function
          multiplies log_loss by -1 to make larger-is-better consistent with other scorers).
        - If a callable, signature should be compatible with either:
            scorer(y_true, y_pred, sample_weight=None, labels=...)   # label-based or prob-based
          The code attempts to pass `labels=classifier.classes_` where relevant, and falls back if
          the scorer does not accept that argument.
        - For probability scorers (log_loss, probability_weighted_accuracy) the function is called
          with `predict_proba` output; for label-based scorers the function is called with `predict`.
        The default is `log_loss`.

    Returns
    -------
    np.ndarray
        1-D array of per-fold scores (float). Order corresponds to the order of splits returned by
        cv_gen.split(X, y).

    Raises
    ------
    KeyError
        If SequentiallyBootstrappedBaggingClassifier is used and its samples_info_sets are not aligned
        with y (index mismatch).
    TypeError / RuntimeError
        If the provided `scoring` callable raises on the provided inputs; the function attempts a
        robust call pattern but will propagate unexpected exceptions.

    Notes
    -----
    - For classifiers that require average/probability inputs (e.g., AUC), pass an appropriate
      scoring callable that accepts probability-like inputs and set scoring to that callable or the
      corresponding string key.
    - For Seq-Bagging classifiers the function disables the estimator's internal OOB scoring during
      cross-validation to avoid interference with the CV scoring flow.
    """
    if config["bar_type"] == "tick":
        bar_size = int(config["tick_bar_size"])
    else:
        bar_size = config["bar_size"]

    df = load_and_prepare_training_data(
        symbol=config["symbol"],
        start_date=test_start,
        end_date=test_end,
        account_name=config["account_name"],
        bar_type=config["bar_type"],
        bar_size=bar_size,
        price=config["price"],
    )
    events = generate_events_triple_barrier(
        df,
        strategy,
        target_config=target_config,
        profit_target=config["profit_target"],
        stop_loss=config["stop_loss"],
        max_holding_period=config["max_holding_period"],
        min_ret=config["min_ret"],
        vertical_barrier_zero=False,
        filter_as_series=None,
    )
    sample_weight = pd.Series(np.ones(len(events)), index=events.index)
    meta_features = calculate_rolling_metrics(events, sample_weight)

    data_config = {
        "account_name": config["account_name"],
        "bar_type": config["bar_type"],
        "bar_size": config["bar_size"],
        "price": config["price"],
    }
    features = create_feature_engineering_pipeline(df, feature_config, data_config)
    features = features.join(meta_features).dropna()

    cont = events.loc[features.index]
    X = features[feature_names]
    y = cont["bin"]

    if weighting_scheme.startswith("uniqueness"):
        sample_weight = cont["tW"]
    elif weighting_scheme.startswith("return"):
        sample_weight = cont["w"]
    else:
        sample_weight = np.ones((X.shape[0],))

    try:
        _, linear, decay = weighting_scheme.split("_")
        decay_vec = get_weights_by_time_decay_optimized(
            triple_barrier_events=cont,
            close_index=df.index,
            last_weight=decay,
            linear=(1 if linear == "linear" else 0),
            av_uniqueness=cont["tW"],
        )
        sample_weight *= decay_vec
    except Exception:
        pass

    classifier = clone(classifier)

    # Check for sequential bootstrap
    seq_bootstrap = isinstance(classifier, SequentiallyBootstrappedBaggingClassifier)
    if seq_bootstrap:
        t1 = classifier.samples_info_sets.copy()

    metrics = []

    # Score model on KFolds
    cv_gen = CombinatorialPurgedCV(n_splits, n_test_splits=2, samples_info_sets=cont["t1"])
    for train, test in tqdm(
        cv_gen.split(X=X, y=y), desc="CPCV splits", total=cv_gen.n_combinations
    ):  # noqa: F821
        if seq_bootstrap:
            classifier = classifier.set_params(
                samples_info_sets=t1.iloc[train], oob_score=False
            )  # Create new instance
        fit = classifier.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train],
        )

        X_test, y_test = X.iloc[test], y.iloc[test]
        prob = pd.Series(fit.predict_proba(X_test)[:, 1], index=X_test.index, name="prob")
        pred = (prob > confidence_threshold).astype(int)

        validation_metrics = evaluate_meta_labeling_performance(
            events=cont.iloc[test],
            meta_probabilities=prob,
            close=df["close"],
            confidence_threshold=confidence_threshold,
            strategy_name=config["strategy"],
            bet_sizing=bet_sizing,
            **kwargs,
        )

        validation_metrics.update(
            dict(
                symbol=config["symbol"],
                bar_size=bar_size,
                bar_type=config["bar_type"],
                bet_sizing=bet_sizing,
                classification_report=classification_report(y.iloc[train], pred),
                data=df,
                X_test=X_test,
                y_test=y_test,
                events_test=cont.iloc[test],
            )
        )

        metrics.append(validation_metrics)

    return metrics


# Example usage
if __name__ == "__main__":
    # Example with simulated data
    from datetime import datetime

    # Create sample results for different bar types
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    sample_results = {}

    for bar_type in ["tick", "time", "dollar"]:
        # Simulate returns
        n_trades = np.random.randint(50, 200)
        trade_dates = np.random.choice(dates, n_trades, replace=False)
        trade_dates.sort()

        primary_returns = pd.Series(np.random.normal(0.001, 0.02, n_trades), index=trade_dates)

        meta_returns = pd.Series(
            np.random.normal(0.0015, 0.015, n_trades - 20),
            index=trade_dates[: n_trades - 20],
        )

        # Simulate metrics
        primary_metrics = {
            "total_return": 0.15,
            "annualized_return": 0.18,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 1.8,
            "max_drawdown": -0.12,
            "volatility": 0.15,
            "win_rate": 0.52,
            "profit_factor": 1.4,
            "num_trades": n_trades,
            "avg_trade_duration": pd.Timedelta(days=3),
            "expectancy": 0.005,
            "kelly_criterion": 0.12,
            "best_trade": 0.08,
            "worst_trade": -0.06,
            "consecutive_wins": 5,
            "consecutive_losses": 3,
        }

        meta_metrics = {
            "total_return": 0.18,
            "annualized_return": 0.22,
            "sharpe_ratio": 1.5 + np.random.uniform(-0.2, 0.2),
            "sortino_ratio": 1.8,
            "calmar_ratio": 2.1,
            "max_drawdown": -0.09,
            "volatility": 0.12,
            "win_rate": 0.58,
            "profit_factor": 1.8,
            "num_trades": n_trades - 20,
            "avg_trade_duration": pd.Timedelta(days=4),
            "signal_filter_rate": 0.25,
            "confidence_threshold": 0.6,
            "expectancy": 0.008,
            "kelly_criterion": 0.15,
            "best_trade": 0.09,
            "worst_trade": -0.05,
            "consecutive_wins": 6,
            "consecutive_losses": 2,
            "trades_per_year": int((n_trades - 20) * 1.2),  # Added this key
        }

        sample_results[bar_type] = {
            "strategy_name": f"Bollinger_{bar_type}",  # This key is REQUIRED
            "primary_metrics": primary_metrics,
            "meta_metrics": meta_metrics,
            "primary_returns": primary_returns,
            "meta_returns": meta_returns,
            "total_primary_signals": n_trades,
            "filtered_signals": n_trades - 20,
        }
