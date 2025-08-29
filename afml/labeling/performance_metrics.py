import pandas as pd
from loguru import logger

from ..backtest_statistics.performance_analysis import calculate_performance_metrics


def calculate_label_metrics(
    data_index: pd.DataFrame,
    side_prediction: pd.Series,
    events: pd.DataFrame,
    trading_hours_per_day: float = 24,
):
    """
    Calculate performance metrics for a labeling strategy.
    Parameters
    ----------
    data_index : pd.DateTimeIndex
        DateTime index of data.
    side_prediction : pd.Series
        Series containing the side predictions (1 for long, -1 for short, 0 for neutral) with a DateTime index.
    events : pd.DataFrame
        DataFrame containing the events with a DateTime index and a 'ret' column for returns.
    trading_hours_per_day : float, optional
        Number of trading hours per day, by default 24.
    Returns
    -------
    pd.Series
        Series containing the calculated performance metrics.
    """
    if events.empty:
        return {}

    metrics = calculate_performance_metrics(
        returns=events["ret"],
        data_index=data_index,
        positions=side_prediction.reindex(data_index).fillna(0),
        trading_hours_per_day=trading_hours_per_day,
    )

    if metrics["avg_trade_duration"] == pd.Timedelta(0):
        logger.info("Calculating avg_trade_duration using (events['t1'] - events.index).mean()")
        metrics["avg_trade_duration"] = (events["t1"] - events.index).mean().round("1s")

    metrics = pd.Series(metrics, name="trade_metrics")
    return metrics
