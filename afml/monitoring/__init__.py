"""
afml.monitoring
===============

Post-deployment health detection. Two detectors that see different failures:
a sequential probability ratio test on the win rate, and a permutation trend test
on granular profit factor.

The design point: a fixed threshold on the incomplete-beta posterior, recomputed
after every trade, is a sequential test whose false-alarm rate is not the one it
appears to promise. :class:`WinRateSPRT` in reflecting mode is the trigger, with
its boundary set by :func:`calibrate_boundary` for the planned horizon;
:func:`deterioration_probability` is for reporting.
"""

from .deterioration import (
    SPRTState,
    WinRateSPRT,
    average_run_length,
    calibrate_boundary,
    cusum_run_lengths,
    deterioration_probability,
    expected_sample_size,
    false_alarm_probability,
)
from .monitor import LiveStrategyMonitor, MonitorStatus
from .profit_factor import (
    TrendResult,
    granular_profit_factor,
    profit_factor,
    profit_factor_trend,
)

__all__ = [
    "LiveStrategyMonitor",
    "MonitorStatus",
    "SPRTState",
    "TrendResult",
    "WinRateSPRT",
    "average_run_length",
    "calibrate_boundary",
    "cusum_run_lengths",
    "deterioration_probability",
    "expected_sample_size",
    "false_alarm_probability",
    "granular_profit_factor",
    "profit_factor",
    "profit_factor_trend",
]
