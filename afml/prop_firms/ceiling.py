"""Size-aware replacement for afml.prop_firm.sizing.prop_firm_sl_ceiling."""

from __future__ import annotations


def prop_firm_sl_ceiling(
    account_usd: float = 25_000.0,
    daily_loss_pct: float = 0.05,
    lot_size: float = 0.5,
    n_concurrent: int = 2,
    pip_value_per_lot: float = 10.0,
    bet_size: float = 1.0,
    aggregate_exposure: bool = False,
) -> float:
    """
    Maximum stop-loss per trade in pips under a prop-firm daily loss cap.

    Parameters
    ----------
    lot_size : float
        Notional at FULL signal strength (|final_size| = 1), not the lot
        actually sent to the broker.
    bet_size : float
        PropFirmAwareSizer output for this trade, in [-1, 1]. The executed
        notional is lot_size * |bet_size|; the ceiling scales as 1/|bet_size|.
    aggregate_exposure : bool
        True  -> avg_active_signals semantics: concurrent bets are averaged
                 into one net position, so exposure never exceeds
                 lot_size * |bet_size| and n_concurrent does NOT divide.
        False -> each bet held at its own notional; total exposure is
                 n_concurrent * lot_size * |bet_size| (legacy behavior).
    daily_loss_pct : float
        Pass state.risk_budget_pct, not rule_set.daily_loss_limit_pct, to
        get a ceiling valid at the current point in the session.

    Returns
    -------
    float
        Maximum stop-loss per trade in pips. inf when bet_size is 0.
    """
    if not 0.0 <= abs(bet_size) <= 1.0:
        raise ValueError(f"bet_size must lie in [-1, 1]; got {bet_size!r}")
    if lot_size <= 0 or pip_value_per_lot <= 0 or n_concurrent < 1:
        raise ValueError("lot_size, pip_value_per_lot must be > 0; n_concurrent >= 1")

    exposure_lots = lot_size * abs(bet_size)
    if not aggregate_exposure:
        exposure_lots *= n_concurrent
    if exposure_lots == 0.0:
        return float("inf")

    daily_limit_usd = account_usd * daily_loss_pct
    return daily_limit_usd / (exposure_lots * pip_value_per_lot)


def assert_timescale_match(max_hold: int, max_hp: int) -> None:
    """Errata checklist item 5: estimation and simulation horizons must match."""
    if int(max_hold) != int(max_hp):
        raise ValueError(
            f"max_hold={max_hold} (path construction) != max_hp={max_hp} "
            "(simulation). O-U parameters are per-bar; a horizon mismatch "
            "silently changes the censoring of the estimation sample."
        )
