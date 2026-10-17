"""
afml/transaction_costs.py

Loads broker transaction cost data exported by TransactionCostCollector.mq5
(v3.00+, tick-sampled) and derives the min_ret threshold for triple-barrier
label construction. Also loads a high-impact news calendar exported by the
companion CalendarCollector.mq5 script for use with filter_news_by_cost_model().

The threshold is the minimum return a trade must achieve to cover the
round-trip transaction cost at a given spread percentile, slippage
assumption, and holding-period-adjusted swap accrual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Swap-mode string vocabulary ─────────────────────────────────────────────
# TransactionCostCollector.mq5 v1/v2 and v3 spell these differently -- v3
# renamed several to track the SYMBOL_SWAP_MODE_* enum names more closely
# and added explicit labels for the two REOPEN_* modes, which v1/v2 both
# silently collapsed into "unknown". Both spellings are accepted here so a
# CSV from either collector version parses the same way.
_SWAP_CURRENCY_MODES = frozenset({
    "currency", "currency_mrgn", "currency_dep",                  # v1/v2
    "currency_symbol", "currency_margin", "currency_deposit",     # v3
})
_SWAP_INTEREST_MODES = frozenset({
    "interest_open", "interest_curr", "interest_current",
})
_SWAP_DISABLED_MODES = frozenset({"unknown", "disabled_or_unknown"})
_SWAP_REOPEN_MODES = frozenset({"reopen_current", "reopen_bid"})


@dataclass
class TransactionCostModel:
    """
    Broker-specific transaction cost model for a single symbol.

    All costs are expressed as fractional returns (e.g., 0.0001 = 1 pip
    on a 1.0000 priced instrument) so they can be directly compared to
    the return series used for triple-barrier labeling.

    Parameters
    ----------
    symbol : str
        Instrument identifier (e.g., "EURUSD").
    spread_pips : float
        Spread to use for cost calculation. Recommend p95 from the
        collected distribution — not the mean — because entries during
        high-spread periods are disproportionately costly.
    slippage_pips : float
        One-way slippage estimate. Derived from live/demo trade log.
        Default 0.5 pips is conservative for major forex pairs.
    commission_per_lot : float
        Round-trip commission in account currency per standard lot.
        Must be confirmed from a reference trade (see MQL5 script note).
    swap_long_per_night : float
        Swap in native MQL5 units for long positions.
    swap_short_per_night : float
        Swap in native MQL5 units for short positions (usually negative).
    swap_mode : str
        MQL5 SYMBOL_SWAP_MODE string from the CSV. Accepts both the v1/v2
        and v3 TransactionCostCollector.mq5 spellings (see module-level
        _SWAP_*_MODES sets).
    swap_triple_day : int
        Weekday on which triple swap is charged (0=Sun … 6=Sat).
    pip_factor : float
        Points per pip. 10 for 5-digit brokers, 1 for 4-digit.
    point : float
        Broker point size (e.g., 0.00001 for EURUSD 5-digit).
    tick_value : float
        Account currency value of one tick per standard lot.
    tick_size : float
        Minimum price movement.
    contract_size : float
        Units per standard lot (e.g., 100,000 for forex).
    lot_size : float
        Lot size used for this strategy (e.g., 0.1 mini-lot).
    account_currency_rate : float
        Exchange rate from profit currency to account currency.
        Set to 1.0 if profit currency == account currency.
    spread_by_hour : dict[int, float]
        Hour-of-day mean spread in pips (broker time). Used for
        session-aware cost estimation.
    currency_base : str
        SYMBOL_CURRENCY_BASE from the broker's own symbol specification
        (e.g., "EUR" for EURUSD). Collected by the MQL5 script since v1;
        not previously read by this loader.
    currency_profit : str
        SYMBOL_CURRENCY_PROFIT (e.g., "USD" for EURUSD).
    currency_margin : str
        SYMBOL_CURRENCY_MARGIN. Recorded for completeness; not used by
        correlated_currencies below, since margin currency is about
        account funding, not what the instrument's price responds to.
    """

    symbol:               str
    spread_pips:          float
    slippage_pips:        float   = 0.5
    commission_per_lot:   float   = 0.0
    swap_long_per_night:  float   = 0.0
    swap_short_per_night: float   = 0.0
    swap_mode:            str     = "points"
    swap_triple_day:      int     = 3        # Wednesday default
    pip_factor:           float   = 10.0
    point:                float   = 0.00001
    tick_value:           float   = 10.0
    tick_size:            float   = 0.00001
    contract_size:        float   = 100_000.0
    lot_size:             float   = 0.01
    account_currency_rate: float  = 1.0
    spread_by_hour:       dict[int, float] = field(default_factory=dict)
    currency_base:        str     = ""
    currency_profit:      str     = ""
    currency_margin:      str     = ""

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def correlated_currencies(self) -> set[str]:
        """
        Currencies whose high-impact news events are relevant to this
        symbol, per FundedNext's own correlation rule (a EUR/USD trade is
        affected by EUR or USD news, not CNY news). Uses base and profit
        currency; margin currency is excluded for the reason noted above.
        Empty if currency_base/currency_profit were never populated --
        callers should treat that as "unknown," not "correlates with
        nothing."
        """
        return {c for c in (self.currency_base, self.currency_profit) if c}

    @property
    def pip_value(self) -> float:
        """Account currency value of one pip per lot_size."""
        return (self.tick_value / self.tick_size) * (
            self.pip_factor * self.point
        ) * self.lot_size * self.account_currency_rate

    def spread_cost_frac(self, entry_price: float) -> float:
        """
        Round-trip spread cost as a fraction of entry price.

        The full spread applies once, round trip, for a market-order
        entry (you cross the spread on entry; the exit price already
        reflects the other side of the book).
        """
        spread_price = self.spread_pips * self.pip_factor * self.point
        return spread_price / entry_price

    def slippage_cost_frac(self, entry_price: float) -> float:
        """Round-trip slippage as a fraction of entry price (2× one-way)."""
        slippage_price = (
            self.slippage_pips * self.pip_factor * self.point * 2
        )
        return slippage_price / entry_price

    def commission_cost_frac(self, entry_price: float) -> float:
        """
        Round-trip commission as a fraction of entry price.

        commission_per_lot is typically per-side; multiply by 2 for
        round trip. Divide by notional value to get a fraction.
        """
        notional = entry_price * self.contract_size * self.lot_size
        if notional == 0:
            return 0.0
        return (self.commission_per_lot * 2 * self.account_currency_rate) / notional

    def swap_cost_frac(
        self,
        entry_price: float,
        holding_days: float,
        side: int = 1,
    ) -> float:
        """
        Swap accrual as a fraction of entry price for a given holding period.

        Sign convention: MQL5 reports a negative swap_long/swap_short as a
        debit from the account (a cost) and a positive value as a carry
        credit. This method negates the raw rate so a debit produces a
        positive cost fraction and a credit produces a negative one --
        i.e. a carry credit *reduces* round_trip_cost_frac rather than
        being folded in as an unsigned cost. Do not wrap the rate in
        abs(): that silently converts every carry credit into a cost,
        which understates edge for carry trades like long AUDJPY.

        Handles the triple-swap day: if holding_days >= 3, an additional
        2 nights of swap are charged, on the simplifying assumption that
        a 3+ day hold spans the rollover day.

        Parameters
        ----------
        entry_price  : float  Entry price of the trade.
        holding_days : float  Expected holding period in calendar days.
        side         : int    +1 for long, -1 for short.

        Raises
        ------
        ValueError
            If swap_mode is a REOPEN_CURRENT/REOPEN_BID variant. Those
            modes recompute swap by reopening the position at a new
            reference price each night rather than accruing a fixed
            per-night rate, so no constant-rate approximation applies --
            silently returning 0.0 would misprice these symbols rather
            than merely approximating them.
        """
        rate = (
            self.swap_long_per_night if side >= 0
            else self.swap_short_per_night
        )
        nights = holding_days
        if holding_days >= 3:
            nights += 2

        if self.swap_mode == "points":
            swap_price = -rate * self.point * nights
            return swap_price / entry_price if entry_price > 0 else 0.0

        elif self.swap_mode in _SWAP_CURRENCY_MODES:
            total_swap = -rate * self.lot_size * nights
            notional = entry_price * self.contract_size * self.lot_size
            return total_swap / notional if notional > 0 else 0.0

        elif self.swap_mode in _SWAP_INTEREST_MODES:
            nightly_rate = -rate / 100.0 / 365.0
            return nightly_rate * nights

        elif self.swap_mode in _SWAP_DISABLED_MODES:
            return 0.0

        elif self.swap_mode in _SWAP_REOPEN_MODES:
            raise ValueError(
                f"symbol={self.symbol!r} has swap_mode={self.swap_mode!r}. "
                "REOPEN_CURRENT/REOPEN_BID swap is not a fixed per-night "
                "rate and is not modeled by swap_cost_frac(). Confirm the "
                "actual mechanism with the broker before adding a case, "
                "or pass holding_days=0.0 if this symbol is only ever "
                "traded intraday."
            )

        raise ValueError(
            f"symbol={self.symbol!r} has unrecognized swap_mode="
            f"{self.swap_mode!r}. Re-run TransactionCostCollector.mq5 "
            "or check the loaded CSV for a corrupted swap section."
        )

    def round_trip_cost_frac(
        self,
        entry_price: float,
        holding_days: float = 0.0,
        side: int = 1,
    ) -> float:
        """
        Total round-trip cost as a fraction of entry price.

        Parameters
        ----------
        entry_price  : Reference price (e.g., close at label bar).
        holding_days : Expected holding period. Used for swap accrual.
                       Pass 0.0 for intraday strategies (no overnight).
        side         : +1 long, -1 short. Affects swap direction.
        """
        return (
            self.spread_cost_frac(entry_price)
            + self.slippage_cost_frac(entry_price)
            + self.commission_cost_frac(entry_price)
            + self.swap_cost_frac(entry_price, holding_days, side)
        )

    def min_ret_for_symbol(
        self,
        price_series: pd.Series,
        holding_days: float = 0.0,
        side: int = 1,
        cost_multiplier: float = 1.5,
    ) -> float:
        """
        Derive the min_ret threshold for triple-barrier labeling.

        The threshold is set at cost_multiplier × median round-trip
        cost across the price series. A cost_multiplier of 1.5 means
        the profit barrier must be at least 1.5× the transaction cost
        to receive a positive label — trades that barely cover costs
        are treated as non-events.

        The result is floored at zero: a strong carry trade can push the
        median round-trip cost negative, and a negative min_ret would
        label every trade as positive regardless of return.

        Parameters
        ----------
        price_series    : pd.Series  Close prices (same index as labels).
        holding_days    : float      Expected average holding period.
        side            : int        +1 or -1. Use 1 if unknown.
        cost_multiplier : float      Safety margin above break-even.
                                     1.0 = break-even; 1.5 = recommended.

        Returns
        -------
        float
            min_ret value to pass to get_events() or equivalent.
        """
        costs = price_series.apply(
            lambda p: self.round_trip_cost_frac(p, holding_days, side)
        )
        return float(max(0.0, costs.median() * cost_multiplier))

    def session_adjusted_spread_pips(self, hour: int) -> float:
        """
        Return mean spread for a given hour-of-day (broker time).
        Falls back to the model's spread_pips if hour is not in data.
        """
        return self.spread_by_hour.get(hour, self.spread_pips)

    def summary(self, entry_price: float, holding_days: float = 1.0) -> dict:
        """Human-readable cost breakdown for a reference trade."""
        return {
            "spread_frac":      self.spread_cost_frac(entry_price),
            "slippage_frac":    self.slippage_cost_frac(entry_price),
            "commission_frac":  self.commission_cost_frac(entry_price),
            "swap_long_frac":   self.swap_cost_frac(entry_price, holding_days, 1),
            "swap_short_frac":  self.swap_cost_frac(entry_price, holding_days, -1),
            "total_long_frac":  self.round_trip_cost_frac(entry_price, holding_days, 1),
            "total_short_frac": self.round_trip_cost_frac(entry_price, holding_days, -1),
            "total_long_pips":  self.round_trip_cost_frac(entry_price, holding_days, 1)
                                * entry_price / (self.pip_factor * self.point),
            "total_short_pips": self.round_trip_cost_frac(entry_price, holding_days, -1)
                                * entry_price / (self.pip_factor * self.point),
        }


# ── CSV loader ────────────────────────────────────────────────────────────────

# TransactionCostCollector.mq5 v3 replaced the single "spread_summary"
# section with four differently-weighted distributions (see the script's
# own header comment). spread_exec_conditional -- the quote prevailing at
# each execution-timeframe boundary -- is what the script recommends for
# cost modeling, so it is the default source here.
_KNOWN_SPREAD_SECTIONS = frozenset({
    "spread_tick_weighted", "spread_time_weighted",
    "spread_exec_conditional", "spread_bar_field",
})


def load_cost_model(
    csv_path: Path,
    spread_section: str = "spread_exec_conditional",
    spread_percentile: str = "p95_pips",
    slippage_pips: float   = 0.5,
    commission_per_lot: float = 0.0,
    lot_size: float        = 0.01,
    account_currency_rate: float = 1.0,
) -> TransactionCostModel:
    """
    Build a TransactionCostModel from a CSV exported by
    TransactionCostCollector.mq5 (v3.00+).

    Parameters
    ----------
    csv_path           : Path to the <symbol>_costs.csv file.
    spread_section      : Which of the four v3 spread distributions to
                          read spread_percentile from. Default
                          "spread_exec_conditional" matches what the
                          collector script itself recommends for cost
                          modeling. "spread_time_weighted" only carries
                          mean_pips (no percentiles) -- pass
                          spread_percentile="mean_pips" if you use it.
    spread_percentile  : Which spread statistic to use as the model
                         spread within spread_section. Options: mean_pips,
                         p50_pips, p95_pips, p99_pips (the latter two are
                         not written under spread_time_weighted).
    slippage_pips      : One-way slippage (not in CSV — must be supplied
                         from live/demo trade log analysis).
    commission_per_lot : Round-trip commission per standard lot in
                         account currency. Must be confirmed from a
                         reference trade (see MQL5 script note in CSV).
    lot_size           : Strategy lot size.
    account_currency_rate : Profit-to-account-currency exchange rate.

    Returns
    -------
    TransactionCostModel

    Raises
    ------
    KeyError
        If the CSV predates the v3 collector (still has a single
        "spread_summary" section instead of the four v3 distributions).
        This is deliberately not silently accepted: spread_summary was
        built from CopySpread(), which a reader of Part 14 correctly
        flagged as returning the lowest spread registered within each
        bar's timespan rather than the real tick-level spread. Loading
        it here would silently reintroduce that understatement. Re-run
        TransactionCostCollector.mq5 v3+ to get a tick-sampled export.
    """
    df = pd.read_csv(csv_path)

    def get(section: str, key: str) -> str:
        row = df[(df["section"] == section) & (df["key"] == key)]
        if row.empty:
            raise KeyError(f"Missing: section={section!r} key={key!r} in {csv_path}")
        return str(row["value"].iloc[0]).strip()

    # ── Status guard ─────────────────────────────────────────────────
    # TransactionCostCollector.mq5 v2+ writes a status row on both the
    # success and failure paths. If tick sampling failed, the spread
    # sections are absent even though symbol_properties/swap/commission
    # were written successfully. Fail loudly here with the actual cause
    # rather than letting a spread-section lookup below raise an
    # unrelated-looking KeyError. CSVs from v1 (pre status row) have no
    # "status" section at all; those are treated as complete for
    # backward compatibility.
    status_rows = df[df["section"] == "status"]
    if not status_rows.empty:
        status_key = str(status_rows["key"].iloc[0]).strip()
        if status_key == "error":
            reason = str(status_rows["value"].iloc[0]).strip()
            raise RuntimeError(
                f"{csv_path} is an incomplete cost export "
                f"(status=error, reason={reason!r}). "
                "Spread sections are missing; re-run "
                "TransactionCostCollector.mq5 with a longer or "
                "different lookback window before building a cost model "
                "from this file."
            )

    if spread_section not in _KNOWN_SPREAD_SECTIONS:
        raise ValueError(
            f"spread_section={spread_section!r} is not one of "
            f"{sorted(_KNOWN_SPREAD_SECTIONS)}."
        )
    if df[df["section"] == spread_section].empty:
        if not df[df["section"] == "spread_summary"].empty:
            raise KeyError(
                f"{csv_path} has a 'spread_summary' section but no "
                f"{spread_section!r} section -- this looks like it was "
                "exported by the pre-v3 collector (CopySpread(), bar "
                "resolution). Re-run TransactionCostCollector.mq5 v3+ "
                "to get a tick-sampled export; do not substitute "
                "spread_summary, which understates spread at the "
                "resolution CopySpread() reports it."
            )
        raise KeyError(
            f"{csv_path} has no {spread_section!r} section at all. "
            f"Re-run TransactionCostCollector.mq5 v3+."
        )

    symbol       = get("symbol_properties", "symbol")
    point        = float(get("symbol_properties", "point"))
    pip_factor   = float(get("symbol_properties", "pip_factor"))
    tick_size    = float(get("symbol_properties", "tick_size"))
    tick_value   = float(get("symbol_properties", "tick_value"))
    contract_sz  = float(get("symbol_properties", "contract_size"))
    currency_base   = get("symbol_properties", "currency_base")
    currency_profit = get("symbol_properties", "currency_profit")
    currency_margin = get("symbol_properties", "currency_margin")

    swap_long    = float(get("swap", "swap_long"))
    swap_short   = float(get("swap", "swap_short"))
    swap_mode    = get("swap", "swap_mode")
    swap_3day    = int(get("swap", "swap_3day"))

    spread_pips  = float(get(spread_section, spread_percentile))

    # Session spreads (unchanged format between collector versions)
    hour_rows = df[df["section"] == "spread_by_hour"]
    spread_by_hour: dict[int, float] = {}
    for _, row in hour_rows.iterrows():
        hour = int(str(row["key"]).replace("hour_", ""))
        spread_by_hour[hour] = float(row["value"])

    return TransactionCostModel(
        symbol               = symbol,
        spread_pips          = spread_pips,
        slippage_pips        = slippage_pips,
        commission_per_lot   = commission_per_lot,
        swap_long_per_night  = swap_long,
        swap_short_per_night = swap_short,
        swap_mode            = swap_mode,
        swap_triple_day      = swap_3day,
        pip_factor           = pip_factor,
        point                = point,
        tick_value           = tick_value,
        tick_size            = tick_size,
        contract_size        = contract_sz,
        lot_size             = lot_size,
        account_currency_rate= account_currency_rate,
        spread_by_hour       = spread_by_hour,
        currency_base        = currency_base,
        currency_profit      = currency_profit,
        currency_margin      = currency_margin,
    )


# ── Calendar loader (companion to CalendarCollector.mq5) ──────────────────────

def load_calendar_csv(csv_path: Path) -> list[tuple[pd.Timestamp, str]]:
    """
    Load high-impact calendar events exported by CalendarCollector.mq5.

    Returns (event_time, currency) pairs in exactly the format
    filter_news_by_cost_model() expects.

    Event times are in the trade-server timezone CalendarValueHistory()
    reports them in (TimeTradeServer()), which is not necessarily UTC.
    news_window_factor documents current_time and news_times as UTC
    (see Machine Learning Under Constraint, Part 3, Section 12) --
    convert server time to UTC before passing these times through that
    function, using the broker's own UTC offset, not an assumed one.
    """
    df = pd.read_csv(csv_path)
    rows = df[df["section"] == "high_impact_events"]
    if rows.empty:
        raise KeyError(
            f"{csv_path} has no 'high_impact_events' section. "
            "Re-run CalendarCollector.mq5 or check the loaded CSV."
        )
    return [
        (pd.Timestamp(row["value"]), str(row["unit"]))
        for _, row in rows.iterrows()
    ]


# ── News-calendar correlation filter (Machine Learning Under Constraint, Part 3) ──

def filter_news_by_cost_model(
    calendar: list[tuple["pd.Timestamp", str]],
    cost_model: TransactionCostModel,
) -> list["pd.Timestamp"]:
    """
    calendar: (event_time, currency) pairs -- e.g. from load_calendar_csv().

    Returns only the event times whose currency is one of cost_model's
    correlated currencies -- e.g., for a EURUSD model, keeps EUR and USD
    events and drops everything else (CNY, JPY, ...), per FundedNext's
    own "must directly correlate with the traded pair" rule.

    Uses currency_base/currency_profit as collected by
    TransactionCostCollector.mq5 from the broker's own symbol
    specification, rather than a hand-maintained currency-pair table.
    """
    relevant = cost_model.correlated_currencies
    if not relevant:
        raise ValueError(
            f"{cost_model.symbol!r} has no currency_base/currency_profit set -- "
            f"re-run TransactionCostCollector.mq5 or check the loaded CSV."
        )
    return [t for t, ccy in calendar if ccy in relevant]
