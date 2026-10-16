# ---------------------------------------------------------------------------
# Polars-accelerated standard bar builder (internal)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from loguru import logger

# Local package imports required by the bar builder
from ..util.misc import flatten_column_names


def _make_standard_bars_polars(
    tick_df: pd.DataFrame,
    bar_type: str,
    bar_size: int,
    price: str,
    tick_num: bool,
) -> pd.DataFrame:
    """
    Polars-accelerated OHLC aggregation for tick / volume / dollar bars.

    Called by make_standard_bars when len(tick_df) >= _POLARS_THRESHOLD.
    Returns the same schema as the pandas path; column order may differ slightly.
    """
    import polars as pl

    price_cols = ["bid", "ask"] if price == "bid_ask" else [price]
    extra_cols = ["spread", "spread_bps"]
    has_volume = "volume" in tick_df.columns and bar_type in ("volume", "dollar")
    if has_volume:
        extra_cols.append("volume")

    sub = tick_df[price_cols + extra_cols].assign(_time=tick_df.index)

    if bar_type == "tick":
        sub["_bar_id"] = np.arange(len(sub), dtype=np.int64) // bar_size
    elif bar_type == "volume":
        sub["_bar_id"] = (tick_df["volume"].cumsum() // bar_size).astype(np.int64).to_numpy()
    else:  # dollar
        sub["_bar_id"] = (
            ((tick_df["volume"] * tick_df["mid_price"]).cumsum() // bar_size)
            .astype(np.int64)
            .to_numpy()
        )

    pldf = pl.from_pandas(sub.reset_index(drop=True))

    # Row index only needed for tick_num on volume/dollar bars.
    if tick_num and bar_type != "tick":
        pldf = pldf.with_row_index("_row_idx")

    if price == "bid_ask":
        ohlc_exprs = [
            expr
            for c in ("bid", "ask")
            for expr in (
                pl.col(c).first().alias(f"{c}_open"),
                pl.col(c).max().alias(f"{c}_high"),
                pl.col(c).min().alias(f"{c}_low"),
                pl.col(c).last().alias(f"{c}_close"),
            )
        ]
    else:
        ohlc_exprs = [
            pl.col(price).first().alias("open"),
            pl.col(price).max().alias("high"),
            pl.col(price).min().alias("low"),
            pl.col(price).last().alias("close"),
        ]

    agg_exprs = ohlc_exprs + [
        pl.col("spread").mean(),
        pl.col("spread_bps").mean(),
        pl.len().alias("tick_volume"),
        pl.col("_time").last().alias("_bar_close"),
    ]
    if has_volume:
        agg_exprs.append(pl.col("volume").sum())
    if tick_num and bar_type != "tick":
        agg_exprs.append(pl.col("_row_idx").last().alias("_last_row"))

    result = pldf.group_by("_bar_id", maintain_order=True).agg(agg_exprs).sort("_bar_id")

    n_ticks = len(tick_df)
    if bar_type == "tick":
        result = result[: n_ticks // bar_size]
    elif n_ticks % bar_size > 0:
        result = result[:-1]

    pdf = result.drop("_bar_id").to_pandas()

    if price == "bid_ask":
        for col in ("open", "high", "low", "close"):
            pdf[col] = (pdf[f"bid_{col}"] + pdf[f"ask_{col}"]) / 2

    pdf.index = pd.DatetimeIndex(pdf.pop("_bar_close")) + pd.Timedelta(microseconds=1)

    if tick_num:
        if bar_type == "tick":
            n_bars = len(pdf)
            pdf["tick_num"] = np.arange(1, n_bars + 1, dtype=np.int64) * bar_size
        else:
            pdf["tick_num"] = (pdf.pop("_last_row") + 1).astype(np.int64)

    return pdf


# ---------------------------------------------------------------------------
# Standard bar builder (internal)
# ---------------------------------------------------------------------------


def make_standard_bars(
    tick_df: pd.DataFrame,
    bar_type: str,
    bar_size: int | str,
    price: str,
    tick_num: bool,
) -> pd.DataFrame:
    """
    Build standard (time / tick / volume / dollar) OHLC bars.

    This is the original make_bars logic extracted verbatim so that
    make_bars can cleanly dispatch between standard and information bars.

    Parameters
    ----------
    tick_df  : prepared tick DataFrame (mid_price / spread already added)
    bar_type : 'time', 'tick', 'volume', or 'dollar'
    bar_size : timeframe string or integer count
    price    : price column strategy
    tick_num : whether to add tick_num column

    Returns
    -------
    pd.DataFrame
        OHLC bars (not yet post-processed).
    """
    from .bars import _POLARS_THRESHOLD
    from .utils import (
        _get_bar_tick_indices,
        _make_bar_type_grouper,
        calculate_ticks_per_period,
    )

    # Polars fast path for large non-time datasets.
    if bar_type != "time" and len(tick_df) >= _POLARS_THRESHOLD:
        resolved_size = bar_size
        if bar_type == "tick" and isinstance(bar_size, str):
            resolved_size = calculate_ticks_per_period(tick_df, bar_size)
        if isinstance(resolved_size, int) and resolved_size > 0:
            try:
                return _make_standard_bars_polars(tick_df, bar_type, resolved_size, price, tick_num)
            except ImportError:
                logger.debug("polars not available; falling back to pandas bar builder")

    price_cols = ["bid", "ask"] if price == "bid_ask" else [price]
    price_cols += ["spread", "spread_bps"]

    if bar_type in ("volume", "dollar"):
        if "volume" not in tick_df.columns:
            raise KeyError(f"'volume' column required for {bar_type} bars")
        price_cols.append("volume")

    bar_group, bar_size, bar_id = _make_bar_type_grouper(tick_df[price_cols], bar_type, bar_size)

    # --- OHLC ---
    if price != "bid_ask":
        ohlc_df = bar_group[price].ohlc()
    else:
        ohlc_df = bar_group.agg({k: "ohlc" for k in ("bid", "ask")})
        ohlc_df.columns = flatten_column_names(ohlc_df)
        for col in ["open", "high", "low", "close"]:
            ohlc_df[col] = ohlc_df.filter(regex=col).sum(axis=1).div(2)

    # --- Additional columns ---
    ohlc_df["spread"] = bar_group["spread"].mean()
    ohlc_df["spread_bps"] = bar_group["spread_bps"].mean()
    ohlc_df["tick_volume"] = bar_group.size() if bar_type != "tick" else bar_size

    if "volume" in tick_df.columns:
        ohlc_df["volume"] = bar_group["volume"].sum()

    # --- Bar-type specific index / tick_num ---
    if bar_type == "time":
        eq_zero = ohlc_df["tick_volume"] == 0
        ohlc_df = ohlc_df[~eq_zero]

        nzeros = eq_zero.sum()
        if nzeros > 0:
            nrows = ohlc_df.shape[0]
            logger.info(
                f"Dropped {nzeros:,} of {nrows:,} "
                f"({nzeros / nrows:.2%}) rows with zero tick volume."
            )

        if tick_num:
            ohlc_df["tick_num"] = ohlc_df["tick_volume"].cumsum()

    else:
        ohlc_df.index = bar_group["time"].last() + pd.Timedelta(microseconds=1)

        if len(tick_df) % bar_size > 0:
            ohlc_df = ohlc_df.iloc[:-1]

        if tick_num:
            ohlc_df["tick_num"] = _get_bar_tick_indices(len(tick_df), bar_size, bar_id)

    return ohlc_df
