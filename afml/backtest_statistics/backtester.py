import asyncio

import numpy as np

from ..data_structures.bars import make_bars
from ..labeling.triple_barrier import triple_barrier_labels
from ..mt5.load_data import load_tick_data
from ..util.constants import DATA_PATH


class BacktestHook:
    """
    Base hook interface. Subclass and override any methods you need.
    Hooks have a `priority` attribute—higher values run earlier.
    """

    priority = 0

    def before_strategy(self, df):
        return df

    def after_strategy(self, strategy_df):
        return strategy_df

    def before_labeling(self, strategy_df, t_events):
        return strategy_df, t_events

    def after_labeling(self, labels):
        return labels


# Hook registry and decorator
HOOK_REGISTRY = []


def register_hook(cls):
    """Decorator to auto-register a hook class."""
    inst = cls()
    HOOK_REGISTRY.append(inst)
    return cls


# Unified (sync/async) hook runner
async def _run_hook(hook, method_name, *args):
    method = getattr(hook, method_name)
    if asyncio.iscoroutinefunction(method):
        return await method(*args)
    return method(*args)


def backtest_engine(
    assets,
    timeframes,
    model_data_template,
    func,
    kargs,
    target,
    pt_sl,
    hooks=None,
    **label_kwargs,
):
    """
    Synchronous backtest engine.
    - `assets`: list of asset identifiers
    - `timeframes`: list of timeframe strings
    - `model_data_template`: dict with template metadata
    - `func`: strategy function, signature func(data=df, **kargs)
    - `kargs`: dict of strategy parameters
    - `target`: pd.Series target volatility or returns
    - `pt_sl`: tuple of (profit_taking, stop_loss)
    - `hooks`: list of BacktestHook instances
    - `label_kwargs`: passed to triple_barrier_labels
    """
    # Sort hooks by priority (descending)
    hooks = sorted(hooks or [], key=lambda h: getattr(h, "priority", 0), reverse=True)
    results = {}

    # Variables needed for load_data function
    global start_date
    global end_date
    global account
    global bar_type
    start_date = model_data_template["start_date"]
    end_date = model_data_template["end_date"]
    account = model_data_template["account"]
    bar_type = model_data_template["bar_type"]

    for asset in assets:
        for tf in timeframes:
            df = load_data(asset, tf)
            model_data = dict(model_data_template, asset=asset, timeframe=tf)

            # pre‐strategy hooks
            for hook in hooks:
                df = hook.before_strategy(df)

            # run strategy
            strategy_df = func(data=df, **kargs)

            # post‐strategy hooks
            for hook in hooks:
                strategy_df = hook.after_strategy(strategy_df)

            # identify entries
            side = strategy_df["side"]
            entry_mask = side.notna() & (side != side.shift()) & (side != 0)
            t_events = side[entry_mask].index

            # before‐labeling hooks
            for hook in hooks:
                strategy_df, t_events = hook.before_labeling(strategy_df, t_events)

            # triple barrier labeling
            labels = triple_barrier_labels(
                close=strategy_df["close"],
                t_events=t_events,
                target=target,
                pt_sl=pt_sl,
                **label_kwargs,
            )

            # after‐labeling hooks
            for hook in hooks:
                labels = hook.after_labeling(labels)

            model_data["labels"] = labels
            results[(asset, tf)] = model_data

    return results


async def backtest_engine_async(
    assets,
    timeframes,
    model_data_template,
    func,
    kargs,
    target,
    pt_sl,
    hooks=None,
    **label_kwargs,
):
    """
    Asynchronous backtest engine.
    Non‐blocking hooks (`async def`) run in parallel where possible.
    """
    hooks = sorted(hooks or [], key=lambda h: getattr(h, "priority", 0), reverse=True)
    results = {}

    # Variables needed for load_data function
    global start_date
    global end_date
    global account
    start_date = model_data_template["start_date"]
    end_date = model_data_template["end_date"]
    account = model_data_template["account"]

    for asset in assets:
        for tf in timeframes:
            df = load_data(asset, tf)
            model_data = dict(model_data_template, asset=asset, timeframe=tf)

            # pre‐strategy hooks (sequential)
            for hook in hooks:
                df = await _run_hook(hook, "before_strategy", df)

            strategy_df = func(data=df, **kargs)

            # post‐strategy hooks
            for hook in hooks:
                strategy_df = await _run_hook(hook, "after_strategy", strategy_df)

            side = strategy_df["side"]
            entry_mask = side.notna() & (side != side.shift()) & (side != 0)
            t_events = side[entry_mask].index

            # before‐labeling hooks
            for hook in hooks:
                strategy_df, t_events = await _run_hook(
                    hook, "before_labeling", strategy_df, t_events
                )

            labels = triple_barrier_labels(
                close=strategy_df["close"],
                t_events=t_events,
                target=target,
                pt_sl=pt_sl,
                **label_kwargs,
            )

            # after‐labeling hooks (parallel)
            await asyncio.gather(*[_run_hook(hook, "after_labeling", labels) for hook in hooks])

            model_data["labels"] = labels
            results[(asset, tf)] = model_data

    return results


# Example hooks


@register_hook
class VolumeFilterHook(BacktestHook):
    priority = 10

    def before_strategy(self, df):
        """Filter out bars where tick_volume ≤ 10."""
        return df[df["tick_volume"] >= 10]


@register_hook
class CleanDataHook(BacktestHook):
    priority = 8

    def before_strategy(self, df):
        """Drop any NaNs or Inf before running strategy."""
        return df.replace([np.inf, -np.inf], np.nan).dropna()


@register_hook
class EntryLoggerHook(BacktestHook):
    priority = 5

    def before_labeling(self, df, t_events):
        print(f"[EntryLogger] {len(t_events):,} of {len(df):,} ({len(t_events) / len(df):.2%})")
        return df, t_events


@register_hook
class MetricsHook(BacktestHook):
    priority = 1

    def after_labeling(self, labels):
        print(f"[Metrics] total labels: {len(labels):,}")
        print(f"Average uniqueness of labels is {events.tW.mean():.4f}.")
        print(value_counts_data(events["bin"]))
        print(f"Cumulative returns: {((1 + events.ret).cumprod() - 1)[-1]:.2%}")
        print(
            f"\nReturns skew: {events.ret.skew():.4f} \nReturns kurtosis: {events.ret.kurt():.4f}\n"
        )

        return labels


# Placeholder functions—implement these to suit your environment
def load_data(asset, timeframe):
    """
    Load and return a pandas.DataFrame for the given asset/timeframe.
    Must contain at least 'close', 'side', and 'volume' columns.
    """
    tick_df = load_tick_data(DATA_PATH, asset, start_date, end_date, account, verbose=False)
    df = make_bars(tick_df, bar_type, timeframe=timeframe, verbose=False)
    return df
