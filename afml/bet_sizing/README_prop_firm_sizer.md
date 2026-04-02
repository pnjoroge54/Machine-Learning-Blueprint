# Prop Firm Position Sizer

A two-stage, stateful position sizing system for algorithmic trading under
prop firm risk rules, specifically calibrated to the
**FundedNext Stellar 2-Step Challenge**.

It integrates three layers of financial ML theory:

1. **AFML Chapter 10** signal sizing — `bet_size_probability` with concurrent
   label averaging
2. **Dynamic w_param calibration** — position compression driven by the
   remaining drawdown budget in real time
3. **Kelly payoff adjustment** — asymmetric win/loss ratio correction as a
   signal multiplier

---

## Table of Contents

1. [Background](#background)
2. [FundedNext Stellar 2-Step Rules](#fundednext-stellar-2-step-rules)
3. [Architecture](#architecture)
4. [Module Contents](#module-contents)
5. [Installation and Dependencies](#installation-and-dependencies)
6. [Quick Start](#quick-start)
7. [Detailed Parameter Guide](#detailed-parameter-guide)
8. [Integration with the AFML Pipeline](#integration-with-the-afml-pipeline)
9. [Live Trading Loop Template](#live-trading-loop-template)
10. [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
11. [Limitations and Known Gaps](#limitations-and-known-gaps)
12. [References](#references)

---

## Background

Standard prop firm sizing guides reduce to simple rules like "risk 1% per
trade." This is correct as a floor but ignores two critical realities:

- The available risk budget is not constant. It shrinks as daily and
  overall drawdown limits are consumed, and it expands when intraday
  profit is made. A fixed 1% rule overcrowds trades late in a bad day.
- The ML signal carries information about confidence and direction that
  a fixed-percentage rule discards entirely.

This module connects those two facts. The remaining risk budget determines
*how aggressively* the sigmoid/power function maps ML signal strength to
position size. A fresh account at the start of the day can deploy the full
budget at full signal strength. An account 80% through its daily limit
flattens the curve dramatically — the same ML signal now produces a much
smaller position — without the trader needing to override anything manually.

---

## FundedNext Stellar 2-Step Rules

All rules encoded in `FundedNextStellar2StepRules` (frozen dataclass).

| Rule | Value | Notes |
|---|---|---|
| Daily loss limit | 5% of initial balance | Expands by intraday profit |
| Overall max loss | 10% of initial balance | Floor fixed at 90% of initial |
| Phase 1 target | 8% profit | No time limit |
| Phase 2 target | 5% profit | No time limit |
| Min trading days | 5 per phase | Calendar days with a trade |
| Max leverage | 1:100 | |
| News window (funded) | ±5 minutes | Only 40% of profits count |
| Challenge profit split | 15% | First payout only |
| Max profit split | 95% | Scaled with account tenure |

### The Daily Limit Is Dynamic

The daily loss limit is **not** simply `initial_balance × 5%`. It expands
intraday as you make profit:

```
daily_limit = initial_balance × 5% + max(0, intraday_realized_pnl)
```

Swaps, commissions, and fees count toward the daily loss. The `state.update()`
method accepts `fees_and_swaps` as a separate argument to handle this correctly.

### The Overall Floor Is Fixed

The balance/equity floor is permanently anchored at `initial_balance × 90%`.
Making profits increases your buffer above the floor but does not move the
floor itself.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: calibrated prob (pd.Series), pred sides, events df       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  STAGE 1    │
                    │  AFML Ch.10 │  bet_size_probability(average_active=True)
                    │  Signal     │  Handles overlapping triple-barrier labels
                    └──────┬──────┘  Output: ml_signal ∈ [−1, 1]
                           │
                    ┌──────▼──────────────────────────────────────┐
                    │  STAGE 2 — Dynamic Risk-Budget Sizing        │
                    │                                              │
                    │  PropFirmAccountState                        │
                    │    ├─ daily_remaining    ─┐                  │
                    │    └─ overall_remaining  ─┴─ risk_budget     │
                    │                            │                  │
                    │  WParamCalibrator          │                  │
                    │    risk_budget × safety_factor / stop_loss   │
                    │    → cal_bet_size → get_w_sigmoid → w_param  │
                    │                            │                  │
                    │  bet_size_sigmoid(w_param, ml_signal)        │
                    │    → prop_sized                              │
                    └──────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────────────────┐
              │            │                        │
       ┌──────▼──────┐ ┌───▼───────────┐ ┌─────────▼──────────┐
       │ Derisking   │ │ Daily         │ │ News Window        │
       │ Factor      │ │ Throttle      │ │ Factor (funded)    │
       │ (phase %)   │ │ (utilisation) │ │ (0.40 credit rule) │
       └──────┬──────┘ └───┬───────────┘ └─────────┬──────────┘
              └────────────┼────────────────────────┘
                           │ × combined_factor
                    ┌──────▼──────┐
                    │  Kelly      │  Payoff-ratio multiplier
                    │  Multiplier │  f* / get_signal  [0, 1.5]
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ discrete_   │  step_size = 0.05 prevents overtrading
                    │ signal()    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ final_size  │  ∈ [−1, 1] in step_size increments
                    └─────────────┘
```

---

## Module Contents

### `FundedNextStellar2StepRules` (frozen dataclass)

Encodes all FundedNext Stellar 2-Step rules as immutable constants.
Never instantiate this directly — use the module-level `RULES` singleton.

---

### `PropFirmAccountState` (dataclass)

Real-time account state. Must be updated by the caller on every bar and
at every daily reset.

**Key properties (computed, not stored):**

| Property | Description |
|---|---|
| `daily_loss_limit` | Dynamic: 5% base + max(0, intraday profit) |
| `daily_loss_used` | Realised + floating losses today |
| `daily_remaining` | daily_loss_limit − daily_loss_used |
| `overall_floor` | Fixed at 90% of initial_balance |
| `overall_remaining` | current_equity − overall_floor |
| `risk_budget_absolute` | min(daily_remaining, overall_remaining) |
| `risk_budget_pct` | risk_budget_absolute / initial_balance |
| `phase_progress` | Fraction of current phase target achieved |
| `daily_limit_utilisation` | Fraction of today's daily limit consumed |

**Methods:**

| Method | When to call |
|---|---|
| `update(realized_pnl_delta, unrealized_pnl, fees_and_swaps)` | Every bar or trade close |
| `reset_daily()` | At 00:00 server time (FundedNext daily reset) |
| `advance_phase()` | When phase profit target is confirmed passed |
| `status_report()` | On demand — prints human-readable state snapshot |

---

### `WParamCalibrator` (dataclass)

Derives `w_param` for `bet_size_sigmoid` / `bet_size_power` from the current
risk budget. This is the mathematical core of the module.

**Chain:**
```
risk_budget_pct × safety_factor / stop_loss_pct
    → cal_bet_size  (maximum position at full signal strength)
    → get_w_sigmoid(cal_divergence, cal_bet_size)
    → w_param
    → bet_size_sigmoid(w_param, ml_signal)
    → position size
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `stop_loss_pct` | 0.01 | Per-trade stop as fraction of account |
| `cal_divergence` | 0.90 | Signal strength at which cal_bet_size is reached |
| `safety_factor` | 0.70 | Fraction of budget to actually use |
| `func` | `'sigmoid'` | `'sigmoid'` or `'power'` |
| `min_cal_bet_size` | 0.02 | Below this: no new positions |
| `max_cal_bet_size` | 0.98 | Prevents numerical instability |

---

### Modifier Functions

| Function | Range | Trigger condition |
|---|---|---|
| `profit_target_derisking_factor(state)` | [0.30, 1.0] | phase_progress > 0.80 |
| `daily_utilisation_factor(state)` | [0.10, 1.0] | daily utilisation > 0.50 |
| `news_window_factor(time, news_times, phase)` | {0.40, 1.0} | funded stage + news ±5 min |

---

### `kelly_payoff_multiplier(prob, avg_win_loss_ratio, kelly_fraction)`

Computes the ratio of fractional Kelly `f*` to the `get_signal` z-score
output at the same probability. When payoffs are symmetric (`b=1`), this
is approximately 1.0 everywhere. When `b > 1` (favourable) it amplifies
moderate-probability signals. When Kelly `f* ≤ 0`, returns 0.0 — the
position is closed regardless of the ML signal.

**Bounded** at `max_amplification` (default 1.5) to prevent runaway leverage.

---

### `PropFirmAwareSizer`

The main class. Orchestrates all stages and modifiers, calls
`bet_size_probability`, and returns a diagnostic DataFrame.

**Output columns from `size()`:**

| Column | Description |
|---|---|
| `ml_signal` | Stage 1 output from bet_size_probability |
| `w_param` | Calibrated w_param for this bar |
| `cal_bet_size` | Maximum allowed position at full signal |
| `prop_sized` | After w_param compression |
| `derisking_factor` | Phase-proximity de-risking scalar |
| `daily_factor` | Daily utilisation throttle scalar |
| `news_factor` | News-window adjustment scalar |
| `kelly_mult` | Kelly payoff multiplier per signal |
| `combined_factor` | Product of all modifiers |
| `final_size` | Final discretised position ∈ [−1, 1] |
| `risk_budget_pct` | Remaining budget as % of initial (informational) |
| `daily_used_pct` | Daily limit utilisation (informational) |

---

### `make_stellar_2step_sizer(...)` (factory function)

Convenience constructor. All parameters have sensible defaults that are
safe for a fresh challenge account.

---

## Installation and Dependencies

This module depends on the `afml` package. Import paths assume the standard
project layout:

```
Machine-Learning-Blueprint/
└── afml/
    ├── bet_sizing/
    │   ├── bet_sizing.py       ← bet_size_probability
    │   ├── ch10_snippets.py    ← bet_size_sigmoid, get_w_sigmoid, discrete_signal
    │   └── ef3m.py
    └── cross_validation/
        └── cross_validation.py
```

Adjust the imports in `prop_firm_sizer.py` if your layout differs:

```python
from afml.bet_sizing.bet_sizing import bet_size_probability
from afml.bet_sizing.ch10_snippets import (
    bet_size_sigmoid, bet_size_power,
    discrete_signal, get_w_sigmoid, get_w_power,
)
```

Additional requirements:

```
numpy
pandas
scipy
numba       # used by bet_sizing and ch10_snippets
```

---

## Quick Start

### 1. Create an account state

```python
from prop_firm_sizer import PropFirmAccountState, Phase

state = PropFirmAccountState(
    initial_balance=100_000.0,
    phase=Phase.CHALLENGE_PHASE_1,
)
```

### 2. Create the sizer

```python
from prop_firm_sizer import make_stellar_2step_sizer

sizer = make_stellar_2step_sizer(
    stop_loss_pct=0.01,       # 1% account risk per trade
    safety_factor=0.70,       # use 70% of available budget
    avg_win_loss_ratio=1.2,   # from your live trade log
    kelly_fraction=0.5,       # half-Kelly
    func='sigmoid',
    step_size=0.05,
)
```

### 3. Check w_param sensitivity across drawdown scenarios

```python
print(sizer.w_param_sensitivity_report(initial_balance=100_000.0).to_string())
```

### 4. Size a batch of signals

```python
result = sizer.size(
    events=events_df,           # DataFrame with 't1' column
    prob=calibrated_probs,      # pd.Series from CVIsotonicCalibrator
    pred=primary_model_sides,   # pd.Series of +1 / -1
    state=state,
    news_times=news_event_list, # list of datetime, optional
    current_time=datetime.utcnow(),
    average_active=True,        # recommended for overlapping labels
)

position_sizes = result['final_size']
```

### 5. Update state after each trade

```python
state.update(
    realized_pnl_delta=closed_trade_pnl,
    unrealized_pnl=total_floating_pnl,
    fees_and_swaps=swap_and_commission,
)

# At midnight server time:
state.reset_daily()

# When phase target confirmed:
state.advance_phase()
```

---

## Detailed Parameter Guide

### `stop_loss_pct` — The Most Important Parameter

This is your strategy's per-trade stop loss expressed as a fraction of
account equity. It directly determines how large positions can be.

```
max_position = (risk_budget × safety_factor) / stop_loss_pct
```

If your strategy risks 0.5% per trade on a fresh account with a 70%
safety factor:

```
max_position = (0.05 × 0.70) / 0.005 = 7.0 → capped at 1.0 (full size)
```

If risk_budget shrinks to 1%:

```
max_position = (0.01 × 0.70) / 0.005 = 1.4 → capped at 1.0
```

If risk_budget shrinks to 0.3%:

```
max_position = (0.003 × 0.70) / 0.005 = 0.42 → actual constraint bites
```

Calibrate `stop_loss_pct` from your strategy's actual stop-loss width, not
an aspirational value. An underestimated stop causes chronic over-sizing
relative to the actual risk; an overestimated stop causes under-sizing.

---

### `safety_factor` — Buffer for Frictions

FundedNext's $5/lot commission and any overnight swap charges count toward
the daily drawdown. The `safety_factor` (default 0.70) ensures that even
after paying these frictions, you never accidentally touch the daily limit.

As a rule of thumb:
- High-frequency (many trades/day): 0.60–0.65
- Medium frequency (5–15 trades/day): 0.70
- Low frequency (1–5 trades/day): 0.75–0.80

---

### `avg_win_loss_ratio` — Update Monthly

The Kelly multiplier depends on this. Set it from your actual trade log:

```python
b = avg_win_usd / avg_loss_usd
```

Do not estimate this from the backtest if your live payoff ratio differs.
A common source of divergence is slippage on stop-loss exits (your actual
losses are larger than the backtest) while wins are filled accurately.

---

### `cal_divergence` — Calibration Point

The w_param is computed so that when the ML signal reaches `cal_divergence`
of its maximum strength, the position size equals `cal_bet_size`. The default
of 0.90 means "at 90% maximum signal, deploy the full budget."

Lower values (0.70–0.80) make the sizer more aggressive — maximum budget is
reached at lower signal strength. Higher values (0.95–0.99) reserve maximum
budget for only the very strongest signals.

---

### `step_size` — Discretisation

`discrete_signal` rounds positions to a grid of this granularity, preventing
micro-adjustments that generate commission without adding meaningful edge.

```
step_size=0.05  →  positions rounded to: 0, ±0.05, ±0.10, ..., ±1.0
```

For a $100,000 account trading standard lots, 0.05 represents 0.05 lots
increments, which is the natural minimum for most brokers.

---

## Integration with the AFML Pipeline

This sizer sits at the end of the pipeline defined across the other modules
in this project:

```
Raw price bars
    │
    ▼
data_structures/  (dollar bars, volume bars)
    │
    ▼
labeling/         (triple-barrier labelling → events + t1)
    │
    ▼
sample_weights/   (uniqueness weights × time-decay)
    │
    ▼
features/         (fractional differentiation, microstructure)
    │
    ▼
nested_cv.py      (UnifiedValidationCalibrator → calibrated prob)
    │
    ▼
prop_firm_sizer.py  ← YOU ARE HERE
    │
    ▼
mt5/ or production/  (execution layer)
```

The calibrated probabilities from `CVIsotonicCalibrator.predict_proba()`
feed directly into `sizer.size(prob=..., pred=...)`. The `events` DataFrame
with its `t1` column comes from the labeling module. The `pred` series is
the primary model's direction prediction (+1 / −1).

---

## Live Trading Loop Template

```python
from datetime import datetime
from prop_firm_sizer import (
    PropFirmAccountState, Phase,
    make_stellar_2step_sizer,
)

# ── One-time setup ────────────────────────────────────────────────────────

state = PropFirmAccountState(
    initial_balance=100_000.0,
    phase=Phase.CHALLENGE_PHASE_1,
)

sizer = make_stellar_2step_sizer(
    stop_loss_pct=0.01,
    safety_factor=0.70,
    avg_win_loss_ratio=1.2,
    kelly_fraction=0.5,
)

# ── On every new bar ──────────────────────────────────────────────────────

def on_new_bar(
    bar_time,
    events_df,
    calibrated_probs,
    primary_sides,
    closed_trade_pnl,
    total_floating_pnl,
    swap_charges,
    is_daily_reset,
    phase_target_hit,
    news_events_today,
):
    if is_daily_reset:
        state.reset_daily()

    if phase_target_hit:
        state.advance_phase()

    state.update(
        realized_pnl_delta=closed_trade_pnl,
        unrealized_pnl=total_floating_pnl,
        fees_and_swaps=swap_charges,
    )

    result = sizer.size(
        events=events_df,
        prob=calibrated_probs,
        pred=primary_sides,
        state=state,
        news_times=news_events_today,
        current_time=bar_time,
        average_active=True,
    )

    return result['final_size'], result[['w_param', 'cal_bet_size',
                                         'risk_budget_pct', 'daily_used_pct']]
```

---

## Design Decisions and Tradeoffs

### Why w_param and not direct position scaling?

Directly multiplying position size by `(remaining_budget / total_budget)`
would produce a linear response. The w_param approach preserves the shape
of the sigmoid — the nonlinear mapping from signal strength to position —
while compressing its amplitude. Weak signals get squeezed more than strong
ones, which is the correct behaviour: as budget shrinks, only the highest-
conviction signals should produce any meaningful size.

### Why Kelly as a multiplier and not a standalone sizer?

`get_signal` handles the concurrent label averaging via `avg_active_signals`,
which Kelly cannot do without explicit simulation of the full position
portfolio. Using Kelly as a multiplier preserves that averaging while
adding payoff-ratio awareness on top.

### Why the derisking factor starts at 0.80 of target?

At 80% phase progress, you have a large enough profit cushion that a bad
day cannot push you below breakeven for the phase. Reducing size from this
point forward protects that cushion at the cost of slightly slower target
achievement. The 0.30 floor at 100% progress allows continued trading
(to satisfy the minimum trading days rule) without material drawdown risk.

### Why the news factor is 0.40 not 0.0?

Trading through news windows is not prohibited — only the profit credit is
reduced to 40%. If the model has genuine edge during news (some do, some
do not), a 40%-scaled position still captures that edge. Setting it to 0.0
would unnecessarily forgo edge during news events.

---

## Limitations and Known Gaps

**Overnight gap risk.** The module assumes positions can be exited at the
stop-loss price. Gap opens can cause losses exceeding `stop_loss_pct`.
Add a hard position-size cap (e.g. 0.02 lots per trade on a $100k account)
as a separate safeguard against gap risk.

**Correlation between concurrent positions.** `avg_active_signals` correctly
averages overlapping signals, but the budget constraint is applied at the
portfolio level rather than the per-signal level. Highly correlated concurrent
positions can still cause concentrated exposure.

**`avg_win_loss_ratio` estimation lag.** The Kelly multiplier uses a static
estimate of `b`. In live trading, `b` drifts with market regime. Consider
using a rolling 60-trade estimate updated weekly.

**News event list.** The module accepts a list of news event datetimes but
does not source them automatically. You must provide these from an economic
calendar feed (e.g. Forex Factory, MT5 calendar API, or Investing.com).

**Phase transition detection.** `advance_phase()` must be called by the
user when the phase target is confirmed. The module does not auto-detect
this because FundedNext may have a confirmation delay.

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Chapter 10 (Bet Sizing).
- Kelly, J. L. (1956). A New Interpretation of Information Rate.
  *Bell System Technical Journal*, 35(4), 917–926.
- FundedNext Help Center — Daily Drawdown:
  https://help.fundednext.com/en/articles/8019811
- FundedNext Help Center — Maximum Loss Limit:
  https://help.fundednext.com/en/articles/8019812
- FundedNext Help Center — Stellar 2-Step Rules:
  https://help.fundednext.com/en/articles/8021076
- Bailey, D. H. & Lopez de Prado, M. (2014).
  The Probability of Backtest Overfitting.
- Murimi Njoroge, P. (2026). Unified Validation Pipeline Against
  Backtest Overfitting. MQL5 Articles.
  https://www.mql5.com/en/articles/21603
