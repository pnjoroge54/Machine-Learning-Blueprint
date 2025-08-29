import numpy as np
import pandas as pd
import ruptures as rpt
from matplotlib import pyplot as plt


def identify_structural_breaks(df, df0, pen=30, verbose=False):
    """
    params df: (pd.DataFrame) Small window raw prices
    params df0: (pd.DataFrame) Large window raw prices
    params pen: (int) Penalty factor applied to ruptures.Pelt model
    return: structural_breaks, breakpoint_dates, regime_features
    """
    structural_breaks = pd.Series(0, index=df.index, name="regimes", dtype="int8")
    algo = rpt.Pelt(model="rbf").fit(np.log(df0.close).values)
    breakpoints = algo.predict(pen=pen)
    breakpoint_dates = df0.index[breakpoints[:-1]]
    idxs = df.index.searchsorted(breakpoint_dates) + 1
    N = len(idxs)

    # Create structural breaks as before
    for i, ix1 in enumerate(idxs):
        if i == 0:
            structural_breaks.iloc[:ix1] = 0
        else:
            ix0 = idxs[i - 1]
            structural_breaks.iloc[ix0:ix1] = i
            if i == N - 1:
                structural_breaks.iloc[ix1:] = N

    # Add regime duration features
    regime_features = create_regime_duration_features(structural_breaks, df.index)

    if verbose:
        print(f"Breakpoint Dates:")
        for i, date in enumerate(breakpoint_dates.strftime("%d-%m-%Y"), 1):
            print(f"\t{i:2d}. {date}")

    return structural_breaks, breakpoint_dates, regime_features


def create_regime_duration_features(structural_breaks, index):
    """
    Create regime duration and related features

    params structural_breaks: (pd.Series) Regime labels for each observation
    params index: (pd.Index) Time index for the data
    return: (pd.DataFrame) Regime duration features
    """
    regime_features = pd.DataFrame(index=index)

    # Calculate regime changes
    regime_changes = structural_breaks != structural_breaks.shift(1)
    regime_change_indices = np.where(regime_changes)[0]

    # Vectorized calculation of days since regime change using forward fill
    change_dates = pd.Series(index=index, dtype="datetime64[ns]")
    change_dates.iloc[regime_change_indices] = index[regime_change_indices]
    change_dates = change_dates.fillna(method="ffill")

    # Calculate days since change vectorized
    regime_features["days_since_regime_change"] = (index - change_dates).dt.days

    # Use groupby for efficient regime processing
    regime_groups = structural_breaks.groupby(structural_breaks)

    # Initialize feature columns
    regime_features["previous_regime_duration"] = 0
    regime_features["regime_age_normalized"] = 0.0
    regime_features["expected_regime_remaining"] = 0

    # Process each regime group
    for regime, group in regime_groups:
        regime_mask = structural_breaks == regime

        # Find regime periods using groupby on consecutive indices
        regime_indices = group.index
        regime_start_indices = regime_indices[regime_changes[regime_indices]]

        # Calculate durations for each regime period
        regime_durations = []
        for start_idx in regime_start_indices:
            start_pos = index.get_loc(start_idx)
            # Find next regime change after this start
            future_changes = regime_change_indices[regime_change_indices > start_pos]
            if len(future_changes) > 0:
                end_pos = future_changes[0]
                end_date = index[end_pos]
                duration = (end_date - start_idx).days
                regime_durations.append(duration)

                # Set previous regime duration for all observations after this regime ended
                post_regime_mask = index > end_date
                regime_features.loc[post_regime_mask, "previous_regime_duration"] = (
                    duration
                )

        # Calculate typical duration for this regime type
        if regime_durations:
            typical_duration = np.mean(regime_durations)
        else:
            typical_duration = 30  # Default assumption

        # Vectorized assignment of regime-specific features
        current_days = regime_features.loc[regime_mask, "days_since_regime_change"]

        regime_features.loc[regime_mask, "regime_age_normalized"] = (
            current_days / typical_duration
        )
        regime_features.loc[regime_mask, "expected_regime_remaining"] = np.maximum(
            0, typical_duration - current_days
        )

    # Vectorized maturity calculation
    regime_features["regime_maturity"] = np.clip(
        regime_features["days_since_regime_change"] / 30, 0, 1
    )

    return regime_features


def combine_regime_features(structural_breaks, regime_features):
    """
    Combine all regime-related features into a single DataFrame

    params structural_breaks: (pd.Series) Regime labels
    params regime_features: (pd.DataFrame) Duration-based features
    return: (pd.DataFrame) Combined features
    """
    # One-hot encode regimes
    regime_dummies = pd.get_dummies(structural_breaks, prefix="regime", dtype="int8")

    # Combine all features
    combined_features = pd.concat([regime_dummies, regime_features], axis=1)
    return combined_features


def plot_structural_breaks(df, breakpoint_dates, symbol, pen, timeframe="D1"):
    plt.figure(figsize=(12, 5))
    plt.plot(
        df.index.values, df.close.values, label="Close", c="blue"
    )  # Plot data with dates on x-axis
    for bp_date in breakpoint_dates:
        plt.axvline(bp_date, color="red", linestyle="--", label=f"{bp_date.date()}")

    plt.ylabel("Close")
    plt.title(f"{symbol}_{timeframe} Breakpoints (rbf pen={pen})")
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
