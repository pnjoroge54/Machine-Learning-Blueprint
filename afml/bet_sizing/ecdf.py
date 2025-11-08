import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# --- 1. ECDF Training Function ---


def train_ecdf_sizing(training_probabilities: pd.Series, training_outcomes: pd.Series) -> interp1d:
    """
    Trains the ECDF function required for position sizing based on
    historically profitable trade probabilities.

    The method filters training data to only include profitable trades
    (P > 0.5) and fits an empirical cumulative distribution function (CDF)
    to these probabilities.

    Args:
        training_probabilities: A pandas Series of probabilities (P) output
                                by the secondary model (M2) on the training set.
        training_outcomes: A pandas Series of the meta-label (1 for profitable,
                           0 for loss/neutral) corresponding to the outcomes.

    Returns:
        An interpolation function (scipy.interpolate.interp1d) that maps a
        probability score (P) to a fractional position size (0 to 1).
    """
    # 1. Filter for positive/profitable trades (meta-label = 1)
    # The ECDF is fitted only to probabilities that resulted in a positive outcome.
    # The general ECDF methodology also typically filters for P > 0.5.[2]
    profitable_probabilities = training_probabilities[training_outcomes == 1]

    if profitable_probabilities.empty:
        raise ValueError("No profitable trades found in the training data to fit the ECDF.")

    # 2. Sort the profitable probabilities
    # ECDF calculation requires the data to be sorted.
    sorted_probabilities = np.sort(profitable_probabilities)

    # 3. Calculate the Empirical Cumulative Distribution Function (ECDF)
    # The ECDF calculates the percentage of observations <= x.
    # y-axis (ECDF value) represents the percentile rank of the probability.
    N = len(sorted_probabilities)

    # We use y = [1/N, 2/N,..., 1] to represent the standard ECDF step function.
    ecdf_values = np.arange(1, N + 1) / N

    # 4. Create an interpolation function
    # The interpolation function allows us to map any new probability score (P_new)
    # to the corresponding position size (ECDF value/percentile rank).
    # We use 'linear' interpolation to map values between observed points.

    # Pad the interpolation points to ensure all possible P values (0 to 1) are covered.
    # The final function should map probabilities below the lowest observed P to 0,
    # and probabilities above the highest observed P to 1 (the size).

    # x-values (probability) for interpolation
    x_interp = np.concatenate(([0.0], sorted_probabilities, [1.0]))

    # y-values (position size / ECDF rank) for interpolation
    y_interp = np.concatenate(([0.0], ecdf_values, [1.0]))

    # The 'bounds_error=False' and 'fill_value=(0.0, 1.0)' ensures that
    # probabilities outside the observed training range are handled gracefully,
    # mapping to the appropriate min (0) or max (1) position size.
    ecdf_func = interp1d(
        x_interp, y_interp, kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
    )

    return ecdf_func


# --- 2. ECDF Application Function ---


def apply_ecdf_sizing(ecdf_func: interp1d, new_probability: float, threshold: float = 0.5) -> float:
    """
    Applies the trained ECDF function to a new probability score (P) to determine
    the fractional position size.

    Args:
        ecdf_func: The interpolation function returned by train_ecdf_sizing.
        new_probability: The predicted probability of a profitable trade for the
                         current signal (output of M2).
        threshold: The minimum probability required to enter a trade. Position size is 0
                   if P is below this threshold (default 0.5).[2]

    Returns:
        The fractional position size (0.0 to 1.0) to be allocated to the trade.
    """
    if new_probability <= threshold:
        # If the probability does not exceed the threshold (e.g., 50%), size is zero.[2]
        return 0.0

    # The position size is directly determined by the ECDF value, which represents
    # the percentile rank of the probability in the historical distribution of successful trades.
    # Higher rank (closer to 1.0) means higher conviction, leading to larger allocation.[1, 2]
    position_size = ecdf_func(new_probability).item()

    # Ensure the size is within valid bounds
    return np.clip(position_size, 0.0, 1.0)


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Simulate Training Data (M2 Probabilities and Meta-Labels)
    np.random.seed(42)
    # Assume 100 trade signals
    num_samples = 100
    # M2 probabilities (raw confidence scores)
    M2_probs_train = pd.Series(np.random.rand(num_samples) * 0.4 + 0.5)  # biased towards > 0.5
    # Meta-Labels (1=profitable, 0=loss/neutral)
    M2_outcomes_train = pd.Series(np.random.choice([1, 0], size=num_samples, p=[0.4, 0.6]))

    # 2. Train the ECDF Sizing Function
    try:
        ecdf_model = train_ecdf_sizing(M2_probs_train, M2_outcomes_train)
        print("ECDF Model Trained Successfully.")
    except ValueError as e:
        print(f"Error: {e}")
        ecdf_model = None

    if ecdf_model:
        # 3. Apply the ECDF function to new trade signals (Out-of-Sample)

        # Signal A: Low Confidence (P = 0.52) - expected size: 0 (or very small)
        prob_A = 0.52
        size_A = apply_ecdf_sizing(ecdf_model, prob_A)

        # Signal B: Medium Confidence (P = 0.70) - expected size: mid-range
        prob_B = 0.70
        size_B = apply_ecdf_sizing(ecdf_model, prob_B)

        # Signal C: High Confidence (P = 0.88) - expected size: large (closer to 1.0)
        prob_C = 0.88
        size_C = apply_ecdf_sizing(ecdf_model, prob_C)

        # Signal D: Below Threshold (P = 0.45) - expected size: 0
        prob_D = 0.45
        size_D = apply_ecdf_sizing(ecdf_model, prob_D)

        print("\n--- Position Sizing Results ---")
        print(f"Probability P={prob_A:.4f} -> Position Size: {size_A:.4f}")
        print(f"Probability P={prob_B:.4f} -> Position Size: {size_B:.4f}")
        print(f"Probability P={prob_C:.4f} -> Position Size: {size_C:.4f}")
        print(f"Probability P={prob_D:.4f} (Below 0.5 Threshold) -> Position Size: {size_D:.4f}")
