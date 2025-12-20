from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def find_slope(lookback: int, x: np.ndarray) -> float:
    """Compute linear slope (trend) - Python version of C++ function"""
    pptr = x[-lookback:]  # Window starts here
    slope = 0.0
    denom = 0.0

    for i in range(lookback):
        coef = i - 0.5 * (lookback - 1)
        denom += coef * coef
        slope += coef * pptr[i]

    return slope / denom


def atr(lookback: int, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Compute average true range - Python version"""
    high_window = high[-lookback:]
    low_window = low[-lookback:]
    close_window = close[-lookback:]

    sum_tr = 0.0
    for i in range(lookback):
        term = high_window[i] - low_window[i]
        if i > 0:  # There is a prior bar
            term = max(term, high_window[i] - close_window[i - 1])
            term = max(term, close_window[i - 1] - low_window[i])
        sum_tr += term

    return sum_tr / lookback


def gap_analyze(x: np.ndarray, thresh: float, gap_sizes: List[int]) -> Dict[int, int]:
    """Perform gap analysis"""
    ngaps = len(gap_sizes) + 1
    gap_count = {i: 0 for i in range(ngaps)}

    if len(x) == 0:
        return gap_count

    count = 1
    above_below = 1 if x[0] >= thresh else 0

    for i in range(1, len(x) + 1):
        if i == len(x):
            new_above_below = 1 - above_below
        else:
            new_above_below = 1 if x[i] >= thresh else 0

        if new_above_below == above_below:
            count += 1
        else:
            # Find the appropriate gap category
            j = 0
            while j < len(gap_sizes) and count > gap_sizes[j]:
                j += 1
            gap_count[j] += 1
            count = 1
            above_below = new_above_below

    return gap_count


class StationarityTest:
    """Python implementation of STATN.CPP functionality"""

    def __init__(self, lookback: int, fractile: float, version: int):
        self.lookback = lookback
        self.fractile = fractile
        self.version = version

        if version == 0:
            self.full_lookback = lookback
        elif version == 1:
            self.full_lookback = 2 * lookback
        else:
            self.full_lookback = version * lookback

    def load_market_data(self, filename: str) -> pd.DataFrame:
        """Load market data from file (YYYYMMDD O H L C)"""
        data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                date_str = parts[0]
                if len(date_str) != 8 or not date_str.isdigit():
                    continue

                try:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])

                    # Convert to datetime
                    date = f"{year}-{month:02d}-{day:02d}"

                    open_price = float(parts[1])
                    high_price = float(parts[2])
                    low_price = float(parts[3])
                    close_price = float(parts[4])

                    # Apply log transformation like C++ code
                    data.append(
                        {
                            "date": date,
                            "open": np.log(open_price),
                            "high": np.log(high_price),
                            "low": np.log(low_price),
                            "close": np.log(close_price),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)

    def compute_trend_indicators(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """Compute trend indicators and perform gap analysis"""
        close_prices = df["close"].values
        nind = len(close_prices) - self.full_lookback + 1

        if nind <= 0:
            return np.array([]), {}

        trend = np.zeros(nind)

        for i in range(nind):
            k = self.full_lookback - 1 + i
            current_window = close_prices[k - self.lookback + 1 : k + 1]

            if self.version == 0:
                trend[i] = find_slope(self.lookback, current_window)
            elif self.version == 1:
                # Current minus prior
                current_trend = find_slope(self.lookback, current_window)
                prior_window = close_prices[
                    k - 2 * self.lookback + 1 : k - self.lookback + 1
                ]
                prior_trend = find_slope(self.lookback, prior_window)
                trend[i] = current_trend - prior_trend
            else:
                # Current minus longer lookback
                current_trend = find_slope(self.lookback, current_window)
                longer_window = close_prices[k - self.full_lookback + 1 : k + 1]
                longer_trend = find_slope(self.full_lookback, longer_window)
                trend[i] = current_trend - longer_trend

        # Sort for quantile calculation
        trend_sorted = np.sort(trend)
        k_idx = int(self.fractile * (nind + 1)) - 1
        k_idx = max(0, k_idx)
        trend_quantile = trend_sorted[k_idx]

        # Gap analysis
        gap_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # Like NGAPS=11
        gap_results = gap_analyze(trend, trend_quantile, gap_sizes)

        trend_stats = {
            "min": np.min(trend),
            "max": np.max(trend),
            "quantile": trend_quantile,
            "gap_analysis": gap_results,
        }

        return trend, trend_stats

    def compute_volatility_indicators(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, dict]:
        """Compute volatility indicators and perform gap analysis"""
        high_prices = df["high"].values
        low_prices = df["low"].values
        close_prices = df["close"].values
        nind = len(high_prices) - self.full_lookback + 1

        if nind <= 0:
            return np.array([]), {}

        volatility = np.zeros(nind)

        for i in range(nind):
            k = self.full_lookback - 1 + i

            if self.version == 0:
                volatility[i] = atr(
                    self.lookback,
                    high_prices[k - self.lookback + 1 : k + 1],
                    low_prices[k - self.lookback + 1 : k + 1],
                    close_prices[k - self.lookback + 1 : k + 1],
                )
            elif self.version == 1:
                current_atr = atr(
                    self.lookback,
                    high_prices[k - self.lookback + 1 : k + 1],
                    low_prices[k - self.lookback + 1 : k + 1],
                    close_prices[k - self.lookback + 1 : k + 1],
                )
                prior_atr = atr(
                    self.lookback,
                    high_prices[k - 2 * self.lookback + 1 : k - self.lookback + 1],
                    low_prices[k - 2 * self.lookback + 1 : k - self.lookback + 1],
                    close_prices[k - 2 * self.lookback + 1 : k - self.lookback + 1],
                )
                volatility[i] = current_atr - prior_atr
            else:
                current_atr = atr(
                    self.lookback,
                    high_prices[k - self.lookback + 1 : k + 1],
                    low_prices[k - self.lookback + 1 : k + 1],
                    close_prices[k - self.lookback + 1 : k + 1],
                )
                longer_atr = atr(
                    self.full_lookback,
                    high_prices[k - self.full_lookback + 1 : k + 1],
                    low_prices[k - self.full_lookback + 1 : k + 1],
                    close_prices[k - self.full_lookback + 1 : k + 1],
                )
                volatility[i] = current_atr - longer_atr

        # Sort for quantile calculation
        volatility_sorted = np.sort(volatility)
        k_idx = int(self.fractile * (nind + 1)) - 1
        k_idx = max(0, k_idx)
        volatility_quantile = volatility_sorted[k_idx]

        # Gap analysis
        gap_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        gap_results = gap_analyze(volatility, volatility_quantile, gap_sizes)

        volatility_stats = {
            "min": np.min(volatility),
            "max": np.max(volatility),
            "quantile": volatility_quantile,
            "gap_analysis": gap_results,
        }

        return volatility, volatility_stats


# Usage example
def run_stationarity_test(lookback: int, fractile: float, version: int, filename: str):
    """Main function to run stationarity test"""
    tester = StationarityTest(lookback, fractile, version)

    # Load data
    df = tester.load_market_data(filename)
    print(f"Market price history read ({len(df)} lines)")
    print(f"Indicator version {version}")

    # Compute trend indicators
    trend, trend_stats = tester.compute_trend_indicators(df)
    if len(trend) > 0:
        print(
            f"\nTrend min={trend_stats['min']:.4f} max={trend_stats['max']:.4f} "
            f"{fractile:.3f} quantile={trend_stats['quantile']:.4f}"
        )

        print(f"\nGap analysis for trend with lookback={lookback}")
        print("  Size   Count")
        for size, count in trend_stats["gap_analysis"].items():
            if size < len(trend_stats["gap_analysis"]) - 1:
                print(f" {size + 1:5d} {count:7d}")
            else:
                print(f">{size:5d} {count:7d}")

    # Compute volatility indicators
    volatility, volatility_stats = tester.compute_volatility_indicators(df)
    if len(volatility) > 0:
        print(
            f"\nVolatility min={volatility_stats['min']:.4f} max={volatility_stats['max']:.4f} "
            f"{fractile:.3f} quantile={volatility_stats['quantile']:.4f}"
        )

        print(f"\nGap analysis for volatility with lookback={lookback}")
        print("  Size   Count")
        for size, count in volatility_stats["gap_analysis"].items():
            if size < len(volatility_stats["gap_analysis"]) - 1:
                print(f" {size + 1:5d} {count:7d}")
            else:
                print(f">{size:5d} {count:7d}")
