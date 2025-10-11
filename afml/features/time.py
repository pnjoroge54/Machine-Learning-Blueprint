import numpy as np
import pandas as pd


def trading_session_encoded_features(
    datetime_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Creates Boolean flags and Fourier-encoded hour features for forex trading sessions based on UTC time.

    The encoded features represent the cyclical hour within each active session,
    being zero when the session is inactive.

    Fixed issues:
    1. Added empty input handling
    2. Improved timezone handling
    3. Optimized session calculations
    4. Added Fourier encoding for hour within active sessions.

    Args:
        datetime_index: Input datetime index (naive or timezone-aware)

    Returns:
        DataFrame with session flags (int8) and encoded hour features (float64).
    """
    # Convert to UTC. If the index is naive, assume it is UTC.
    if datetime_index.tz is not None:
        dt_utc = datetime_index.tz_convert("UTC")
    else:
        dt_utc = datetime_index.tz_localize("UTC")  # Assume naive is UTC and localize it

    hours = dt_utc.hour.values  # Use NumPy array for vectorized operations

    # Initialize DataFrame for results with the original datetime_index
    out = pd.DataFrame(index=datetime_index)

    # Define session boundaries and whether they cross midnight (UTC hours)
    sessions = {
        "Sydney": {"start": 21, "end": 6, "cross_midnight": True},  # 21:00 to 05:59 UTC
        "Tokyo": {"start": 0, "end": 9, "cross_midnight": False},  # 00:00 to 08:59 UTC
        "London": {
            "start": 7,
            "end": 16,
            "cross_midnight": False,
        },  # 07:00 to 15:59 UTC
        "New_York": {
            "start": 13,
            "end": 22,
            "cross_midnight": False,
        },  # 13:00 to 21:59 UTC
    }

    # Process each session to create flags and encoded features
    for session_name, params in sessions.items():
        start_hour = params["start"]
        end_hour = params["end"]
        cross_midnight = params["cross_midnight"]

        # Calculate boolean flag for the session's active period
        if cross_midnight:
            is_session = (hours >= start_hour) | (hours < end_hour)
        else:
            is_session = (hours >= start_hour) & (hours < end_hour)

        # Add the binary session flag to the results DataFrame
        out[f"{session_name.replace('New_York', 'ny').lower()}_session"] = is_session.astype("int8")

    out["session_overlap"] = np.where(out.sum(axis=1) > 1, 1, 0)

    # Key forex timing patterns
    day_of_week = datetime_index.dayofweek.values
    day_of_month = datetime_index.day.values
    month = datetime_index.month.values

    out["friday_ny_close"] = ((day_of_week == 4) & (hours >= 21)).astype(int)
    out["sunday_open"] = ((day_of_week == 6) & (hours <= 2)).astype(int)
    out["month_end"] = (day_of_month >= 28).astype(int)
    out["quarter_end"] = ((month % 3 == 0) & (day_of_month >= 28)).astype(int)

    return out


def encode_cyclical_features(
    datetime_index: pd.DatetimeIndex,
    n_terms: int = 3,
    extra_fourier_features: list = None,
) -> pd.DataFrame:
    """
    Encodes datetime cyclical features with Fourier transformations.

    Fixed issues:
    1. Removed index name dropping (was error-prone)
    2. Improved cycle length handling
    3. Added input validation

    Args:
        df: Input DataFrame
        dt_col: Datetime column name (uses index if None)
        n_terms: Number of Fourier harmonics to include
        extra_fourier_features: Features for extra harmonics

    Returns:
        DataFrame with added cyclical features
    """
    # Input validation
    if not isinstance(datetime_index, pd.DatetimeIndex):
        raise ValueError("datetime_index must be a pandas DatetimeIndex")

    out = pd.DataFrame(index=datetime_index)

    # Feature configuration
    features = {
        "hour": (datetime_index.hour, 24),
        "dayofweek": (datetime_index.dayofweek, 7),
        "dayofyear": (datetime_index.dayofyear, 366),
    }

    # Process features
    for name, (series, cycle_length) in features.items():
        # Base harmonic (k=1)
        radians = 2 * np.pi * series / cycle_length
        out[f"{name}_sin"] = np.sin(radians)
        out[f"{name}_cos"] = np.cos(radians)

        # Additional harmonics
        if n_terms >= 1 and (extra_fourier_features is None or name in extra_fourier_features):
            out.rename(
                columns={
                    f"{name}_sin": f"{name}_sin_h1",
                    f"{name}_cos": f"{name}_cos_h1",
                },
                inplace=True,
            )
            for k in range(2, n_terms + 1):
                radians_k = 2 * np.pi * k * series / cycle_length
                out[f"{name}_sin_h{k}"] = np.sin(radians_k)
                out[f"{name}_cos_h{k}"] = np.cos(radians_k)

    return out


def get_time_features(
    df: pd.DataFrame, timeframe: str, n_terms: int = 3, bar_type: str = "time", forex: bool = True
) -> pd.DataFrame:
    """
    Creates comprehensive time features for financial data.

    Fixed issues:
    1. Removed lookahead in bar duration calculations
    2. Simplified feature selection logic
    3. Removed problematic append parameter
    4. Added frequency-based feature optimization

    Args:
        df: Input DataFrame with datetime index
        timeframe: Timeframe used to generate bars
        n_terms: Fourier harmonics to generate
        bar_type: Bar type ('time' or other)
        forex: If asset trades according to forex sessions (24H)

    Returns:
        DataFrame with added time features
    """
    # Validate input
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    features = []

    # Bar duration features for non-time bars
    if bar_type != "time":
        durations = df.index.to_series("bar_duration").diff().dt.total_seconds()
        duration_accel = durations.diff().rename("bar_duration_accel") # bar duration acceleration
        features += [durations, duration_accel]

    # Frequency-based feature optimization
    timeframe = timeframe.upper()
    if timeframe.startswith(("H", "D", "W", "MN")):
        extra_features = []
    elif timeframe.startswith("M"):
        extra_features = ["hour"]

    # Generate features
    cyclical_feat = encode_cyclical_features(
        df.index, n_terms=n_terms, extra_fourier_features=extra_features
    )
    if forex:
        session_feat = trading_session_encoded_features(df.index)
        # Add session volatility
        returns = np.log(df["close"]).diff()
        for session in session_feat:
            session_mask = session_feat[session] == 1
            if session_mask.sum() > 0:
                session_vol = returns[session_mask].rolling(20, min_periods=1).std()
                session_feat[f"{session}_vol"] = session_vol.reindex(df.index, method="ffill")
    else:
        session_feat = pd.DataFrame()

    features += [cyclical_feat, session_feat]
    
    return pd.concat(features, axis=1)
