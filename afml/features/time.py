import numpy as np
import pandas as pd


def trading_session_encoded_features(
    datetime_index: pd.DatetimeIndex,
    n_terms: int = 3,  # Number of Fourier harmonics to include for the hour feature
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
        n_terms: Number of Fourier harmonics to include for the hour feature within each session.
                 For k=1, you get sin(2*pi*hour/24) and cos(2*pi*hour/24).
                 For k=2, you get sin(2*pi*2*hour/24) and cos(2*pi*2*hour/24), and so on.

    Returns:
        DataFrame with session flags (int8) and encoded hour features (float64).
    """
    # Handle empty input
    if datetime_index.empty:
        # Define columns for an empty DataFrame to match expected output structure
        base_cols = ["is_Sydney", "is_Tokyo", "is_London", "is_New_York"]
        encoded_cols = []
        session_names = ["Sydney", "Tokyo", "London", "New_York"]
        for session_name in session_names:
            for k in range(1, n_terms + 1):
                encoded_cols.append(f"{session_name}_hour_sin_h{k}")
                encoded_cols.append(f"{session_name}_hour_cos_h{k}")
        return pd.DataFrame(columns=base_cols + encoded_cols)

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

    # Pre-calculate all hour harmonics for the full 24-hour cycle
    hour_cycle_length = 24
    hour_sin_terms = {}
    hour_cos_terms = {}
    for k in range(1, n_terms + 1):
        radians_k = 2 * np.pi * k * hours / hour_cycle_length
        hour_sin_terms[k] = np.sin(radians_k)
        hour_cos_terms[k] = np.cos(radians_k)

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
        out[f"is_{session_name}"] = is_session.astype("int8")

        # Create encoded hour features for the active session
        # These features will be 0 when the session is inactive,
        # and represent the cyclical hour when active.
        for k in range(1, n_terms + 1):
            out[f"{session_name}_hour_sin_h{k}"] = hour_sin_terms[k] * is_session
            out[f"{session_name}_hour_cos_h{k}"] = hour_cos_terms[k] * is_session

    return out


def encode_cyclical_features(
    df: pd.DataFrame,
    dt_col: str = None,
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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Handle datetime source
    if dt_col:
        dt_series = pd.to_datetime(df[dt_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series()
    else:
        raise TypeError("Must provide dt_col or have a DatetimeIndex")

    out = pd.DataFrame(index=df.index)

    # Feature configuration
    features = {
        "minute": (dt_series.dt.minute, 60),
        "hour": (dt_series.dt.hour, 24),
        "dayofweek": (dt_series.dt.dayofweek, 7),
        "dayofyear": (dt_series.dt.dayofyear, 366),
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
    df: pd.DataFrame, timeframe: str, n_terms: int = 3, bar_type: str = "time"
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

    Returns:
        DataFrame with added time features
    """
    # Validate input
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    df = df.copy()

    # Bar duration features for non-time bars
    if bar_type != "time":
        durations = df.index.to_series().diff().dt.total_seconds()
        df["bar_duration"] = durations
        df["bar_duration_accel"] = durations.diff()

    # Frequency-based feature optimization
    timeframe = timeframe.upper()
    if timeframe.startswith(("B", "D", "W", "MN")):
        extra_features = []
    elif timeframe.startswith("H"):
        extra_features = []
    elif timeframe.startswith("M"):
        extra_features = ["minute"]

    # Generate features
    cyclical_feat = encode_cyclical_features(
        df, n_terms=n_terms, extra_fourier_features=extra_features
    )
    session_feat = trading_session_encoded_features(df.index, n_terms)
    to_drop = (
        session_feat.columns[session_feat.columns.str.startswith("is_")].to_list()
        + cyclical_feat.columns[cyclical_feat.columns.str.startswith("hour_")].to_list()
    )
    df = pd.concat([cyclical_feat, session_feat], axis=1).drop(columns=to_drop)
    df.columns = df.columns.str.lower()

    return df
