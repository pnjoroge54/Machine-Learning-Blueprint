from os import cpu_count
from pathlib import Path

from pytz import timezone

DATA_PATH = Path.home() / "tick_data_parquet"
CLEAN_DATA_PATH = Path.home() / "tick_data_clean_parquet"
OHLCV = ["open", "high", "low", "close", "volume"]
DATE_COMPONENTS = ["year", "month", "day", "hour", "minute", "second", "microsecond"]
UTC = timezone("UTC")
PERCENTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
NUM_THREADS = cpu_count() - 1
GREEKS = ["β", "γ", "ρ", "φ", "χ", "δ", "α", "σ", "λ", "μ", "τ", "θ", "ε", "ψ"]

# ------------ Symbol Groups ------------

FX_MAJORS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "USDCHF",
)

COMMODITIES = (
    "XAUUSD",
    "USOUSD",
    "UKOUSD",
)

CRYPTO = (
    "ADAUSD",
    "BTCUSD",
    "DOGUSD",
    "ETHUSD",
    "LNKUSD",
    "LTCUSD",
    "XLMUSD",
    "XMRUSD",
    "XRPUSD",
)
