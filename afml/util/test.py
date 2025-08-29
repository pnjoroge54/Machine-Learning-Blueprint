import warnings
import winsound
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import seaborn as sns

from afml.data_structures.bars import *
from afml.labeling import (
    fixed_time_horizon,
    get_bins,
    get_bins_from_trend,
    get_events,
    triple_barrier_labels,
)
from afml.strategies import (
    BaseStrategy,
    BollingerMeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    TripleBarrierEvaluator,
    get_entries,
)
from afml.util import (
    CLEAN_DATA_PATH,
    COMMODITIES,
    CRYPTO,
    DATA_PATH,
    FX_MAJORS,
    PERCENTILES,
    UTC,
    DataFrameFormatter,
    get_ticks,
    load_tick_data,
    login_mt5,
    save_data_to_parquet,
    value_counts_data,
    verify_or_create_account_info,
)

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
sns.set_palette("husl")
