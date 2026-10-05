"""
Global configuration, feature lists, and dataset split constants.
"""

import os
import torch
import pandas as pd

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "krdu_modeling_2000_2026_pre_sep17.parquet")
LOCKED_TEST_PATH = os.path.join(DATA_DIR, "test", "supertest_ground_truth_locked.csv")
REPORT_DIR = os.path.join(BASE_DIR, "report")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chronological Partition Dates (Fixed Setup)
TRAIN_START = pd.Timestamp("2000-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2026-04-01 00:00:00")       # 26.25 years (230,088 continuous hours)
VAL_END = pd.Timestamp("2026-07-01 00:00:00")         # 78 rolling 14-day spring windows
TEST_END = pd.Timestamp("2026-09-16 23:00:00")        # 65 rolling 14-day summer windows
LOCKED_START = pd.Timestamp("2026-09-17 00:00:00")    # Official competition window (336 hours)
LOCKED_END = pd.Timestamp("2026-10-01 00:00:00")

# Horizons
HISTORY_LEN = 720  # 30 days
HORIZON_LEN = 336  # 14 days
PATCH_LEN = 24     # 1 day diurnal patch
STRIDE = 12        # 12-hour overlap

# 14 Continuous Physical Features (for DenseNet & PLE)
CONT_FEATURES = [
    "target_tmpf",
    "dew_point_2m_f",
    "surface_pressure",
    "dp_3h",
    "dp_24h",
    "dt_24h",
    "u_wind",
    "v_wind",
    "vpd",
    "q_spec",
    "theta",
    "cloud_cover",
    "shortwave_radiation",
    "solar_elevation",
]

# 12 PatchTST Multivariate Input Features
PT_FEATURES = [
    "target_tmpf",
    "dew_point_2m_f",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "solar_elevation",
    "extraterrestrial_flux",
    "vpd",
    "theta",
    "u_wind",
    "v_wind",
]

# Future Conditioning Solar Features
SOLAR_FUTURE_COLS = [
    "solar_elevation",
    "extraterrestrial_flux",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
