"""
PyTorch Dataset classes for sliding historical training and rolling validation/test evaluation.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CONT_FEATURES, PT_FEATURES, SOLAR_FUTURE_COLS, HISTORY_LEN, HORIZON_LEN
from src.data.preprocessing import calculate_solar_features

def get_window_indices(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizon_hours: int = HORIZON_LEN,
    stride_hours: int = 24
) -> List[Dict]:
    """Generate list of window dictionaries for rolling evaluation."""
    windows = []
    curr = start_date
    while curr + pd.Timedelta(hours=horizon_hours) <= end_date + pd.Timedelta(hours=1):
        target_end = curr + pd.Timedelta(hours=horizon_hours)
        hist_start = curr - pd.Timedelta(hours=HISTORY_LEN)
        windows.append({
            "start": curr,
            "end": target_end,
            "hist_start": hist_start
        })
        curr += pd.Timedelta(hours=stride_hours)
    return windows

class FixedRollingDataset(Dataset):
    """Dataset for evaluating rolling windows with daily stride."""
    def __init__(
        self,
        df: pd.DataFrame,
        window_starts: List[pd.Timestamp],
        clim_lookup: Dict,
        history_len: int = HISTORY_LEN,
        horizon_len: int = HORIZON_LEN
    ):
        self.window_starts = window_starts
        self.history_len = history_len
        self.horizon_len = horizon_len
        
        self.cont_data = df[CONT_FEATURES].values.astype(np.float32)
        self.y_true_data = df["target_tmpf"].values.astype(np.float32)
        self.hour_arr = df["hour"].values.astype(np.int64)
        self.month_arr = df["month"].values.astype(np.int64)
        self.season_arr = df["season"].values.astype(np.int64)
        self.day_arr = df["day"].values.astype(np.int64)
        self.weekday_arr = df["weekday"].values.astype(np.int64)
        self.year_arr = df["year_idx"].values.astype(np.int64)
        self.future_solar_data = df[["solar_elevation", "extraterrestrial_flux"]].values.astype(np.float32)
        
        self.pt_data = df[PT_FEATURES].values.astype(np.float32)
        solar_df = calculate_solar_features(df["time"])
        self.pt_future_solar = solar_df[SOLAR_FUTURE_COLS].values.astype(np.float32)
        
        self.time_to_idx = {t: i for i, t in enumerate(df["time"])}
        
        doy_arr = df["time"].dt.dayofyear.values
        hour_arr = df["time"].dt.hour.values
        self.clim_arr = np.array([clim_lookup.get((d, h), 60.0) for d, h in zip(doy_arr, hour_arr)], dtype=np.float32)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx: int):
        start_time = self.window_starts[idx]
        split = self.time_to_idx[start_time]
        start = split - self.history_len
        end = split + self.horizon_len
        
        return {
            "x_cont": torch.from_numpy(self.cont_data[start:split]),
            "y_true": torch.from_numpy(self.y_true_data[split:end]),
            "y_clim": torch.from_numpy(self.clim_arr[split:end]),
            "future_solar": torch.from_numpy(self.future_solar_data[split:end]),
            "x_time": {
                "hour": torch.from_numpy(self.hour_arr[start:split]),
                "month": torch.from_numpy(self.month_arr[start:split]),
                "season": torch.from_numpy(self.season_arr[start:split]),
                "day": torch.from_numpy(self.day_arr[start:split]),
                "weekday": torch.from_numpy(self.weekday_arr[start:split]),
                "year": torch.from_numpy(self.year_arr[start:split])
            },
            "future_time": {
                "hour": torch.from_numpy(self.hour_arr[split:end]),
                "month": torch.from_numpy(self.month_arr[split:end]),
                "season": torch.from_numpy(self.season_arr[split:end]),
                "day": torch.from_numpy(self.day_arr[split:end]),
                "weekday": torch.from_numpy(self.weekday_arr[split:end]),
                "year": torch.from_numpy(self.year_arr[split:end])
            },
            "pt_hist": torch.from_numpy(self.pt_data[start:split]),
            "pt_solar_fut": torch.from_numpy(self.pt_future_solar[split:end]),
            "start_time_str": str(start_time)
        }

class SlidingTrainDataset(Dataset):
    """Training dataset over historical records with configurable step."""
    def __init__(
        self,
        df: pd.DataFrame,
        clim_lookup: Dict,
        history_len: int = HISTORY_LEN,
        horizon_len: int = HORIZON_LEN,
        step: int = 12
    ):
        self.history_len = history_len
        self.horizon_len = horizon_len
        
        total_len = len(df)
        window_size = history_len + horizon_len
        self.valid_indices = [i for i in range(0, total_len - window_size + 1, step)]
        
        self.cont_data = df[CONT_FEATURES].values.astype(np.float32)
        self.y_true_data = df["target_tmpf"].values.astype(np.float32)
        self.hour_arr = df["hour"].values.astype(np.int64)
        self.month_arr = df["month"].values.astype(np.int64)
        self.season_arr = df["season"].values.astype(np.int64)
        self.day_arr = df["day"].values.astype(np.int64)
        self.weekday_arr = df["weekday"].values.astype(np.int64)
        self.year_arr = df["year_idx"].values.astype(np.int64)
        self.future_solar_data = df[["solar_elevation", "extraterrestrial_flux"]].values.astype(np.float32)
        
        self.pt_data = df[PT_FEATURES].values.astype(np.float32)
        solar_df = calculate_solar_features(df["time"])
        self.pt_future_solar = solar_df[SOLAR_FUTURE_COLS].values.astype(np.float32)
        
        doy_arr = df["time"].dt.dayofyear.values
        hour_arr = df["time"].dt.hour.values
        self.clim_arr = np.array([clim_lookup.get((d, h), 60.0) for d, h in zip(doy_arr, hour_arr)], dtype=np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        start = self.valid_indices[idx]
        split = start + self.history_len
        end = split + self.horizon_len
        
        return {
            "x_cont": torch.from_numpy(self.cont_data[start:split]),
            "y_true": torch.from_numpy(self.y_true_data[split:end]),
            "y_clim": torch.from_numpy(self.clim_arr[split:end]),
            "future_solar": torch.from_numpy(self.future_solar_data[split:end]),
            "x_time": {
                "hour": torch.from_numpy(self.hour_arr[start:split]),
                "month": torch.from_numpy(self.month_arr[start:split]),
                "season": torch.from_numpy(self.season_arr[start:split]),
                "day": torch.from_numpy(self.day_arr[start:split]),
                "weekday": torch.from_numpy(self.weekday_arr[start:split]),
                "year": torch.from_numpy(self.year_arr[start:split])
            },
            "future_time": {
                "hour": torch.from_numpy(self.hour_arr[split:end]),
                "month": torch.from_numpy(self.month_arr[split:end]),
                "season": torch.from_numpy(self.season_arr[split:end]),
                "day": torch.from_numpy(self.day_arr[split:end]),
                "weekday": torch.from_numpy(self.weekday_arr[split:end]),
                "year": torch.from_numpy(self.year_arr[split:end])
            },
            "pt_hist": torch.from_numpy(self.pt_data[start:split]),
            "pt_solar_fut": torch.from_numpy(self.pt_future_solar[split:end])
        }
