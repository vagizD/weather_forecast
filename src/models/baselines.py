"""
Classical and Statistical Baseline Forecasters:
1. ClimatologyBase: Zero-parameter empirical historical average by (DOY, Hour).
2. ClimatologyRidgeForecaster: Direct multi-output L2-regularized linear model on cutoff atmospheric state.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data.preprocessing import compute_historical_climatology

class ClimatologyBase:
    """Zero-parameter empirical climatological baseline."""
    def __init__(self):
        self.clim_lookup = {}

    def fit(self, df_train: pd.DataFrame, target_col: str = "target_tmpf"):
        clim_df = compute_historical_climatology(df_train, target_col=target_col)
        self.clim_lookup = clim_df.set_index(["doy", "hour"])["clim_mean"].to_dict()

    def predict(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        return np.array([
            self.clim_lookup.get((t.dayofyear, t.hour), 60.0)
            for t in timestamps
        ], dtype=np.float32)

class ClimatologyRidgeForecaster:
    """Direct multi-output Ridge regressor predicting 336-hour residuals from cutoff state."""
    def __init__(self, alpha: float = 50.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.clim_df = None
        self.clim_lookup = {}
        self.horizon_len = 336

    def extract_state_features(self, df_window: pd.DataFrame, clim_lookup: Dict) -> np.ndarray:
        t0_row = df_window.iloc[-1]
        t24_row = df_window.iloc[-25] if len(df_window) >= 25 else df_window.iloc[0]
        t48_row = df_window.iloc[-49] if len(df_window) >= 49 else df_window.iloc[0]
        
        t0 = t0_row["target_tmpf"]
        dp0 = t0_row.get("dew_point_2m_f", t0)
        p0 = t0_row.get("surface_pressure", 1013.25)
        w0 = t0_row.get("wind_speed_10m", 5.0)
        
        delta_24 = t0 - t24_row["target_tmpf"]
        delta_48 = t0 - t48_row["target_tmpf"]
        
        recent_72 = df_window.iloc[-72:] if len(df_window) >= 72 else df_window
        anomalies = []
        for _, row in recent_72.iterrows():
            doy = row["time"].dayofyear
            hr = row["time"].hour
            c = clim_lookup.get((doy, hr), t0)
            anomalies.append(row["target_tmpf"] - c)
        mean_anomaly_72 = float(np.mean(anomalies)) if anomalies else 0.0
        
        cutoff_time = pd.to_datetime(t0_row["time"])
        hr = cutoff_time.hour
        doy = cutoff_time.dayofyear
        
        features = [
            t0,
            dp0,
            p0,
            w0,
            delta_24,
            delta_48,
            mean_anomaly_72,
            np.sin(2 * np.pi * hr / 24.0),
            np.cos(2 * np.pi * hr / 24.0),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25)
        ]
        return np.array(features, dtype=np.float32)

    def fit(self, df_train: pd.DataFrame, history_len: int = 168, horizon_len: int = 336, step: int = 24):
        self.horizon_len = horizon_len
        self.clim_df = compute_historical_climatology(df_train, target_col="target_tmpf")
        self.clim_lookup = self.clim_df.set_index(["doy", "hour"])["clim_mean"].to_dict()
        
        X_list = []
        Y_res_list = []
        
        window_size = history_len + horizon_len
        total_len = len(df_train)
        
        for i in range(0, total_len - window_size + 1, step):
            hist_df = df_train.iloc[i : i + history_len]
            fut_df = df_train.iloc[i + history_len : i + window_size]
            
            x_feat = self.extract_state_features(hist_df, self.clim_lookup)
            y_true = fut_df["target_tmpf"].values
            y_clim = np.array([
                self.clim_lookup.get((t.dayofyear, t.hour), y_true[0])
                for t in pd.to_datetime(fut_df["time"])
            ])
            y_res = y_true - y_clim
            
            X_list.append(x_feat)
            Y_res_list.append(y_res)
            
        X = np.array(X_list)
        Y_res = np.array(Y_res_list)
        
        self.model.fit(X, Y_res)

    def predict(self, df_cutoff_history: pd.DataFrame, target_timestamps: pd.DatetimeIndex) -> np.ndarray:
        x_state = self.extract_state_features(df_cutoff_history, self.clim_lookup).reshape(1, -1)
        pred_res = self.model.predict(x_state).flatten()
        
        y_clim = np.array([
            self.clim_lookup.get((t.dayofyear, t.hour), 60.0)
            for t in target_timestamps
        ])
        return y_clim + pred_res
