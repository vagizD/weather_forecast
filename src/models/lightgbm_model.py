"""
Direct Multi-Step LightGBM Forecaster with Climatological Residuals.

Fits 14 daily/24 hourly models or direct multi-step estimators using lag features,
rolling statistics, and deterministic calendar features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data.preprocessing import compute_historical_climatology

class LightGBMWeatherForecaster:
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, num_leaves: int = 31):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model = None
        self.clim_lookup = None
        self.horizon_len = 336

    def extract_features(self, df_window: pd.DataFrame, clim_lookup: dict) -> np.ndarray:
        """
        Extract tabular features from pre-cutoff observation history:
        - Current state (T0, DewPoint, Pressure, Wind, Cloud)
        - Lags: T-1, T-2, T-3, T-6, T-12, T-24, T-48, T-72
        - Rolling means & stds: 24h, 72h, 168h
        - Anomalies relative to climatology
        - Harmonic calendar features
        """
        t0_row = df_window.iloc[-1]
        temps = df_window["target_tmpf"].values
        
        t0 = temps[-1]
        t1 = temps[-2] if len(temps) >= 2 else t0
        t2 = temps[-3] if len(temps) >= 3 else t0
        t6 = temps[-7] if len(temps) >= 7 else t0
        t12 = temps[-13] if len(temps) >= 13 else t0
        t24 = temps[-25] if len(temps) >= 25 else t0
        t48 = temps[-49] if len(temps) >= 49 else t0
        t72 = temps[-73] if len(temps) >= 73 else t0
        
        roll_24_mean = float(np.mean(temps[-24:]))
        roll_24_std = float(np.std(temps[-24:])) if len(temps) >= 24 else 0.0
        roll_72_mean = float(np.mean(temps[-72:])) if len(temps) >= 72 else roll_24_mean
        roll_72_std = float(np.std(temps[-72:])) if len(temps) >= 72 else roll_24_std
        
        dp0 = t0_row.get("dew_point_2m_f", t0)
        p0 = t0_row.get("surface_pressure", 1013.25)
        w0 = t0_row.get("wind_speed_10m", 5.0)
        c0 = t0_row.get("cloud_cover", 50.0)
        
        cutoff_time = pd.to_datetime(t0_row["time"])
        hr = cutoff_time.hour
        doy = cutoff_time.dayofyear
        
        # 72h mean anomaly
        recent_72 = df_window.iloc[-72:] if len(df_window) >= 72 else df_window
        anomalies = [
            row["target_tmpf"] - clim_lookup.get((row["time"].dayofyear, row["time"].hour), t0)
            for _, row in recent_72.iterrows()
        ]
        mean_anomaly = float(np.mean(anomalies)) if anomalies else 0.0
        
        feats = [
            t0, t1, t2, t6, t12, t24, t48, t72,
            roll_24_mean, roll_24_std, roll_72_mean, roll_72_std,
            dp0, p0, w0, c0, mean_anomaly,
            np.sin(2 * np.pi * hr / 24.0), np.cos(2 * np.pi * hr / 24.0),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        ]
        return np.array(feats, dtype=np.float32)

    def fit(self, df_train: pd.DataFrame, history_len: int = 168, horizon_len: int = 336, step: int = 24):
        """Fit MultiOutput LightGBM on historical windows."""
        self.horizon_len = horizon_len
        print(f"Fitting LightGBMWeatherForecaster (trees={self.n_estimators}, horizon={horizon_len})...")
        
        self.clim_df = compute_historical_climatology(df_train, target_col="target_tmpf")
        clim_lookup = self.clim_df.set_index(["doy", "hour"])["clim_mean"].to_dict()
        self.clim_lookup = clim_lookup
        
        X_list = []
        Y_res_list = []
        
        window_size = history_len + horizon_len
        total_len = len(df_train)
        
        for i in range(0, total_len - window_size + 1, step):
            hist_df = df_train.iloc[i : i + history_len]
            fut_df = df_train.iloc[i + history_len : i + window_size]
            
            x_feat = self.extract_features(hist_df, clim_lookup)
            
            y_true = fut_df["target_tmpf"].values
            y_clim = np.array([
                clim_lookup.get((t.dayofyear, t.hour), y_true[0])
                for t in fut_df["time"]
            ])
            y_res = y_true - y_clim
            
            X_list.append(x_feat)
            Y_res_list.append(y_res)
            
        X = np.vstack(X_list)
        Y_res = np.vstack(Y_res_list)
        
        base_lgb = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        self.model = MultiOutputRegressor(base_lgb, n_jobs=-1)
        self.model.fit(X, Y_res)
        print(f"Fitted MultiOutput LightGBM on {len(X)} windows.")

    def predict(self, df_history: pd.DataFrame, future_timestamps: pd.Series) -> np.ndarray:
        """Predict 336 hours using history up to cutoff."""
        x_feat = self.extract_features(df_history, self.clim_lookup).reshape(1, -1)
        pred_res = self.model.predict(x_feat).flatten()
        
        fut_dt = pd.to_datetime(future_timestamps)
        clim_future = np.array([
            self.clim_lookup.get((t.dayofyear, t.hour), df_history["target_tmpf"].iloc[-1])
            for t in fut_dt
        ])
        return clim_future + pred_res
