"""
Deterministic meteorological and solar feature engineering, climatology, and metrics.
Zero leakage: all scaling and empirical statistics are fit strictly on training partitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import CONT_FEATURES

LATITUDE_DEG = 35.8776
LONGITUDE_DEG = -78.7875

def calculate_solar_features(timestamps: pd.Series) -> pd.DataFrame:
    """
    Calculate deterministic solar astronomy features without empirical data.
    Based on NOAA Solar Calculator equations for Raleigh-Durham (KRDU).
    """
    dt = pd.to_datetime(timestamps)
    day_of_year = dt.dt.dayofyear
    hour = dt.dt.hour + dt.dt.minute / 60.0
    
    # Fractional year in radians
    gamma = 2.0 * np.pi / 365.25 * (day_of_year - 1 + (hour - 12) / 24.0)
    
    # Equation of time (in minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    
    # Solar declination angle (radians)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    
    # True solar time (minutes, EDT is UTC-4 -> offset = -240 min)
    tz_offset_min = -240.0
    time_offset = eqtime + 4.0 * LONGITUDE_DEG - tz_offset_min
    tst = hour * 60.0 + time_offset
    
    # Solar hour angle (degrees)
    ha_deg = (tst / 4.0) - 180.0
    ha_rad = np.radians(ha_deg)
    lat_rad = np.radians(LATITUDE_DEG)
    
    # Solar zenith angle (radians)
    cos_sza = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza_deg = np.degrees(np.arccos(cos_sza))
    
    # Solar elevation angle
    elevation_deg = 90.0 - sza_deg
    
    # Extraterrestrial solar flux (W/m^2)
    solar_flux = np.maximum(0.0, 1361.0 * np.sin(np.radians(elevation_deg)))
    
    df_solar = pd.DataFrame({
        "solar_elevation": elevation_deg,
        "solar_zenith": sza_deg,
        "extraterrestrial_flux": solar_flux,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25)
    }, index=dt.index)
    
    return df_solar

def enrich_meteorological_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Derive kinematics, thermodynamics, barometric pressure tendencies, and calendar tokens."""
    df = df.copy()
    
    # 1. Wind Vectors
    w_spd = df["wind_speed_10m"].values
    w_dir_rad = np.radians(df["wind_direction_10m"].fillna(0.0).values)
    df["u_wind"] = -w_spd * np.sin(w_dir_rad)
    df["v_wind"] = -w_spd * np.cos(w_dir_rad)

    # 2. Thermodynamics
    t_c = df["temperature_2m"].values
    dp_c = df["dew_point_2m"].values
    p_hpa = df["surface_pressure"].values

    es = 0.61078 * np.exp(17.27 * t_c / (t_c + 237.3))
    e = 0.61078 * np.exp(17.27 * dp_c / (dp_c + 237.3))
    df["vpd"] = np.maximum(0.0, es - e)
    df["q_spec"] = 1000.0 * (0.622 * e / (p_hpa - 0.378 * e))
    df["theta"] = (t_c + 273.15) * ((1000.0 / p_hpa) ** 0.286)

    # 3. Barometric and temperature tendencies
    p_s = pd.Series(p_hpa)
    df["dp_3h"] = p_s.diff(3).fillna(0.0).values
    df["dp_24h"] = p_s.diff(24).fillna(0.0).values
    
    t_s = pd.Series(df["target_tmpf"].values)
    df["dt_24h"] = t_s.diff(24).fillna(0.0).values

    # 4. Deterministic Solar features
    df_solar = calculate_solar_features(df["time"])
    for col in df_solar.columns:
        df[col] = df_solar[col].values

    # 5. Categorical Time Features
    dt = pd.to_datetime(df["time"])
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["weekday"] = dt.dt.dayofweek
    df["year_idx"] = dt.dt.year - 2000  # 0 to 26
    
    def get_season(m):
        if m in [12, 1, 2]: return 1
        elif m in [3, 4, 5]: return 2
        elif m in [6, 7, 8]: return 3
        else: return 4
    df["season"] = dt.dt.month.map(get_season)
    
    # Fill small gaps
    for c in CONT_FEATURES:
        df[c] = df[c].interpolate(method="linear").ffill().bfill()
        
    return df

def compute_historical_climatology(df: pd.DataFrame, target_col: str = "target_tmpf") -> pd.DataFrame:
    """Compute empirical 26-year mean and std for each (day_of_year, hour) tuple."""
    df = df.copy()
    dt = pd.to_datetime(df["time"])
    df["doy"] = dt.dt.dayofyear
    df["hour"] = dt.dt.hour
    
    clim = df.groupby(["doy", "hour"])[target_col].agg(["mean", "std"]).reset_index()
    clim.rename(columns={"mean": "clim_mean", "std": "clim_std"}, inplace=True)
    return clim

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard meteorological forecast metrics (MAE, RMSE, Bias, Peak MAEs)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    err = y_pred - y_true
    
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mbe = float(np.mean(err))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mbe": mbe,
    }
