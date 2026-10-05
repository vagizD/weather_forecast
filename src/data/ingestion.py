"""
Robust Data Ingestion Pipeline for RDU Airport Weather Forecasting.

Features:
- Individual chunk caching to disk (resumable)
- HTTP 429 rate limit backoff and retry
- Clean fallback and merging
- Strict Cutoff: Modeling data < 2026-09-17 00:00:00
- SuperTest ground truth locked in data/test/supertest_ground_truth_locked.csv
"""

import os
import sys
import time
import json
import io
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import numpy as np

LATITUDE = 35.8776
LONGITUDE = -78.7875
TIMEZONE = "America/New_York"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TEST_DIR = "data/test"

def fetch_with_retry(url: str, max_retries: int = 5, initial_backoff: float = 5.0) -> bytes:
    """Fetch URL with exponential backoff on HTTP 429 or network errors."""
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DukeWeatherProject/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    [Rate Limited 429] Waiting {backoff:.1f}s before retry {attempt}/{max_retries}...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"    [HTTP Error {e.code}] Attempt {attempt}/{max_retries}")
                time.sleep(backoff)
                backoff *= 1.5
        except Exception as e:
            print(f"    [Network Error: {e}] Attempt {attempt}/{max_retries}")
            time.sleep(backoff)
            backoff *= 1.5
            
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")

def fetch_open_meteo_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly reanalysis data from Open-Meteo Archive API with chunk caching."""
    os.makedirs(RAW_DIR, exist_ok=True)
    chunk_file = os.path.join(RAW_DIR, f"om_{start_date}_{end_date}.parquet")
    if os.path.exists(chunk_file):
        print(f"    Using cached Open-Meteo chunk: {chunk_file}")
        return pd.read_parquet(chunk_file)
        
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "cloud_cover",
            "shortwave_radiation",
            "wind_speed_10m",
            "wind_direction_10m"
        ]),
        "timezone": TIMEZONE
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    raw_bytes = fetch_with_retry(url)
    data = json.loads(raw_bytes.decode("utf-8"))
    
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_parquet(chunk_file, index=False)
    print(f"    Downloaded & cached chunk: {chunk_file} ({len(df)} hours)")
    time.sleep(3.0)  # Gentle delay between requests
    return df

def fetch_all_open_meteo(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """Fetch full historical archive from Open-Meteo in 4-year chunks."""
    final_cache = os.path.join(RAW_DIR, f"open_meteo_krdu_{start_year}_{end_year}.parquet")
    if os.path.exists(final_cache):
        print(f"Loading cached full Open-Meteo dataset from {final_cache}...")
        return pd.read_parquet(final_cache)
        
    print(f"Fetching Open-Meteo data for KRDU ({start_year} to {end_year})...")
    chunks = []
    chunk_size = 4
    for y in range(start_year, end_year + 1, chunk_size):
        y_end = min(y + chunk_size - 1, end_year)
        start_d = f"{y}-01-01"
        end_d = f"{y_end}-12-31" if y_end < 2026 else "2026-09-30"
        print(f"  Processing chunk {start_d} to {end_d}...")
        chunk = fetch_open_meteo_chunk(start_d, end_d)
        chunks.append(chunk)
        
    df_all = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df_all.to_parquet(final_cache, index=False)
    print(f"Saved complete Open-Meteo dataset: {final_cache} ({len(df_all)} hours)")
    return df_all

def fetch_iem_asos_krdu(start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    """Fetch official ASOS hourly observations from Iowa Environmental Mesonet."""
    final_cache = os.path.join(RAW_DIR, f"asos_krdu_{start_year}_{end_year}.parquet")
    if os.path.exists(final_cache):
        print(f"Loading cached full ASOS dataset from {final_cache}...")
        return pd.read_parquet(final_cache)
        
    print(f"Fetching IEM ASOS data for KRDU station ({start_year} to {end_year})...")
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    chunks = []
    chunk_size = 4
    for y in range(start_year, end_year + 1, chunk_size):
        y_end = min(y + chunk_size - 1, end_year)
        chunk_file = os.path.join(RAW_DIR, f"asos_{y}_{y_end}.parquet")
        
        if os.path.exists(chunk_file):
            print(f"    Using cached ASOS chunk: {chunk_file}")
            df_chunk = pd.read_parquet(chunk_file)
            chunks.append(df_chunk)
            continue
            
        params = {
            "station": "RDU",
            "data": ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby", "p01i"],
            "year1": y, "month1": 1, "day1": 1,
            "year2": y_end, "month2": 12 if y_end < 2026 else 9, "day2": 31 if y_end < 2026 else 30,
            "tz": TIMEZONE,
            "format": "onlycomma",
            "latlon": "no",
            "missing": "M"
        }
        url = f"{base_url}?{urllib.parse.urlencode(params, doseq=True)}"
        print(f"  Fetching ASOS chunk {y} to {y_end}...")
        raw_bytes = fetch_with_retry(url, max_retries=3, initial_backoff=3.0)
        df_raw = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8")), comment="#")
        df_raw["valid"] = pd.to_datetime(df_raw["valid"])
        
        numeric_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "vsby", "p01i"]
        for col in numeric_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
                
        df_hourly = df_raw.set_index("valid")[numeric_cols].resample("1h").mean().reset_index()
        df_hourly = df_hourly.rename(columns={"valid": "time"})
        df_hourly.to_parquet(chunk_file, index=False)
        print(f"    Downloaded & cached ASOS chunk: {chunk_file} ({len(df_hourly)} hours)")
        chunks.append(df_hourly)
        time.sleep(1.0)
        
    df_all = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df_all.to_parquet(final_cache, index=False)
    print(f"Saved complete ASOS dataset: {final_cache} ({len(df_all)} hours)")
    return df_all

def prepare_and_split_datasets():
    """Download, merge, and split strictly at 2026-09-17 00:00:00."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    print("\n=== STEP 1: Ingesting Open-Meteo & ASOS Data ===")
    df_om = fetch_all_open_meteo(2000, 2026)
    df_asos = fetch_iem_asos_krdu(2000, 2026)
    
    print("\n=== STEP 2: Merging Datasets ===")
    df_om["temperature_2m_f"] = df_om["temperature_2m"] * 9/5 + 32
    df_om["dew_point_2m_f"] = df_om["dew_point_2m"] * 9/5 + 32
    
    merged = pd.merge(df_om, df_asos, on="time", how="left", suffixes=("_om", "_asos"))
    # The evaluation ground truth is ASOS tmpf; fill missing sensor readings with high-res ERA5 reanalysis
    merged["target_tmpf"] = merged["tmpf"].combine_first(merged["temperature_2m_f"])
    merged["target_tmpc"] = (merged["target_tmpf"] - 32) * 5/9
    
    merged = merged.sort_values("time").reset_index(drop=True)
    print(f"Total merged records: {len(merged)} hours ({merged['time'].min()} to {merged['time'].max()})")
    
    # Strict Cutoff: 2026-09-17 00:00:00
    cutoff = pd.Timestamp("2026-09-17 00:00:00")
    supertest_end = pd.Timestamp("2026-09-30 23:00:00")
    
    modeling_data = merged[merged["time"] < cutoff].copy()
    supertest_data = merged[(merged["time"] >= cutoff) & (merged["time"] <= supertest_end)].copy()
    
    print(f"\n=== STEP 3: Saving Modeling Data & SuperTest Holdout ===")
    modeling_path = os.path.join(PROCESSED_DIR, "krdu_modeling_2000_2026_pre_sep17.parquet")
    modeling_data.to_parquet(modeling_path, index=False)
    print(f"Modeling dataset saved: {modeling_path} ({len(modeling_data)} hours, up to {modeling_data['time'].max()})")
    
    supertest_path = os.path.join(TEST_DIR, "supertest_ground_truth_locked.csv")
    supertest_data.to_csv(supertest_path, index=False)
    print(f"SuperTest ground truth locked: {supertest_path} ({len(supertest_data)} hours: {supertest_data['time'].min()} to {supertest_data['time'].max()})")
    print("WARNING: Do NOT touch supertest_ground_truth_locked.csv until final model selection is complete!")

if __name__ == "__main__":
    prepare_and_split_datasets()
