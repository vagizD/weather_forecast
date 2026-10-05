"""
Master Evaluation Script:
Evaluates all models (Baselines, LightGBM, PatchTST variants, DenseCrossTransformer)
on the Fixed Setup (Validation, Test) and the Official Locked SuperTest Window (Sep 17-30, 2026).

Usage:
    python evaluate.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from src.config import REPORT_DIR, EXPERIMENTS_DIR

def print_master_table():
    summary_path = os.path.join(REPORT_DIR, "fixed_setup_all_models_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    print("\n" + "=" * 110)
    print("MASTER EVALUATION BENCHMARK: 14-DAY HOURLY TEMPERATURE FORECASTING (KRDU)")
    print("=" * 110)
    
    display_cols = [
        "model",
        "val_mae_mean",
        "test_mae_mean",
        "test_rmse_mean",
        "locked_test_mae",
        "retrained_locked_test_mae",
        "retrained_locked_test_rmse"
    ]
    cols_present = [c for c in display_cols if c in df.columns]
    renamed = {
        "model": "Model Architecture",
        "val_mae_mean": "Val MAE (°F)",
        "test_mae_mean": "Test MAE (°F)",
        "test_rmse_mean": "Test RMSE (°F)",
        "locked_test_mae": "Locked MAE (Zero-Shot)",
        "retrained_locked_test_mae": "Retrained Locked MAE",
        "retrained_locked_test_rmse": "Retrained Locked RMSE"
    }
    
    print(df[cols_present].rename(columns=renamed).to_string(index=False))
    print("=" * 110)
    print("Full results report available in: results.md\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 14-Day Weather Forecasting Models at KRDU")
    args = parser.parse_args()
    print_master_table()
