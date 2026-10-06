"""
Train and Evaluate PatchTST with MLE Loss Functions (Student-t and Laplace).
Evaluates on:
  1. 78 Spring Validation Windows
  2. 65 Summer Test Windows
  3. Locked SuperTest Window (Zero-Shot: Train < 2026-04-01)
  4. Locked SuperTest Window (Retrained: Train + Val < 2026-07-01)
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_END, VAL_END, TEST_END, LOCKED_START, LOCKED_END,
    HISTORY_LEN, HORIZON_LEN, PATCH_LEN, STRIDE,
    PT_FEATURES, SOLAR_FUTURE_COLS, DEVICE,
    PROCESSED_DATA_PATH, LOCKED_TEST_PATH, REPORT_DIR, EXPERIMENTS_DIR
)
from src.data.dataset import get_window_indices, SlidingTrainDataset, FixedRollingDataset
from src.data.preprocessing import (
    compute_historical_climatology, enrich_meteorological_dataset, calculate_solar_features
)
from src.models.patch_tst import PatchTSTWithTimeEmbedding
from src.training.loss import CompositeWeatherLoss

def evaluate_window_predictions(y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    w1_mae = float(np.mean(np.abs(y_true[:168] - y_pred[:168])))
    w1_rmse = float(np.sqrt(np.mean((y_true[:168] - y_pred[:168]) ** 2)))
    w2_mae = float(np.mean(np.abs(y_true[168:] - y_pred[168:])))
    w2_rmse = float(np.sqrt(np.mean((y_true[168:] - y_pred[168:]) ** 2)))

    eval_df = pd.DataFrame({"time": timestamps, "y_true": y_true, "y_pred": y_pred})
    eval_df["date"] = eval_df["time"].dt.date
    daily = eval_df.groupby("date").agg({
        "y_true": ["max", "min"],
        "y_pred": ["max", "min"]
    })
    tmax_mae = float(np.mean(np.abs(daily["y_true"]["max"] - daily["y_pred"]["max"])))
    tmin_mae = float(np.mean(np.abs(daily["y_true"]["min"] - daily["y_pred"]["min"])))

    return {
        "mae": mae,
        "rmse": rmse,
        "tmax_mae": tmax_mae,
        "tmin_mae": tmin_mae,
        "w1_mae": w1_mae,
        "w1_rmse": w1_rmse,
        "w2_mae": w2_mae,
        "w2_rmse": w2_rmse,
    }

def evaluate_rolling(model: nn.Module, loader: DataLoader) -> tuple:
    model.eval()
    maes, rmses = [], []
    with torch.no_grad():
        for batch in loader:
            x_hist = batch["pt_hist"].to(DEVICE)
            y_clim = batch["y_clim"].to(DEVICE)
            y_true = batch["y_true"].to(DEVICE)
            future_solar = batch["pt_solar_fut"].to(DEVICE)
            future_time = {k: v.to(DEVICE) for k, v in batch["future_time"].items()}
            pred = model(x_hist, y_clim, future_solar, future_time)
            
            err = (pred - y_true).cpu().numpy()
            w_mae = np.mean(np.abs(err), axis=1)
            w_rmse = np.sqrt(np.mean(err**2, axis=1))
            maes.extend(w_mae)
            rmses.extend(w_rmse)
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(rmses)), float(np.std(rmses))

def predict_locked_window(model: nn.Module, df: pd.DataFrame, clim_lookup: dict) -> tuple:
    model.eval()
    split = df[df["time"] == LOCKED_START].index[0]
    hist_start = split - HISTORY_LEN
    end = split + HORIZON_LEN

    pt_hist = torch.from_numpy(df[PT_FEATURES].values[hist_start:split].astype(np.float32)).unsqueeze(0).to(DEVICE)
    solar_df = calculate_solar_features(df["time"])
    future_cols = ["solar_elevation", "extraterrestrial_flux", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    pt_solar_fut = torch.from_numpy(solar_df[future_cols].values[split:end].astype(np.float32)).unsqueeze(0).to(DEVICE)

    fut_time = {
        "hour": torch.from_numpy(df["hour"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE),
        "month": torch.from_numpy(df["month"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE),
        "season": torch.from_numpy(df["season"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE),
        "day": torch.from_numpy(df["day"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE),
        "weekday": torch.from_numpy(df["weekday"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE),
        "year": torch.from_numpy(df["year_idx"].values[split:end].astype(np.int64)).unsqueeze(0).to(DEVICE)
    }

    y_clim_np = np.array([clim_lookup.get((t.dayofyear, t.hour), 60.0) for t in df["time"].iloc[split:end]], dtype=np.float32)
    y_clim = torch.from_numpy(y_clim_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(pt_hist, y_clim, pt_solar_fut, fut_time).squeeze().cpu().numpy()

    y_true = df["target_tmpf"].iloc[split:end].values
    timestamps = pd.to_datetime(df["time"].iloc[split:end])
    metrics = evaluate_window_predictions(y_true, pred, timestamps)
    return pred, metrics

def train_model(
    loss_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 3,
    save_path: str = None
) -> tuple:
    torch.manual_seed(42)
    model = PatchTSTWithTimeEmbedding(
        history_len=HISTORY_LEN,
        horizon_len=HORIZON_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        num_features=len(PT_FEATURES),
        d_model=128
    ).to(DEVICE)
    
    # Initialize from transferred time embedding weights
    ckpt = torch.load("experiments/patchtst_time_emb_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = CompositeWeatherLoss(loss_type=loss_type, lambda_slope=0.3, lambda_extrema=0.2)

    best_val_mae = 999.0
    best_state = None
    best_metrics = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x_hist = batch["pt_hist"].to(DEVICE)
            y_clim = batch["y_clim"].to(DEVICE)
            y_true = batch["y_true"].to(DEVICE)
            future_solar = batch["pt_solar_fut"].to(DEVICE)
            future_time = {k: v.to(DEVICE) for k, v in batch["future_time"].items()}

            optimizer.zero_grad()
            pred = model(x_hist, y_clim, future_solar, future_time)
            loss, _ = criterion(pred, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_mae, val_mae_std, val_rmse, val_rmse_std = evaluate_rolling(model, val_loader)
        test_mae, test_mae_std, test_rmse, test_rmse_std = evaluate_rolling(model, test_loader)
        print(f"[{loss_type.upper()}] Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val MAE: {val_mae:.3f}°F | Test MAE: {test_mae:.3f}±{test_mae_std:.3f}°F (RMSE: {test_rmse:.3f}°F)")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_mae_mean": val_mae,
                "val_rmse_mean": val_rmse,
                "test_mae_mean": test_mae,
                "test_mae_std": test_mae_std,
                "test_rmse_mean": test_rmse,
                "test_rmse_std": test_rmse_std,
                "best_epoch": epoch
            }

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    if save_path:
        torch.save(best_state, save_path)
        print(f"Saved best model checkpoint to {save_path}")

    return model, best_metrics

def main():
    print("=" * 80)
    print("TRAINING & EVALUATION OF PATCHTST UNDER MLE OBJECTIVES (STUDENT-T & LAPLACE)")
    print("=" * 80)

    # Load and merge full dataset
    pre_df = pd.read_parquet(PROCESSED_DATA_PATH)
    locked_df = pd.read_csv(LOCKED_TEST_PATH)
    locked_df["time"] = pd.to_datetime(locked_df["time"])
    full_df = pd.concat([pre_df, locked_df], ignore_index=True)
    full_df = enrich_meteorological_dataset(full_df)

    # Climatology strictly on Train
    clim_df_p1 = compute_historical_climatology(full_df[full_df["time"] < TRAIN_END])
    clim_lookup_p1 = clim_df_p1.set_index(["doy", "hour"])["clim_mean"].to_dict()

    # Dataloaders for Phase 1
    train_ds_p1 = SlidingTrainDataset(full_df[full_df["time"] < TRAIN_END], clim_lookup_p1, step=24)
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=64, shuffle=True)

    val_windows = [w["start"] for w in get_window_indices(full_df, TRAIN_END, VAL_END)]
    test_windows = [w["start"] for w in get_window_indices(full_df, VAL_END, TEST_END)]
    val_ds = FixedRollingDataset(full_df, val_windows, clim_lookup_p1)
    test_ds = FixedRollingDataset(full_df, test_windows, clim_lookup_p1)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # -------------------------------------------------------------------------
    # PHASE 1: Zero-Shot Training (Train < 2026-04-01)
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 1: Training on Train (< 2026-04-01)...")
    m_student_t, p1_student_t_metrics = train_model(
        "student_t", train_loader_p1, val_loader, test_loader, num_epochs=3,
        save_path="experiments/patchtst_student_t_best.pt"
    )
    pred_st_locked, st_locked_metrics = predict_locked_window(m_student_t, full_df, clim_lookup_p1)
    print(f"Student-t Locked Zero-Shot: MAE={st_locked_metrics['mae']:.2f}°F | RMSE={st_locked_metrics['rmse']:.2f}°F")

    m_laplace, p1_laplace_metrics = train_model(
        "laplace", train_loader_p1, val_loader, test_loader, num_epochs=3,
        save_path="experiments/patchtst_laplace_best.pt"
    )
    pred_lap_locked, lap_locked_metrics = predict_locked_window(m_laplace, full_df, clim_lookup_p1)
    print(f"Laplace Locked Zero-Shot: MAE={lap_locked_metrics['mae']:.2f}°F | RMSE={lap_locked_metrics['rmse']:.2f}°F")

    # -------------------------------------------------------------------------
    # PHASE 2: Retraining on Train + Val (< 2026-07-01)
    # -------------------------------------------------------------------------
    print("\n>>> PHASE 2: Retraining on Train + Val (< 2026-07-01)...")
    clim_df_p2 = compute_historical_climatology(full_df[full_df["time"] < VAL_END])
    clim_lookup_p2 = clim_df_p2.set_index(["doy", "hour"])["clim_mean"].to_dict()

    train_ds_p2 = SlidingTrainDataset(full_df[full_df["time"] < VAL_END], clim_lookup_p2, step=24)
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=64, shuffle=True)

    val_ds_p2 = FixedRollingDataset(full_df, val_windows, clim_lookup_p2)
    test_ds_p2 = FixedRollingDataset(full_df, test_windows, clim_lookup_p2)
    val_loader_p2 = DataLoader(val_ds_p2, batch_size=32, shuffle=False)
    test_loader_p2 = DataLoader(test_ds_p2, batch_size=32, shuffle=False)

    m_st_retrained, _ = train_model(
        "student_t", train_loader_p2, val_loader_p2, test_loader_p2, num_epochs=2,
        save_path="experiments/retrained_patchtst_student_t_best.pt"
    )
    pred_st_retrained_locked, st_retrained_locked_metrics = predict_locked_window(m_st_retrained, full_df, clim_lookup_p2)
    print(f"Student-t Locked Retrained: MAE={st_retrained_locked_metrics['mae']:.2f}°F | RMSE={st_retrained_locked_metrics['rmse']:.2f}°F")

    m_lap_retrained, _ = train_model(
        "laplace", train_loader_p2, val_loader_p2, test_loader_p2, num_epochs=2,
        save_path="experiments/retrained_patchtst_laplace_best.pt"
    )
    pred_lap_retrained_locked, lap_retrained_locked_metrics = predict_locked_window(m_lap_retrained, full_df, clim_lookup_p2)
    print(f"Laplace Locked Retrained: MAE={lap_retrained_locked_metrics['mae']:.2f}°F | RMSE={lap_retrained_locked_metrics['rmse']:.2f}°F")

    # -------------------------------------------------------------------------
    # Save Predictions into CSV files
    # -------------------------------------------------------------------------
    p1_pred_df = pd.read_csv("experiments/locked_test_predictions_phase1.csv")
    p1_pred_df["PatchTST + Student-t MLE Loss"] = pred_st_locked
    p1_pred_df["PatchTST + Laplace MLE Loss"] = pred_lap_locked
    p1_pred_df.to_csv("experiments/locked_test_predictions_phase1.csv", index=False)

    p2_pred_df = pd.read_csv("experiments/locked_test_predictions_phase2_retrained.csv")
    p2_pred_df["PatchTST + Student-t MLE Loss"] = pred_st_retrained_locked
    p2_pred_df["PatchTST + Laplace MLE Loss"] = pred_lap_retrained_locked
    p2_pred_df.to_csv("experiments/locked_test_predictions_phase2_retrained.csv", index=False)

    # -------------------------------------------------------------------------
    # Update Phase 1 & Phase 2 Locked Results CSVs
    # -------------------------------------------------------------------------
    p1_res_df = pd.read_csv("experiments/locked_test_results_phase1.csv")
    for name, m in [("PatchTST + Student-t MLE Loss", st_locked_metrics), ("PatchTST + Laplace MLE Loss", lap_locked_metrics)]:
        p1_res_df = p1_res_df[p1_res_df["model"] != name]
        new_row = {
            "model": name,
            "locked_test_mae": m["mae"],
            "locked_test_rmse": m["rmse"],
            "locked_test_tmax_mae": m["tmax_mae"],
            "locked_test_tmin_mae": m["tmin_mae"],
            "locked_test_w1_mae": m["w1_mae"],
            "locked_test_w1_rmse": m["w1_rmse"],
            "locked_test_w2_mae": m["w2_mae"],
            "locked_test_w2_rmse": m["w2_rmse"]
        }
        p1_res_df = pd.concat([p1_res_df, pd.DataFrame([new_row])], ignore_index=True)
    p1_res_df.to_csv("experiments/locked_test_results_phase1.csv", index=False)

    p2_res_df = pd.read_csv("experiments/locked_test_results_phase2_retrained.csv")
    for name, m in [("PatchTST + Student-t MLE Loss", st_retrained_locked_metrics), ("PatchTST + Laplace MLE Loss", lap_retrained_locked_metrics)]:
        p2_res_df = p2_res_df[p2_res_df["model"] != name]
        new_row = {
            "model": name,
            "retrained_locked_test_mae": m["mae"],
            "retrained_locked_test_rmse": m["rmse"],
            "retrained_locked_test_tmax_mae": m["tmax_mae"],
            "retrained_locked_test_tmin_mae": m["tmin_mae"],
            "retrained_locked_test_w1_mae": m["w1_mae"],
            "retrained_locked_test_w1_rmse": m["w1_rmse"],
            "retrained_locked_test_w2_mae": m["w2_mae"],
            "retrained_locked_test_w2_rmse": m["w2_rmse"]
        }
        p2_res_df = pd.concat([p2_res_df, pd.DataFrame([new_row])], ignore_index=True)
    p2_res_df.to_csv("experiments/locked_test_results_phase2_retrained.csv", index=False)

    # -------------------------------------------------------------------------
    # Output rows for Master Evaluation Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NEW MODEL RESULTS FOR MASTER TABLE:")
    print("=" * 80)
    row_st = {
        "model": "PatchTST + Student-t MLE Loss",
        "best_params": f"best_epoch={p1_student_t_metrics['best_epoch']}, nu=8.44, sigma=4.38",
        "val_mae_mean": round(p1_student_t_metrics["val_mae_mean"], 2),
        "test_mae_mean": round(p1_student_t_metrics["test_mae_mean"], 2),
        "test_mae_std": round(p1_student_t_metrics["test_mae_std"], 2),
        "test_rmse_mean": round(p1_student_t_metrics["test_rmse_mean"], 2),
        "test_rmse_std": round(p1_student_t_metrics["test_rmse_std"], 2),
        "locked_test_mae": round(st_locked_metrics["mae"], 2),
        "locked_test_rmse": round(st_locked_metrics["rmse"], 2),
        "retrained_locked_test_mae": round(st_retrained_locked_metrics["mae"], 2),
        "retrained_locked_test_rmse": round(st_retrained_locked_metrics["rmse"], 2),
    }
    row_lap = {
        "model": "PatchTST + Laplace MLE Loss",
        "best_params": f"best_epoch={p1_laplace_metrics['best_epoch']}, L1 loss, median estimator",
        "val_mae_mean": round(p1_laplace_metrics["val_mae_mean"], 2),
        "test_mae_mean": round(p1_laplace_metrics["test_mae_mean"], 2),
        "test_mae_std": round(p1_laplace_metrics["test_mae_std"], 2),
        "test_rmse_mean": round(p1_laplace_metrics["test_rmse_mean"], 2),
        "test_rmse_std": round(p1_laplace_metrics["test_rmse_std"], 2),
        "locked_test_mae": round(lap_locked_metrics["mae"], 2),
        "locked_test_rmse": round(lap_locked_metrics["rmse"], 2),
        "retrained_locked_test_mae": round(lap_retrained_locked_metrics["mae"], 2),
        "retrained_locked_test_rmse": round(lap_retrained_locked_metrics["rmse"], 2),
    }
    print(pd.DataFrame([row_st, row_lap]).to_string(index=False))

    # Update summary table
    summary_path = os.path.join(REPORT_DIR, "fixed_setup_all_models_summary.csv")
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[~summary_df["model"].isin(["PatchTST + Student-t MLE Loss", "PatchTST + Laplace MLE Loss"])]
    summary_df = pd.concat([summary_df, pd.DataFrame([row_st, row_lap])], ignore_index=True)
    
    # Sort strictly by test_mae_mean
    # Split non-ablation vs ablation
    main_models = summary_df[~summary_df["model"].str.contains("Ablation")].copy()
    ablation_models = summary_df[summary_df["model"].str.contains("Ablation")].copy()
    main_models["test_mae_num"] = pd.to_numeric(main_models["test_mae_mean"], errors="coerce")
    main_models = main_models.sort_values(by="test_mae_num", ascending=True).drop(columns=["test_mae_num"])
    
    final_summary_df = pd.concat([main_models, ablation_models], ignore_index=True)
    final_summary_df.to_csv(summary_path, index=False)
    print(f"\nSuccessfully updated {summary_path} with strict test_mae_mean sorting!")

if __name__ == "__main__":
    main()
