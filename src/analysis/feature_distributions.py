"""
Continuous Meteorological Feature Distribution and 64-Bin Analysis:
Computes rigorous summary statistics across 26.25 years of training data (2000-01-01 to 2026-04-01, 230,088 hours):
- Mean, Standard Deviation, Median, Min, Max, 1st & 99th percentiles, Skewness, Kurtosis
- 64 Empirical Quantile Bins for Piecewise Linear Encoding (PLE)
- Generates 14-panel distribution visualization saved to experiments/continuous_features_64bins_distribution.png
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from src.data.preprocessing import enrich_meteorological_dataset
from src.config import CONT_FEATURES

def analyze_and_plot_distributions():
    print("=" * 80)
    print("ANALYZING CONTINUOUS FEATURE DISTRIBUTIONS & COMPUTING 64 QUANTILE BINS")
    print("Train Period: 2000-01-01 to 2026-04-01 00:00:00 (230,088 continuous hours)")
    print("=" * 80)
    
    raw_df = pd.read_parquet("data/processed/krdu_modeling_2000_2026_pre_sep17.parquet")
    df = enrich_meteorological_dataset(raw_df)
    train_end = pd.Timestamp("2026-04-01 00:00:00")
    train_df = df[df["time"] < train_end].copy()
    
    stats_list = []
    bins_dict = {}
    
    num_bins = 64
    quantiles = np.linspace(0.005, 0.995, num_bins)
    
    # Friendly labels and units for each feature
    feature_meta = {
        "target_tmpf": ("2m Air Temperature", "°F"),
        "dew_point_2m_f": ("2m Dew Point Temperature", "°F"),
        "surface_pressure": ("Surface Barometric Pressure", "hPa"),
        "dp_3h": ("3-Hour Pressure Tendency (dp/3h)", "hPa"),
        "dp_24h": ("24-Hour Pressure Tendency (dp/24h)", "hPa"),
        "dt_24h": ("24-Hour Temperature Change (dt/24h)", "°F"),
        "u_wind": ("Zonal Wind Component (u_wind)", "m/s"),
        "v_wind": ("Meridional Wind Component (v_wind)", "m/s"),
        "vpd": ("Vapor Pressure Deficit (VPD)", "hPa"),
        "q_spec": ("Specific Humidity (q_spec)", "g/kg"),
        "theta": ("Potential Temperature (θ)", "K"),
        "cloud_cover": ("Total Cloud Cover", "%"),
        "shortwave_radiation": ("Surface Shortwave Radiation", "W/m²"),
        "solar_elevation": ("Astronomical Solar Elevation", "degrees")
    }
    
    # 1. Compute summary stats and bins
    for feat in CONT_FEATURES:
        vals = train_df[feat].values
        q_bins = np.unique(np.quantile(vals, quantiles))
        if len(q_bins) < num_bins:
            q_bins = np.linspace(vals.min(), vals.max(), num_bins)
        bins_dict[feat] = q_bins
        
        name, unit = feature_meta[feat]
        stats_list.append({
            "Feature": feat,
            "Description": name,
            "Unit": unit,
            "Mean": float(np.mean(vals)),
            "Std": float(np.std(vals)),
            "Min": float(np.min(vals)),
            "P01": float(np.percentile(vals, 1)),
            "P25 (Q1)": float(np.percentile(vals, 25)),
            "Median": float(np.median(vals)),
            "P75 (Q3)": float(np.percentile(vals, 75)),
            "P99": float(np.percentile(vals, 99)),
            "Max": float(np.max(vals)),
            "Skewness": float(skew(vals)),
            "Kurtosis": float(kurtosis(vals)),
            "Bin_Width_Median": float(np.median(np.diff(q_bins)))
        })
        
    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv("experiments/continuous_features_distribution_stats.csv", index=False)
    print("\n" + "=" * 90)
    print("CONTINUOUS METEOROLOGICAL FEATURE DISTRIBUTION TABLE (26.25 Years at KRDU):")
    print("=" * 90)
    print(df_stats[["Feature", "Unit", "Mean", "Std", "Min", "Median", "Max", "Skewness", "Bin_Width_Median"]].round(2).to_string(index=False))
    
    # Save bins tensor for model training
    bins_array = np.array([bins_dict[f] for f in CONT_FEATURES], dtype=np.float32)
    np.save("experiments/ple_64bins_tensor.npy", bins_array)
    print(f"\nSaved 64-bin tensor of shape {bins_array.shape} to experiments/ple_64bins_tensor.npy")
    
    # 2. Generate 14-Panel Publication-Quality Distribution Figure
    print("\nGenerating 14-panel distribution figure with 64 empirical quantile bins marked...")
    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    axes = axes.flatten()
    
    for i, feat in enumerate(CONT_FEATURES):
        ax = axes[i]
        vals = train_df[feat].values
        q_bins = bins_dict[feat]
        name, unit = feature_meta[feat]
        
        # Subsample for plotting if large
        plot_vals = vals if len(vals) <= 100000 else np.random.choice(vals, 100000, replace=False)
        
        # Plot histogram
        n, bins_hist, patches = ax.hist(plot_vals, bins=80, density=True, color="#3b82f6", alpha=0.65, edgecolor="none")
        
        # Plot markers along the bottom for 64 quantile bin edges
        ax.plot(q_bins, np.zeros_like(q_bins), "|", color="#dc2626", markersize=10, markeredgewidth=1.2, label=f"64 Quantile Bins (med Δ={np.median(np.diff(q_bins)):.2f})")
        
        # Mean and Median vertical lines
        ax.axvline(np.mean(vals), color="#1e3a8a", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(vals):.1f}")
        ax.axvline(np.median(vals), color="#059669", linestyle="-.", linewidth=1.5, label=f"Median: {np.median(vals):.1f}")
        
        ax.set_title(f"{name} ({feat})\n[{unit}]", fontsize=11, fontweight="bold")
        ax.set_xlabel(unit, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        
    # Hide unused subplots (indices 14, 15)
    for j in range(len(CONT_FEATURES), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    out_fig = "experiments/continuous_features_64bins_distribution.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved 14-panel distribution plot to: {out_fig}")
    return df_stats, bins_array

if __name__ == "__main__":
    analyze_and_plot_distributions()
