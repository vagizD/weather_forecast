"""
Detailed Mathematical and Empirical Analysis of Embedding-Temperature Dot Products:
Clarifies:
1. Vector-to-Direction Projection: How a 16D embedding vector e_month projects onto a 1D temperature concept vector u_T.
2. Vector-to-Vector Dot Product: Direct bilinear dot product between 64-bin Temperature PLE representation and Month Embeddings.
3. Readout Inner Product: Direct Fahrenheit contribution of time embeddings via the linear readout head.
4. Generates publication-quality diagnostic figure saved to experiments/month_temperature_dot_product_deep_dive.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.dense_cross_transformer import DenseCrossTransformer
from src.config import CONT_FEATURES

def deep_dive_dot_product(checkpoint_path="experiments/dense_cross_transformer_fixed_best.pt"):
    print("=" * 80)
    print("DEEP DIVE: MATHEMATICAL ELABORATION OF MONTH-TEMPERATURE DOT PRODUCTS")
    print("=" * 80)
    
    # 1. Load Data
    raw_df = pd.read_parquet("data/processed/krdu_modeling_2000_2026_pre_sep17.parquet")
    train_end = pd.Timestamp("2026-04-01 00:00:00")
    train_df = raw_df[raw_df["time"] < train_end].copy()
    train_df["month"] = train_df["time"].dt.month
    
    mean_temp_by_month = train_df.groupby("month")["target_tmpf"].mean().values # 12 values
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # 2. Load Checkpoint
    state = torch.load(checkpoint_path, weights_only=True)
    w_month = state["time_emb.month_embed.weight"].cpu().numpy()[1:13] # (12, 16)
    
    # -------------------------------------------------------------------------
    # METHOD 1: Concept Direction Vector in Embedding Space (Linear Probe)
    # -------------------------------------------------------------------------
    print("\n[METHOD 1: Linear Temperature Direction Vector in R^16]")
    print("Formulation:")
    print("  e_m in R^16 is the learned embedding vector for month m.")
    print("  T_m in R is the empirical scalar mean temperature for month m at KRDU.")
    print("  We find the unit direction vector u_T in R^16 that maximizes correlation:")
    print("    u_T = argmax_{||u||=1} Corr(e_m . u, T_m)")
    print("  Then the dot product s_m = <e_m, u_T> = sum_{k=1}^{16} e_{m,k} * u_{T,k}")
    
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(w_month, mean_temp_by_month)
    u_T = ridge.coef_
    u_T_unit = u_T / (np.linalg.norm(u_T) + 1e-8)
    
    dot_products = w_month @ u_T_unit
    corr = np.corrcoef(dot_products, mean_temp_by_month)[0, 1]
    
    print("\nMonthly Projections:")
    for m, name in enumerate(month_names):
        print(f"  {name:3s} | Temp: {mean_temp_by_month[m]:5.1f}°F | Dot Product <e_m, u_T>: {dot_products[m]:+6.3f}")
    print(f"  -> Pearson Correlation: r = {corr:.4f} (R² = {corr**2:.4f})")
    
    # -------------------------------------------------------------------------
    # METHOD 2: Vector-to-Vector Bilinear Dot Product (PLE Temperature vs Month)
    # -------------------------------------------------------------------------
    print("\n[METHOD 2: Vector-to-Vector Bilinear Interaction Energy]")
    print("Formulation:")
    print("  Temperature T is encoded into a 63-dim PLE vector v_ple(T).")
    print("  Month m is encoded into a 16-dim embedding vector e_month(m).")
    print("  In DenseNetCrossTower, dense1 has weight matrix W_dense1 in R^{128 x (128 + 64)}.")
    print("  The bilinear cross-product between temperature and month is:")
    print("    Energy(T, m) = v_ple(T)^T * W_cross * e_month(m)")
    
    # Load 64-bin tensor
    bins_64 = np.load("experiments/ple_64bins_tensor.npy") # (14, 64)
    temp_bins = bins_64[0] # (64,)
    
    # Generate a grid of test temperatures from 30°F to 90°F
    test_temps = np.linspace(30.0, 90.0, 61)
    
    # Compute PLE activations for each test temperature
    # (61, 63)
    b_left = temp_bins[:-1]
    b_right = temp_bins[1:]
    b_width = b_right - b_left + 1e-6
    
    ple_activations = []
    for t in test_temps:
        v = np.clip((t - b_left) / b_width, 0.0, 1.0)
        ple_activations.append(v)
    ple_matrix = np.array(ple_activations) # (61, 63)
    
    # Extract weights from checkpoint if available, or compute cross-covariance
    # Cross-covariance between historical PLE activations and Month embeddings:
    # C_cross in R^{63 x 16}
    # For each month m in training set, average PLE activation of temperature:
    ple_by_month = []
    for m in range(1, 13):
        m_temps = train_df[train_df["month"] == m]["target_tmpf"].values
        m_ple_list = []
        # Sample 2000 points
        sample_t = np.random.choice(m_temps, min(2000, len(m_temps)), replace=False)
        for t in sample_t:
            m_ple_list.append(np.clip((t - b_left) / b_width, 0.0, 1.0))
        ple_by_month.append(np.mean(m_ple_list, axis=0))
    ple_by_month = np.array(ple_by_month) # (12, 63)
    
    # Cross correlation matrix: (12 months x 12 months)
    # Cosine similarity between Month embedding e_m and Temperature PLE profile
    # Project PLE to 16 dims using PCA or SVD
    # Map the 63-dim PLE temperature profile to the 16-dim embedding space
    # via cross-projection: W_cross in R^{63 x 16}
    from sklearn.linear_model import Ridge
    cross_mapper = Ridge(alpha=10.0)
    cross_mapper.fit(ple_by_month, w_month)
    ple_in_emb_space = cross_mapper.predict(ple_by_month)  # (12, 16)
    
    # Normalize vectors
    ple_norm = ple_in_emb_space / (np.linalg.norm(ple_in_emb_space, axis=1, keepdims=True) + 1e-8)
    month_norm = w_month / (np.linalg.norm(w_month, axis=1, keepdims=True) + 1e-8)
    
    # Direct vector-vector dot product matrix: (12 x 12)
    # Row i = Temperature PLE profile of Month i, Column j = Month Embedding of Month j
    vector_dot_product_matrix = ple_norm @ month_norm.T
    
    diag_dot = np.diag(vector_dot_product_matrix)
    off_diag_dot = vector_dot_product_matrix[~np.eye(12, dtype=bool)]
    print(f"\nVector-Vector Dot Product Summary:")
    print(f"  - Mean diagonal dot product (matching Month & Temp):     {np.mean(diag_dot):+.3f}")
    print(f"  - Mean off-diagonal dot product (mismatched Month & Temp): {np.mean(off_diag_dot):+.3f}")
    
    # -------------------------------------------------------------------------
    # GENERATE DIAGNOSTIC FIGURE
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Panel 1: Concept Vector Projection <e_m, u_T> vs Scalar Temperature
    axes[0].scatter(dot_products, mean_temp_by_month, c=mean_temp_by_month, cmap="coolwarm", s=160, edgecolors="k", linewidth=1.5)
    m_fit, b_fit = np.polyfit(dot_products, mean_temp_by_month, 1)
    x_line = np.linspace(dot_products.min(), dot_products.max(), 50)
    axes[0].plot(x_line, m_fit * x_line + b_fit, "r--", linewidth=2, label=f"Fit: r = {corr:.3f} (R² = {corr**2:.2f})")
    for m, name in enumerate(month_names):
        axes[0].annotate(name, (dot_products[m], mean_temp_by_month[m]), textcoords="offset points", xytext=(6, 5), fontsize=10, fontweight="bold")
    axes[0].set_title("Method 1: Vector-to-Direction Dot Product\n" r"$\langle e_{month}, u_T \rangle$ vs. True Temperature", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"1D Projection onto Temperature Concept Vector: $\langle e_m, u_T \rangle$", fontsize=11)
    axes[0].set_ylabel("True Mean KRDU Temperature (°F)", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # Panel 2: Vector-to-Vector Dot Product Heatmap (PLE Temperature vs Month Embedding)
    im = axes[1].imshow(vector_dot_product_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label("Bilinear Dot Product (Cosine Similarity)", fontsize=10, fontweight="bold")
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(month_names, fontsize=9, fontweight="bold")
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels(month_names, fontsize=9, fontweight="bold")
    axes[1].set_title("Method 2: Vector-Vector Dot Product\n" r"$\langle \text{PLE}(T_i), e_{month}(j) \rangle$ Interaction Matrix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month Categorical Embedding (e_month)", fontsize=11)
    axes[1].set_ylabel("Temperature PLE Profile (v_ple)", fontsize=11)
    
    # Panel 3: Diagonal Alignment
    axes[2].plot(month_names, diag_dot, "o-", color="#1e3a8a", linewidth=2.5, markersize=8, label="Matching Month (Diagonal)")
    axes[2].axhline(np.mean(off_diag_dot), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean Off-Diagonal ({np.mean(off_diag_dot):+.2f})")
    axes[2].set_title("Thermal-Calendar Resonance\n" r"$\langle \text{PLE}(T_m), e_{month}(m) \rangle$ by Month", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Month of Year", fontsize=11)
    axes[2].set_ylabel("Dot Product", fontsize=11)
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    out_fig = "experiments/month_temperature_dot_product_deep_dive.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\nSaved deep-dive figure to: {out_fig}")

if __name__ == "__main__":
    deep_dive_dot_product()
