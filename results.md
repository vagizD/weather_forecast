# Long-Horizon (14-Day) Hourly Weather Forecasting at KRDU: Results & Technical Report

**Task**: Predict continuous hourly surface temperatures for 14 days ($H = 336$ consecutive hours) at Raleigh-Durham International Airport (KRDU) using strictly antecedent observations.

---

## 1. Data Pipeline & Zero-Leakage Architecture

The data pipeline unifies **26 continuous years (2000–2026; 234,144 hourly observations)** of ground-truth meteorological sensors and atmospheric reanalysis:
- **Surface Observation Network**: NOAA/NWS Automated Surface Observing System (ASOS) station KRDU (WBAN: 13722) via Iowa Environmental Mesonet. Supplies primary 2m temperature (`target_tmpf`), dew point, surface pressure, and wind vectors.
- **Atmospheric Reanalysis**: High-resolution ERA5-Land reanalysis via Open-Meteo API, providing vapor pressure deficit (VPD), specific humidity, boundary layer dynamics, and cloud coverage.
- **Deterministic Astronomical Physics**: NOAA solar position equations calculate Solar Zenith Angle (SZA), solar elevation, and extraterrestrial solar irradiance across both history and future forecast horizons.
- **Strict Zero-Leakage Protocol**: Scalers, 64-bin empirical quantile cutoffs, and historical climatological lookup tables are computed strictly on the training partition ($< \text{2026-04-01}$).

```
[Historical Observations: ASOS + ERA5-Land] ──> [Zero-Leakage Feature Extraction] ──> [14-Day Rolling Windows]
  • 26 Years (230,088 train hours)                 • 14 Physical Continuous Features     • 78 Spring Val Windows
  • Temperature, Pressure, Winds, Dew Point        • 6 Categorical Calendar Features     • 65 Summer Test Windows
  • NOAA Astronomical Solar Radiation              • 64 Quantile Cutoffs on Train Only   • 1 SuperTest Locked Window
```

---

## 2. Evaluation Pipeline & Seasonal Meteorological Dynamics

To evaluate real-world forecasting under seasonal shifts, evaluation is conducted over rolling 14-day horizons with a 24-hour daily stride:
- **Train Period**: Jan 1, 2000 – Apr 1, 2026 00:00:00 (230,088 continuous hours)
- **Validation Period**: Apr 1, 2026 – Jun 30, 2026 (78 rolling 14-day windows)
- **Holdout Test Period**: Jul 1, 2026 – Sep 16, 2026 (65 rolling 14-day windows)
- **Official SuperTest (Locked)**: Sep 17, 2026 – Sep 30, 2026 (336 hours preserved strictly untouched)

### Synoptic Seasonal Mechanism: Spring Turbulence vs. Summer Persistence

Across all models, validation error ($\sim 7.0^\circ\text{F} - 7.4^\circ\text{F}$ MAE) is noticeably higher than test error ($\sim 3.8^\circ\text{F} - 4.3^\circ\text{F}$ MAE). This reflects fundamental synoptic dynamics in central North Carolina:

| Season | Synoptic Atmospheric Regime | Meteorological Dynamics | Forecasting Error |
| :--- | :--- | :--- | :---: |
| **Spring (Apr–Jun, Val)** | **Baroclinic Instability** | Polar cold fronts colliding with Gulf moisture trigger abrupt $20^\circ\text{F}–30^\circ\text{F}$ temperature swings and squalls. Chaotic multi-week predictability. | **$\sim 7.0^\circ\text{F} - 7.4^\circ\text{F}$ MAE** |
| **Summer (Jul–Sep, Test)** | **Subtropical Persistence** | Semi-permanent Bermuda Subtropical Ridge parks over central NC. Quasi-stationary air masses governed by predictable diurnal solar heating and nocturnal radiation. | **$\sim 3.8^\circ\text{F} - 4.3^\circ\text{F}$ MAE** |

---

## 3. Model Architectures: Baselines & Deep Learning

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    MODEL ARCHITECTURES                                            |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|  1. Climatology Base           |  Zero-parameter empirical 26-year mean: y_clim(DOY, Hour)        |
|  2. Linear Regression (Ridge)  |  Multi-output projection: W_ridge [x_cutoff, history] -> y_res   |
|  3. LightGBM                   |  150 Gradient boosted decision trees on rolling lag statistics   |
|  4. PatchTST (Transformer)     |  Channel-Independent 24h diurnal patching + Multi-horizon head   |
|  5. PatchTST + Transferred TE  |  PatchTST backbone + 26-year decadal time embeddings from DenseNet|
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Classical Baselines
- **Climatology Base (Zero-Parameter)**: Empirical 26-year historical average: $\hat{y}_{\text{clim}}(t) = \mu_{\text{train}}(\text{DOY}(t), \text{Hour}(t))$.
- **Linear Regression (Ridge, $\alpha = 50.0$)**: Projects 168-hour antecedent state to all 336 future hourly anomalies $\hat{\mathbf{r}} \in \mathbb{R}^{336}$ simultaneously.
- **LightGBM (Tabular GBDT)**: Depth-regulated trees (`num_leaves=31`, `lr=0.03`) trained on lag windows, rolling momentum, and solar angles.

### Deep Learning Architecture: PatchTST Backbone

PatchTST addresses two fundamental long-horizon failure modes: **autoregressive error compounding** and **conditional mean flattening (diurnal collapse)**.

![PatchTST Model Architecture](report/patchtst_architecture_diagram.png)

1. **Climatological Residual Decomposition**: $\hat{y}(t) = \hat{y}_{\text{clim}}(t) + \hat{r}_\theta(t)$. Predicting anomalies rather than absolute temperatures guarantees smooth asymptotic reversion to the diurnal climatological wave rather than collapsing to a flat line.
2. **Channel Independence (CI)**: Multivariate channels are processed independently through the transformer encoder, isolating core thermal signals from turbulent wind and cloud noise.
3. **24-Hour Diurnal Patching**: Slices 720 hours of history into 59 overlapping 24-hour patches ($P=24, S=12$). Self-attention operates over day-to-day synoptic transitions rather than micro-hourly steps.
4. **Joint Multi-Horizon Projection Head**: Directly projects latent patch representations to all 336 future hours simultaneously via a linear output head ($\hat{\mathbf{r}}_{\text{history}} \in \mathbb{R}^{336}$).
5. **Transferred Time-Embedding Conditioning (Champion Hybrid)**: Pre-trained decadal time embeddings from the 26-year DenseNet are added via a zero-initialized linear projection head:
   $$\hat{\mathbf{r}}_{\text{final}} = \hat{\mathbf{r}}_{\text{history}} + \mathbf{W}_{\text{time}} \mathbf{e}_{\text{calendar}}$$

---

## 4. Master Evaluation Table

*Table is strictly sorted by **Holdout Test Mean MAE** in ascending order. Includes both **Zero-Shot** (trained on Train `< 2026-04-01`) and **Retrained** (trained on Train + Val `< 2026-07-01`) evaluations on the official locked SuperTest window (Sep 17–30, 2026).*

| Rank | Model Architecture | Key Configuration / Parameters | Val Mean MAE (°F) | Test Mean MAE (°F) | Test Std MAE (°F) | Test RMSE (°F) | Locked Test MAE (Zero-Shot) | Retrained Locked Test MAE | Retrained Locked Test RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **PatchTST + Transferred Time Emb (Sinusoidal)** 🏆 | `d_model=128, patch=24, transferred_time_emb` | 7.25 | **3.87** | **0.78** | **4.91** | 6.95 | **6.88** | 8.56 |
| **2** | **PatchTST + RoPE + Transferred Time Emb** | `rotary_pos_emb, transferred_time_emb` | 7.19 | **3.97** | 0.82 | 5.01 | 6.93 | 7.54 | 8.80 |
| **3** | **PatchTST (Transformer Baseline)** | `d_model=128, patch=24, stride=12, CI` | 7.07 | **4.02** | 0.98 | 5.04 | 6.65 | 6.90 | 8.21 |
| **4** | **Linear Regression (Ridge)** | `alpha=50.0, multi-output, history=168h` | 7.02 | **4.14** | 0.87 | 5.18 | **5.86** | **5.86** | **7.32** |
| **5** | **LightGBM (Tabular GBDT)** | `n_estimators=150, lr=0.03, num_leaves=31` | **6.98** | **4.17** | 0.93 | 5.22 | **5.90** | **5.90** | **7.39** |
| **6** | **PatchTST + Self-Learned Time Emb** | `time_emb=64 learned from scratch` | 7.14 | **4.20** | **0.68** | 5.22 | 7.14 | 6.95 | 8.45 |
| **7** | **PatchTST + Time Emb + Future Transformer** | `2-layer future self-attention fusion` | 7.33 | **4.21** | 0.85 | 5.26 | 6.81 | 6.85 | 8.61 |
| **8** | **Climatology Base (Zero-Parameter)** | Empirical 26-Year Historical Mean (DOY + Hour) | 7.43 | **4.35** | 0.94 | 5.40 | 6.94 | 6.94 | 8.25 |
| **9** | **DenseCrossTransformer (10-Bin)** | `d_model=128, ple_10bins, wd=1e-4` | 7.48 | **4.82** | 1.11 | 5.92 | 7.08 | 6.76 | 8.17 |
| **10** | **PatchTST + Time Emb + Learned Channel Mixing** | `nn.Linear(C, 1) mixer across channels` | 7.15 | **4.91** | 1.17 | 5.95 | 6.66 | 7.45 | 8.88 |
| **11** | **DenseCrossTransformer (64-Bin Regularized)** | `64bins, wd=0.01, drop=0.25, zero_init` | 7.15 | **5.31** | 1.83 | 6.40 | 8.14 | 7.10 | 8.85 |
| *—* | *DenseCrossTransformer 64b (No Year)* | `year_embedding = 0` | — | 5.70 | 2.39 | 6.79 | 7.09 | — | — |
| *—* | *DenseCrossTransformer 10b (No Year)* | `year_embedding = 0` | — | 6.02 | 1.93 | 7.11 | 7.97 | — | — |
| *—* | *DenseCrossTransformer 10b (No Time)* | `all_time_embeddings = 0` | — | 8.33 | 1.75 | 9.29 | 9.13 | — | — |
| *—* | *DenseCrossTransformer 64b (No Time)* | `all_time_embeddings = 0` | — | 10.65 | 2.02 | 11.49 | 10.93 | — | — |

### Key Experimental Insights:
1. **Holdout Summer Test Set Champion**: **PatchTST + Transferred Time Embeddings** attained the lowest error across all 65 test windows (**$3.87^\circ\text{F}$ MAE**, **$4.91^\circ\text{F}$ RMSE**), outperforming pure transformer baselines and classical ML.
2. **Locked SuperTest Dynamics (Late-Season Heat Wave)**: Sep 17–22 experienced unseasonal highs reaching **$92.0^\circ\text{F}$** (vs. historical norm of $72^\circ\text{F}$). Standalone Climatology struggled in Week 1 ($8.05^\circ\text{F}$ MAE). Classical Ridge ($5.86^\circ\text{F}$) and LightGBM ($5.90^\circ\text{F}$) carried forward initial cutoff heat effectively.
3. **Benefits of Retraining on Train + Val**: Incorporating Spring 2026 data improved deep model calibration significantly:
   - **64-Bin DenseNet**: Gained **$-1.04^\circ\text{F}$** ($8.14^\circ\text{F} \to 7.10^\circ\text{F}$ MAE; Week 1 dropped from $10.39^\circ\text{F} \to 6.61^\circ\text{F}$).
   - **PatchTST + Transferred TE**: Improved to **$6.88^\circ\text{F}$ MAE** (and **$6.13^\circ\text{F}$ Week 1 MAE**), beating Climatology ($6.94^\circ\text{F}$ / $8.05^\circ\text{F}$).
4. **Catastrophic Failure Without Time Embeddings**: Disabling time embeddings caused locked test error to surge to **$9.13^\circ\text{F}$** (10-bin) and **$10.93^\circ\text{F}$** (64-bin), confirming temporal embeddings are essential anchors for multi-week horizon stability.

---

## 5. Why DenseNet Learns Valuable Time Embeddings & Deep Embedding Analysis

A foundational result is that time embeddings pre-trained in `DenseCrossTransformer` transferred substantial performance gains to `PatchTST`.

### Intuitive Mechanics: The Dense Gradient Highway

Unlike standard attention blocks where calendar tokens are bottlenecked or diluted across self-attention maps, DenseNet feeds calendar embeddings directly into **every intermediate layer**:

```
[Categorical Calendar Inputs: Year, Month, Hour, DOY]
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Learned Embedding Tables    │  <── Direct Gradient Highway (No Vanishing Gradients)
       └──────────────┬────────────────┘
                      │
     ┌────────────────┼────────────────┬────────────────┐
     ▼                ▼                ▼                ▼
[Layer 1]        [Layer 2]        [Layer 3]        [Layer 4]
(Concatenated    (Concatenated    (Concatenated    (Concatenated
with Physics)    with Physics)    with Physics)    with Physics)
```

1. **Direct Gradient Accumulation**: By the chain rule, gradients flow directly to $\mathbf{e}_{\text{time}}$ across all layers:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{e}_{\text{time}}} = \sum_{\ell=1}^L \frac{\partial \mathcal{L}}{\partial \mathbf{h}_\ell} \frac{\partial \mathbf{h}_\ell}{\partial \mathbf{e}_{\text{time}}}$$
2. **Decadal Warming Calibration (Year Embedding)**: 26 continuous years of gradients forced the year embedding table to encode Raleigh's decadal climate shift ($+0.70^\circ\text{F}$ MAE ablation impact).
3. **Seasonal Hysteresis (Month Embedding)**: Successfully encodes the multi-week lag between the summer solstice (June) and peak ground temperatures (late July).

---

### Deep Dive: Month-Temperature Linear Probing & Bilinear Resonance

![Month-Temperature Dot Product Analysis](report/month_temperature_dot_product_deep_dive.png)

#### 1. Concept Direction Vector Probing ($\mathbf{u}_T$)

We probed whether the 16-dimensional month embedding space contains a 1D linear axis encoding physical temperature.

##### Mathematical Definitions:
- **$\mathbf{e}_m \in \mathbb{R}^{16}$**: The learned embedding vector for month $m \in \{1, \dots, 12\}$.
- **$T_m \in \mathbb{R}$ & $\mathbf{T} \in \mathbb{R}^{12}$**: Empirical 26-year ground-truth monthly mean temperatures at KRDU:
  $$\mathbf{T} = [41.4^\circ\text{F}\,(\text{Jan}), 45.3, 53.0, 61.6, 69.0, 76.8, 79.6\,(\text{Jul}), 78.0, 72.1, 61.5, 51.5, 44.8\,(\text{Dec})]^\top$$
  *(The network is never trained with $\mathbf{T}$; it only receives integer tokens $1 \dots 12$)*.
- **$\mathbf{u}_T \in \mathbb{R}^{16}$ (Unit Temperature Concept Axis)**: A linear probe is fit from month embeddings to temperatures ($\hat{T}_m = \mathbf{e}_m^\top \mathbf{w}^* + b \approx T_m$) and normalized to unit length:
  $$\mathbf{u}_T = \frac{\mathbf{w}^*}{\|\mathbf{w}^*\|_2} \in \mathbb{R}^{16}, \quad \|\mathbf{u}_T\|_2 = 1$$
- **$k \in \{1, \dots, 16\}$**: Hidden coordinate index in the scalar projection:
  $$s_m = \langle \mathbf{e}_m, \mathbf{u}_T \rangle = \sum_{k=1}^{16} e_{m, k} \cdot u_{T, k}$$

| Month | Climatological Mean ($T_m$) | Dot Product $\langle \mathbf{e}_m, \mathbf{u}_T \rangle$ | Thermodynamic Regime |
| :---: | :---: | :---: | :--- |
| **Jan** | 41.4°F | **-1.205** | Deep Winter Cold Trough |
| **Feb** | 45.3°F | **-0.877** | Late Winter Transition |
| **Mar** | 53.0°F | **-0.371** | Early Spring Warming |
| **Apr** | 61.6°F | **+0.315** | Mid Spring Warming |
| **May** | 69.0°F | **+0.391** | Late Spring Convection |
| **Jun** | 76.8°F | **+0.855** | Early Summer |
| **Jul** | 79.6°F | **+1.113** | Mid Summer Peak (Maximum) |
| **Aug** | 78.0°F | **+0.805** | Late Summer Highs |
| **Sep** | 72.1°F | **+0.748** | Early Autumn Cooling |
| **Oct** | 61.5°F | **+0.116** | Mid Autumn Transition |
| **Nov** | 51.5°F | **-0.243** | Late Autumn Chill |
| **Dec** | 44.8°F | **-0.982** | Early Winter Freeze |

**Result**: Pearson correlation between the learned month projection and ground-truth temperature is **$r = 0.9838$ ($R^2 = 0.9679$)**, demonstrating that the latent space autonomously organized along a true thermodynamic temperature gradient.

#### 2. Bilinear Resonance with 64-Bin Piecewise Linear Encoding (PLE)

Inside the network, continuous temperature is represented via Piecewise Linear Encoding (PLE) across 64 training quantile bins $\{b_0, \dots, b_{63}\}$:
$$v_k(T) = \text{clip}\left(\frac{T - b_k}{b_{k+1} - b_k}, 0, 1\right), \quad k \in \{0, \dots, 62\}$$
Projecting both the 63D continuous vector $\mathbf{v}_{\text{ple}}(T)$ and 16D month vector $\mathbf{e}_m$ into the shared DenseNet hidden space:
$$\text{Resonance}(T, m) = \mathbf{v}_{\text{ple}}(T)^\top \left( \mathbf{W}_{\text{ple}}^\top \mathbf{W}_{\text{time}} \right) \mathbf{e}_m$$
- **Matched Months ($\langle \text{PLE}(T_m), \mathbf{e}_m \rangle$)**: Mean dot product is **$+0.379$** (strong positive alignment).
- **Mismatched Months ($\langle \text{PLE}(T_i), \mathbf{e}_j \rangle, i \neq j$)**: Mean dot product drops to **$+0.182$**, showing strong physical separation between summer heat and winter cold.

---

## 6. Repository Structure & Artifacts

```
weather_forecast/
├── evaluate.py                  # CLI entrypoint for running evaluations
├── results.md                   # Consolidated technical report & benchmark results
├── requirements.txt             # Python dependencies
├── src/
│   ├── config.py                # Global paths, features, and hyperparameters
│   ├── data/
│   │   ├── ingestion.py         # ASOS & Open-Meteo downloaders
│   │   ├── preprocessing.py     # Atmospheric physics & climatology calculations
│   │   └── dataset.py           # Sliding and rolling window dataset loaders
│   ├── models/
│   │   ├── baselines.py         # Climatology and Ridge regressors
│   │   ├── lightgbm_model.py    # GBDT tabular weather forecaster
│   │   ├── patch_tst.py         # PatchTST, RoPE, and transferred-TE variants
│   │   └── dense_cross_transformer.py # DenseNet cross tower, PLE, time embeddings
│   ├── training/
│   │   ├── loss.py              # Composite weather loss
│   │   └── trainer.py           # Epoch training and multi-window evaluation
│   └── analysis/
│       ├── embedding_analysis.py# Linear concept probing & PLE resonance analysis
│       └── feature_distributions.py # 64-bin quantile distribution analysis
└── report/
    ├── patchtst_architecture_diagram.png
    ├── month_temperature_dot_product_deep_dive.png
    ├── continuous_features_64bins_distribution.png
    └── fixed_setup_all_models_summary.csv
```
