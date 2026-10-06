"""
Data handling module: ingestion, preprocessing, and PyTorch dataset loading.
"""

from src.data.preprocessing import (
    calculate_solar_features,
    enrich_meteorological_dataset,
    compute_historical_climatology,
    compute_metrics,
    compute_anomaly_correlation,
)
from src.data.dataset import (
    get_window_indices,
    FixedRollingDataset,
    SlidingTrainDataset,
)

__all__ = [
    "calculate_solar_features",
    "enrich_meteorological_dataset",
    "compute_historical_climatology",
    "compute_metrics",
    "compute_anomaly_correlation",
    "get_window_indices",
    "FixedRollingDataset",
    "SlidingTrainDataset",
]
