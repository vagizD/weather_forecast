"""
Models package: Classical Baselines, LightGBM, PatchTST architectures, and DenseCrossTransformer.
"""

from src.models.baselines import ClimatologyBase, ClimatologyRidgeForecaster
from src.models.lightgbm_model import LightGBMWeatherForecaster
from src.models.patch_tst import (
    PatchTST,
    PatchTSTWithTimeEmbedding,
    PatchTSTRoPE,
    ModularPatchTST,
)
from src.models.dense_cross_transformer import (
    DenseCrossTransformer,
    RegularizedDenseCrossTransformer,
    TimeEmbedding,
    PiecewiseLinearEncoding,
)

__all__ = [
    "ClimatologyBase",
    "ClimatologyRidgeForecaster",
    "LightGBMWeatherForecaster",
    "PatchTST",
    "PatchTSTWithTimeEmbedding",
    "PatchTSTRoPE",
    "ModularPatchTST",
    "DenseCrossTransformer",
    "RegularizedDenseCrossTransformer",
    "TimeEmbedding",
    "PiecewiseLinearEncoding",
]
