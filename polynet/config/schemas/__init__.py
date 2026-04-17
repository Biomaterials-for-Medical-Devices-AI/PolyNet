"""
polynet.config.schemas
=======================
Pydantic configuration schemas for all pipeline stages.

Import directly from this package for convenience::

    from polynet.config.schemas import (
        DataConfig,
        GeneralConfig,
        PlottingConfig,
        RepresentationConfig,
        TrainGNNConfig,
        TrainTMLConfig,
    )
"""

from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.explainability import ExplainabilityConfig
from polynet.config.schemas.feature_preprocessing import FeatureTransformConfig
from polynet.config.schemas.general import GeneralConfig
from polynet.config.schemas.plotting import PlottingConfig
from polynet.config.schemas.representation import RepresentationConfig
from polynet.config.schemas.split_data import SplitConfig
from polynet.config.schemas.target_preprocessing import TargetTransformConfig
from polynet.config.schemas.training import TrainGNNConfig, TrainTMLConfig

__all__ = [
    "DataConfig",
    "ExplainabilityConfig",
    "GeneralConfig",
    "FeatureTransformConfig",
    "PlottingConfig",
    "RepresentationConfig",
    "SplitConfig",
    "TargetTransformConfig",
    "TrainGNNConfig",
    "TrainTMLConfig",
]
