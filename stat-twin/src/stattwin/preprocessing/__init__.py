"""Preprocessing module for STAT-TWIN.

Phase 1: missing-value imputation, outlier handling, scaling, and
data-quality flagging.  All transforms are fit on train portions only,
causal (features at cycle *t* depend only on cycles *≤ t*), and
serialisable via joblib.
"""

from stattwin.preprocessing.dq import DQEngine, DQFlags, separate_anomalies
from stattwin.preprocessing.missing import (
    CausalForwardFill,
    CausalLinearInterpolation,
    MissingnessIndicators,
    TrainMedianFill,
)
from stattwin.preprocessing.outliers import (
    IQRFencesDetector,
    RobustZDetector,
    Winsorizer,
    ZScoreDetector,
)
from stattwin.preprocessing.pipeline import PreprocessingPipeline
from stattwin.preprocessing.scaler import (
    PerConditionScaler,
    TrainFittedScaler,
)

__all__ = [
    "CausalForwardFill",
    "CausalLinearInterpolation",
    "DQEngine",
    "DQFlags",
    "IQRFencesDetector",
    "MissingnessIndicators",
    "PerConditionScaler",
    "PreprocessingPipeline",
    "RobustZDetector",
    "separate_anomalies",
    "TrainFittedScaler",
    "TrainMedianFill",
    "Winsorizer",
    "ZScoreDetector",
]
