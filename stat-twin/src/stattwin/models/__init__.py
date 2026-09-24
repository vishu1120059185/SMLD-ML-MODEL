"""STAT-TWIN prediction models module.

All models implement the ``BaseModel`` interface providing ``fit``,
``predict_proba``, ``predict_rul``, and ``score_raw`` methods.
"""

from stattwin.models.anomaly import AnomalyModel
from stattwin.models.base import BaseModel, ModelResult
from stattwin.models.gru import GRUModel
from stattwin.models.hybrid import HybridModel
from stattwin.models.logistic import LogisticModel
from stattwin.models.random_forest import RandomForestModel
from stattwin.models.threshold import ThresholdModel
from stattwin.models.xgboost_model import XGBoostModel

__all__ = [
    "AnomalyModel",
    "BaseModel",
    "GRUModel",
    "HybridModel",
    "LogisticModel",
    "ModelResult",
    "RandomForestModel",
    "ThresholdModel",
    "XGBoostModel",
]
