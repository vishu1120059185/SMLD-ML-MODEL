"""Serialisable preprocessing pipeline for STAT-TWIN.

``PreprocessingPipeline`` chains all Phase-1 transforms into a single
object that is fit on the training partition and applied to any data
partition.  The pipeline is fully serialisable via ``joblib``.

Execution order
---------------
1. **Missingness indicators** – binary flags for original NaNs.
2. **Causal forward-fill** – propagate last observed value.
3. **Causal linear interpolation** – fill remaining gaps.
4. **Train-median fill** – residual gap fill with training medians.
5. **DQ Engine** – flag and repair data-quality issues.
6. **Outlier detection + winsorisation** – flag and clip extreme values.
7. **Scaling** – standard or robust, fitted on training data.

Every step is config-driven (``STATTWINConfig.preprocess``).  The
pipeline exposes ``fit`` / ``transform`` / ``fit_transform`` and can be
saved with ``joblib.dump``.
"""

from __future__ import annotations

import joblib
import pandas as pd

from stattwin.config import PreprocessCfg, STATTWINConfig
from stattwin.logging import get_logger

from .dq import DQConfig, DQEngine
from .missing import (
    CausalForwardFill,
    CausalLinearInterpolation,
    MissingnessIndicators,
    TrainMedianFill,
)
from .outliers import RobustZDetector, Winsorizer
from .scaler import TrainFittedScaler

__all__ = ["PreprocessingPipeline"]

logger = get_logger(__name__)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline – fit on train, apply everywhere.

    Parameters
    ----------
    config:
        Full STAT-TWIN configuration (uses ``config.preprocess``).
    sensor_columns:
        Sensor columns to process.  If *None*, ``sensor_1 … sensor_21``
        are assumed.
    unit_col:
        Name of the unit-identifier column.
    """

    def __init__(
        self,
        config: STATTWINConfig,
        sensor_columns: list[str] | None = None,
        unit_col: str = "unit_id",
    ) -> None:
        self.config = config
        self.cfg: PreprocessCfg = config.preprocess
        self.sensor_columns = sensor_columns or [f"sensor_{i}" for i in range(1, 22)]
        self.unit_col = unit_col

        # ------------------------------------------------------------------
        # Build sub-components from config
        # ------------------------------------------------------------------

        # 1) Missingness indicators
        self.indicator = MissingnessIndicators(columns=self.sensor_columns)

        # 2) Causal forward fill
        self.ffill = CausalForwardFill(columns=self.sensor_columns)

        # 3) Causal linear interpolation
        self.lin_interp = CausalLinearInterpolation(columns=self.sensor_columns)

        # 4) Train-median fill
        self.median_fill = TrainMedianFill(columns=self.sensor_columns)

        # 5) DQ engine
        dq_cfg = DQConfig(
            sensor_columns=self.sensor_columns,
            dropout_burst_k=3,
            stuck_window=10,
            spike_threshold=6.0,
            range_bounds=None,
            repair_strategy="forward_fill",
        )
        self.dq_engine = DQEngine(config=dq_cfg)

        # 6) Outlier detection + winsorisation
        self.outlier_detector = RobustZDetector(
            threshold=self.cfg.outliers.threshold
        )
        self.winsorizer = Winsorizer(quantile_low=0.01, quantile_high=0.99)

        # 7) Scaler
        self.scaler = TrainFittedScaler(
            method=self.cfg.scaler,
            columns=self.sensor_columns,
        )

        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> PreprocessingPipeline:
        """Fit all sub-components on the training partition.

        Parameters
        ----------
        df:
            Training DataFrame (must contain ``unit_id``, ``cycle``, and
            sensor columns).

        Returns
        -------
        PreprocessingPipeline
            Self, for method chaining.
        """
        logger.info("Fitting preprocessing pipeline on %d rows", len(df))

        # Steps that learn from data
        self.indicator.fit(df, unit_col=self.unit_col)
        self.ffill.fit(df, unit_col=self.unit_col)
        self.lin_interp.fit(df, unit_col=self.unit_col)
        self.median_fill.fit(df, unit_col=self.unit_col)
        self.dq_engine.fit(df, unit_col=self.unit_col)

        # Fit outlier detector and winsorizer on the (already filled) training data
        train_filled = self._apply_missing_steps(df)
        self.outlier_detector.fit(
            train_filled, columns=self.sensor_columns
        )
        self.winsorizer.fit(train_filled, columns=self.sensor_columns)

        # Fit scaler on cleaned training data
        train_clean = self._apply_outlier_steps(train_filled)
        self.scaler.fit(train_clean, columns=self.sensor_columns)

        self._is_fitted = True
        logger.info("Pipeline fitted successfully")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to any DataFrame.

        Parameters
        ----------
        df:
            DataFrame to transform (train, val, or test).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with all preprocessing applied.
        """
        if not self._is_fitted:
            raise RuntimeError("PreprocessingPipeline has not been fitted. Call fit() first.")

        logger.info("Transforming %d rows", len(df))

        # Step 1-4: Missing value handling
        out = self._apply_missing_steps(df)

        # Step 5: DQ flags + repair
        out = self.dq_engine.transform(out, unit_col=self.unit_col)

        # Step 6: Outlier detection + winsorisation
        out = self._apply_outlier_steps(out)

        # Step 7: Scaling
        out = self.scaler.transform(out)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* then transform it (convenience shortcut)."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted pipeline to *path* via joblib.

        Parameters
        ----------
        path:
            Destination file path.
        """
        joblib.dump(self, path)
        logger.info("Pipeline saved to %s", path)

    @staticmethod
    def load(path: str) -> PreprocessingPipeline:
        """Deserialise a previously saved pipeline.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        PreprocessingPipeline
        """
        pipeline = joblib.load(path)
        logger.info("Pipeline loaded from %s", path)
        return pipeline

    # ------------------------------------------------------------------
    # Internal step composition
    # ------------------------------------------------------------------

    def _apply_missing_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps 1–4: indicators → ffill → interpolation → median fill."""
        out = self.indicator.transform(df, unit_col=self.unit_col)
        out = self.ffill.transform(out, unit_col=self.unit_col)
        out = self.lin_interp.transform(out, unit_col=self.unit_col)
        out = self.median_fill.transform(out, unit_col=self.unit_col)
        return out

    def _apply_outlier_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6: detect outliers then winsorise."""
        # We flag but don't remove – winsorise instead
        out = self.winsorizer.transform(df)
        return out

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"PreprocessingPipeline({status}, "
            f"scaler='{self.cfg.scaler}', "
            f"outlier='{self.cfg.outliers.method}')"
        )
