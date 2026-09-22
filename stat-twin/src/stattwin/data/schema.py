"""Column definitions, sensor names, and failure horizons for C-MAPSS data.

C-MAPSS turbofan engine run-to-failure files are space-separated with no
header and contain 26 columns:

    unit_id, cycle, op_setting_1, op_setting_2, op_setting_3,
    sensor_1 … sensor_21
"""

from __future__ import annotations

from typing import Final, List

# ---------------------------------------------------------------------------
# 26 columns as they appear in the raw C-MAPSS text files
# ---------------------------------------------------------------------------

COLUMN_NAMES: Final[List[str]] = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

assert len(COLUMN_NAMES) == 26, "C-MAPSS must have exactly 26 columns"

# ---------------------------------------------------------------------------
# Friendly / readable names for each sensor
# ---------------------------------------------------------------------------

SENSOR_NAMES: Final[dict[str, str]] = {
    "sensor_1": "Fan inlet temperature (deg R)",
    "sensor_2": "Total pressure at fan outlet (psia)",
    "sensor_3": "Static pressure at fan outlet (psia)",
    "sensor_4": "Pressure ratio at fan outlet",
    "sensor_5": "Physical fan speed (rpm)",
    "sensor_6": "Corrected fan speed (rpm)",
    "sensor_7": "Bypass duct pressure (psia)",
    "sensor_8": "Bleed enthalpy (BTU/min)",
    "sensor_9": "Desired fan speed (rpm)",
    "sensor_10": "Designed bleed enthalpy (BTU/min)",
    "sensor_11": "Pressure ratio at burner outlet",
    "sensor_12": "Fan outlet temperature (deg R)",
    "sensor_13": "Static pressure at burner outlet (psia)",
    "sensor_14": "Fan vibration (mil)",
    "sensor_15": "Burner exit A/R ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Fan inlet pressure ratio",
    "sensor_18": "Fan outlet temperature (deg R)",
    "sensor_19": "Fuel flow (pps)",
    "sensor_20": "Fan speed (rpm)",
    "sensor_21": "Engine condition monitoring index",
}

# Operational setting column names (convenience)
OP_SETTINGS: Final[List[str]] = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]

# ---------------------------------------------------------------------------
# Failure horizons for binary degradation labels
# ---------------------------------------------------------------------------

FAILURE_HORIZONS: Final[List[int]] = [10, 20, 30, 40, 50]
"""Horizons *h* at which we define ``y_h = 1[rul <= h]``."""

# Column name template for the binary failure label at a given horizon
LABEL_COL_TEMPLATE: Final[str] = "fail_h{h}"
"""``LABEL_COL_TEMPLATE.format(h=20)`` → ``"fail_h20"``."""


def label_col_for(horizon: int) -> str:
    """Return the canonical column name for a failure label at *horizon*."""
    return LABEL_COL_TEMPLATE.format(h=horizon)
