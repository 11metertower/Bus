"""Rules for handling 'Other' direct-driver dispatches.

In the MVP data, driver column value 'Other' means an external/direct driver
will drive the assigned vehicle. The system must assign a vehicle only and must
not consume a military driver resource.
"""

from __future__ import annotations

import pandas as pd

from .utils import clean_text

OTHER_DRIVER_MODE = "OTHER_DIRECT_DRIVER"
MILITARY_DRIVER_MODE = "MILITARY_DRIVER_REQUIRED"


def is_other_driver_value(value) -> bool:
    text = clean_text(value)
    return text is not None and text.strip().lower() == "other"


def add_driver_mode_columns(df: pd.DataFrame, driver_col: str = "driver_raw") -> pd.DataFrame:
    out = df.copy()
    if driver_col not in out.columns:
        out["is_other"] = False
        out["requires_driver"] = True
        out["driver_mode"] = MILITARY_DRIVER_MODE
        return out
    out["is_other"] = out[driver_col].apply(is_other_driver_value)
    out["requires_driver"] = ~out["is_other"]
    out["driver_mode"] = out["is_other"].map(
        {True: OTHER_DRIVER_MODE, False: MILITARY_DRIVER_MODE}
    )
    return out
