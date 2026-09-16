"""Excel loading utilities for G-LIFT MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import RAW_DIR, RAW_FILE_NAMES


def read_first_sheet(path: Path) -> pd.DataFrame:
    """Read the first sheet of an Excel workbook as a DataFrame."""
    return pd.read_excel(path, sheet_name=0)


def load_raw_inputs(raw_dir: Path = RAW_DIR, file_names: Dict[str, str] = RAW_FILE_NAMES) -> Dict[str, pd.DataFrame]:
    """Load all expected raw Excel files.

    Raises FileNotFoundError with a clear message if a required workbook is
    missing. This is intentional: preprocessing should fail early when the raw
    data package is incomplete.
    """
    raw_dir = Path(raw_dir)
    tables: Dict[str, pd.DataFrame] = {}
    missing = []
    for key, name in file_names.items():
        path = raw_dir / name
        if not path.exists():
            missing.append(str(path))
            continue
        tables[key] = read_first_sheet(path)
    if missing:
        raise FileNotFoundError("Missing raw input files:\n" + "\n".join(missing))
    return tables
