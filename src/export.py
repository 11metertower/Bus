"""Export helpers for G-LIFT MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def export_tables_to_excel(tables: Dict[str, pd.DataFrame], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
