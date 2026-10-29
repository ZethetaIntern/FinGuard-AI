from typing import Iterable, Optional

import pandas as pd

from src.utils.constants import TARGET


def validate_dataframe(df: pd.DataFrame, require_target: bool = True,
                       required_cols: Optional[Iterable[str]] = None) -> dict:
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")
    missing = [c for c in (required_cols or []) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_pct": {k: round(float(v), 2) for k, v in (df.isna().mean() * 100).items() if v > 0},
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }
    if require_target:
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found.")
        vals = set(pd.Series(df[TARGET]).dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            raise ValueError("Target must be binary (0/1).")
        if len(vals) < 2:
            raise ValueError("Target has only one class.")
        report["positive_rate"] = float(df[TARGET].mean())
    return report
