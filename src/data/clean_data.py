import numpy as np
import pandas as pd

from src.utils.constants import PREV_COLS, STATEMENT_COLS, TARGET


def clean_dataframe(df: pd.DataFrame, drop_missing_target: bool = True,
                    drop_dupes: bool = True) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in STATEMENT_COLS + PREV_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if drop_dupes:
        out = out.drop_duplicates()
    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
        if drop_missing_target:
            out = out.dropna(subset=[TARGET])
            out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)
