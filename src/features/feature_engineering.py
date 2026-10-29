from typing import List

import pandas as pd

from src.features.financial_ratios import add_ratios, has_statements
from src.features.temporal_features import add_temporal
from src.utils.constants import ID_COLS, RATIO_COLS, TARGET, TEMPORAL_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if has_statements(out):
        out = add_ratios(out)
        out = add_temporal(out)
    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    if all(c in df.columns for c in RATIO_COLS):
        return RATIO_COLS + [c for c in TEMPORAL_COLS if c in df.columns]
    skip = set(ID_COLS + [TARGET])
    return [c for c in df.select_dtypes(include="number").columns if c not in skip]
