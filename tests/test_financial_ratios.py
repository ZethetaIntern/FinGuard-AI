import numpy as np
import pandas as pd

from src.data.download_data import generate_synthetic_financials
from src.features.feature_engineering import build_features, get_feature_columns
from src.features.financial_ratios import safe_div
from src.utils.constants import RATIO_COLS


def test_safe_div_zero():
    r = safe_div([1, 2], [0, 2])
    assert np.isnan(r.iloc[0]) and r.iloc[1] == 1


def test_ratios_present():
    df = build_features(generate_synthetic_financials(100, seed=2))
    assert all(c in df.columns for c in RATIO_COLS)
    assert "bankrupt" not in get_feature_columns(df)
