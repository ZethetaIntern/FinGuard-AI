import pandas as pd

from src.features.financial_ratios import safe_div
from src.utils.constants import PREV_COLS, STATEMENT_COLS, TEMPORAL_COLS


def has_prev(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in PREV_COLS) and all(c in df.columns for c in STATEMENT_COLS)


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    if not has_prev(df):
        return df.copy()
    out = df.copy()
    d = out.reset_index(drop=True)
    new = pd.DataFrame({
        "d_roa": safe_div(d["net_income"], d["total_assets"]) - safe_div(d["net_income_prev"], d["total_assets_prev"]),
        "d_debt_ratio": safe_div(d["total_liabilities"], d["total_assets"]) - safe_div(d["total_liabilities_prev"], d["total_assets_prev"]),
        "d_current_ratio": safe_div(d["current_assets"], d["current_liabilities"]) - safe_div(d["current_assets_prev"], d["current_liabilities_prev"]),
        "sales_growth": safe_div(d["sales"], d["sales_prev"]) - 1,
        "asset_growth": safe_div(d["total_assets"], d["total_assets_prev"]) - 1,
    })[TEMPORAL_COLS]
    new.index = out.index
    for c in TEMPORAL_COLS:
        out[c] = new[c]
    return out
