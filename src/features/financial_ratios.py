import numpy as np
import pandas as pd

from src.utils.constants import RATIO_COLS, STATEMENT_COLS


def safe_div(a, b) -> pd.Series:
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    b = b.where(b != 0, np.nan)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def has_statements(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in STATEMENT_COLS)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STATEMENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing statement columns: {missing}")
    out = df.copy()
    d = out.reset_index(drop=True)
    ratios = pd.DataFrame({
        "wc_ta": safe_div(d["current_assets"] - d["current_liabilities"], d["total_assets"]),
        "re_ta": safe_div(d["retained_earnings"], d["total_assets"]),
        "ebit_ta": safe_div(d["ebit"], d["total_assets"]),
        "equity_tl": safe_div(d["market_value_equity"], d["total_liabilities"]),
        "sales_ta": safe_div(d["sales"], d["total_assets"]),
        "current_ratio": safe_div(d["current_assets"], d["current_liabilities"]),
        "debt_ratio": safe_div(d["total_liabilities"], d["total_assets"]),
        "roa": safe_div(d["net_income"], d["total_assets"]),
        "cfo_ta": safe_div(d["cfo"], d["total_assets"]),
    })[RATIO_COLS]
    ratios.index = out.index
    for c in RATIO_COLS:
        out[c] = ratios[c]
    return out
