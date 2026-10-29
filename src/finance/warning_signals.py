import pandas as pd

from src.features.financial_ratios import safe_div
from src.finance.altman_zscore import altman_zscore
from src.finance.beneish_mscore import beneish_mscore
from src.finance.piotroski_fscore import piotroski_fscore


def compute_warning_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return Z/M/F scores plus rule-based warning flags (kept separate from ML output)."""
    z = altman_zscore(df)
    m = beneish_mscore(df)
    f = piotroski_fscore(df)[["f_score", "f_label"]]
    d = df.reset_index(drop=True)
    flags = pd.DataFrame({
        "Low current ratio": safe_div(d["current_assets"], d["current_liabilities"]) < 1,
        "Negative ROA": safe_div(d["net_income"], d["total_assets"]) < 0,
        "High leverage": safe_div(d["total_liabilities"], d["total_assets"]) > 0.8,
        "Negative cash flow": d["cfo"] < 0,
        "Sales decline": d["sales"] < d["sales_prev"],
        "Z-score distress": z["z_zone"].values == "Distress",
        "Earnings manipulation risk": m["m_flag"].values == "Likely manipulator",
    }).fillna(False)
    flags.index = df.index
    out = pd.concat([z, m, f], axis=1)
    out["warning_count"] = flags.sum(axis=1).astype(int)
    out["warning_text"] = flags.apply(lambda r: "; ".join([k for k, v in r.items() if v]), axis=1)
    return out
