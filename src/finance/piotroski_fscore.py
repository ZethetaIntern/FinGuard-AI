import numpy as np
import pandas as pd

from src.features.financial_ratios import safe_div


def piotroski_fscore(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    roa = safe_div(d["net_income"], d["total_assets"])
    roa_prev = safe_div(d["net_income_prev"], d["total_assets_prev"])
    lev = safe_div(d["long_term_debt"], d["total_assets"])
    lev_prev = safe_div(d["long_term_debt_prev"], d["total_assets_prev"])
    cr = safe_div(d["current_assets"], d["current_liabilities"])
    cr_prev = safe_div(d["current_assets_prev"], d["current_liabilities_prev"])
    gm = safe_div(d["sales"] - d["cogs"], d["sales"])
    gm_prev = safe_div(d["sales_prev"] - d["cogs_prev"], d["sales_prev"])
    at = safe_div(d["sales"], d["total_assets"])
    at_prev = safe_div(d["sales_prev"], d["total_assets_prev"])
    signals = pd.DataFrame({
        "f_roa_pos": roa > 0,
        "f_cfo_pos": d["cfo"] > 0,
        "f_droa_pos": roa > roa_prev,
        "f_accrual": d["cfo"] > d["net_income"],
        "f_lev_down": lev < lev_prev,
        "f_cr_up": cr > cr_prev,
        "f_no_dilution": d["shares_outstanding"] <= d["shares_outstanding_prev"],
        "f_gm_up": gm > gm_prev,
        "f_at_up": at > at_prev,
    }).fillna(False).astype(int)
    score = signals.sum(axis=1)
    label = np.select([score >= 7, score >= 4], ["Strong", "Moderate"], default="Weak")
    out = signals.copy()
    out["f_score"] = score.values
    out["f_label"] = label
    out.index = df.index
    return out
