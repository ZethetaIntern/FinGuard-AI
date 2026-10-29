import numpy as np
import pandas as pd

from src.features.financial_ratios import safe_div

THRESHOLD = -1.78


def beneish_mscore(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    dsri = safe_div(safe_div(d["receivables"], d["sales"]), safe_div(d["receivables_prev"], d["sales_prev"]))
    gm = safe_div(d["sales"] - d["cogs"], d["sales"])
    gm_prev = safe_div(d["sales_prev"] - d["cogs_prev"], d["sales_prev"])
    gmi = safe_div(gm_prev, gm)
    aq = 1 - safe_div(d["current_assets"] + d["ppe"], d["total_assets"])
    aq_prev = 1 - safe_div(d["current_assets_prev"] + d["ppe_prev"], d["total_assets_prev"])
    aqi = safe_div(aq, aq_prev)
    sgi = safe_div(d["sales"], d["sales_prev"])
    dep_rate = safe_div(d["depreciation"], d["depreciation"] + d["ppe"])
    dep_rate_prev = safe_div(d["depreciation_prev"], d["depreciation_prev"] + d["ppe_prev"])
    depi = safe_div(dep_rate_prev, dep_rate)
    sgai = safe_div(safe_div(d["sga"], d["sales"]), safe_div(d["sga_prev"], d["sales_prev"]))
    lev = safe_div(d["current_liabilities"] + d["long_term_debt"], d["total_assets"])
    lev_prev = safe_div(d["current_liabilities_prev"] + d["long_term_debt_prev"], d["total_assets_prev"])
    lvgi = safe_div(lev, lev_prev)
    tata = safe_div(d["net_income"] - d["cfo"], d["total_assets"])
    m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    flag = np.select([m.isna(), m > THRESHOLD], ["Unknown", "Likely manipulator"], default="Unlikely")
    return pd.DataFrame({"m_score": m.values, "m_flag": flag}, index=df.index)
