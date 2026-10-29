import numpy as np
import pandas as pd

from src.features.financial_ratios import safe_div


def altman_zscore(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    z = (
        1.2 * safe_div(d["current_assets"] - d["current_liabilities"], d["total_assets"])
        + 1.4 * safe_div(d["retained_earnings"], d["total_assets"])
        + 3.3 * safe_div(d["ebit"], d["total_assets"])
        + 0.6 * safe_div(d["market_value_equity"], d["total_liabilities"])
        + 1.0 * safe_div(d["sales"], d["total_assets"])
    )
    zone = np.select([z.isna(), z > 2.99, z >= 1.81], ["Unknown", "Safe", "Grey"], default="Distress")
    return pd.DataFrame({"z_score": z.values, "z_zone": zone}, index=df.index)
