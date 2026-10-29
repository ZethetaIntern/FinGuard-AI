import math
from typing import Optional, Tuple

import pandas as pd

from src.data.clean_data import clean_dataframe
from src.data.load_data import load_config, load_universe
from src.features.feature_engineering import build_features
from src.features.financial_ratios import has_statements
from src.features.temporal_features import has_prev
from src.finance.warning_signals import compute_warning_signals
from src.modeling.predict import predict_proba
from src.utils.constants import OUTPUTS_DIR, PROFILES_FILE, RISK_QUEUE_FILE

LEAN_COLS = [
    "company_id", "company_name", "sector", "year", "risk_probability", "risk_rank",
    "risk_percentile", "in_top_queue", "z_score", "z_zone", "m_score", "m_flag",
    "f_score", "f_label", "warning_count", "warning_text",
]


def build_company_profiles(df: pd.DataFrame) -> pd.DataFrame:
    base = build_features(clean_dataframe(df, drop_missing_target=False, drop_dupes=False))
    base = base.reset_index(drop=True)
    base["risk_probability"] = predict_proba(df)
    if has_statements(base) and has_prev(base):
        base = pd.concat([base, compute_warning_signals(base)], axis=1)
    base["risk_rank"] = base["risk_probability"].rank(ascending=False, method="first").astype(int)
    base["risk_percentile"] = (1 - (base["risk_rank"] - 1) / max(len(base), 1)).round(4)
    return base


def generate_queue(df: Optional[pd.DataFrame] = None, top_pct: Optional[float] = None,
                   save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = load_universe()
    top_pct = float(top_pct if top_pct is not None else load_config()["top_pct"])
    profiles = build_company_profiles(df)
    k = max(1, int(math.ceil(len(profiles) * top_pct)))
    profiles["in_top_queue"] = profiles["risk_rank"] <= k
    profiles = profiles.sort_values("risk_rank").reset_index(drop=True)
    queue = profiles[profiles["in_top_queue"]].copy()
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        profiles.to_csv(PROFILES_FILE, index=False)
        queue[[c for c in LEAN_COLS if c in queue.columns]].to_csv(RISK_QUEUE_FILE, index=False)
    return profiles, queue


if __name__ == "__main__":
    p, q = generate_queue()
    print(q[[c for c in LEAN_COLS if c in q.columns]].head(10).to_string(index=False))
