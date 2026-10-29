import json
from typing import Optional

import pandas as pd

from src.data.load_data import load_config, load_indian_data
from src.modeling.evaluate import evaluate
from src.modeling.predict import predict_proba
from src.finance.altman_zscore import altman_zscore
from src.utils.constants import OUTPUTS_DIR, TARGET


def run_case_study(df: Optional[pd.DataFrame] = None, save: bool = True) -> dict:
    df = load_indian_data() if df is None else df
    top = float(load_config()["top_pct"])
    proba = predict_proba(df)
    z = altman_zscore(df)["z_score"].fillna(0.0).values
    result = {
        "model": evaluate(df[TARGET], proba, top),
        "altman_zscore_only": evaluate(df[TARGET], -z, top),
        "note": "Synthetic stand-in for historical Indian companies; replace with real data.",
    }
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUTS_DIR / "indian_validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
