import json
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.clean_data import clean_dataframe
from src.data.load_data import load_config, load_training_data
from src.data.validate_data import validate_dataframe
from src.features.feature_engineering import build_features, get_feature_columns
from src.modeling.evaluate import evaluate
from src.utils.constants import (
    METRICS_FILE, MODEL_FILE, MODELS_DIR, OUTPUTS_DIR, PREPROCESS_FILE, TARGET,
)
from src.utils.logging_config import get_logger

log = get_logger(__name__)


def train_and_save(df: Optional[pd.DataFrame] = None, save: bool = True) -> dict:
    cfg = load_config()
    if df is None:
        df = load_training_data()
    df = clean_dataframe(df)
    validate_dataframe(df)
    df = build_features(df)
    feats = get_feature_columns(df)
    X, y = df[feats], df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"])

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(Xtr)
    Xtr_p = pd.DataFrame(imputer.transform(Xtr), columns=feats)
    Xte_p = pd.DataFrame(imputer.transform(Xte), columns=feats)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=cfg["seed"]))
    baseline.fit(Xtr_p, ytr)

    rf = RandomForestClassifier(
        n_estimators=int(cfg["n_estimators"]), max_depth=int(cfg["max_depth"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]), class_weight="balanced_subsample",
        n_jobs=-1, random_state=cfg["seed"])
    rf.fit(Xtr_p, ytr)

    top = float(cfg["top_pct"])
    metrics = {
        "baseline_logistic_regression": evaluate(yte, baseline.predict_proba(Xte_p)[:, 1], top),
        "random_forest": evaluate(yte, rf.predict_proba(Xte_p)[:, 1], top),
        "train_rows": int(len(Xtr)),
        "test_rows": int(len(Xte)),
        "positive_rate": round(float(y.mean()), 4),
        "features": feats,
    }
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": imputer, "features": feats}, PREPROCESS_FILE)
        joblib.dump({"model": rf, "baseline": baseline, "features": feats}, MODEL_FILE)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info("Saved model and metrics.")
    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print(json.dumps({k: v for k, v in m.items() if k != "features"}, indent=2))
