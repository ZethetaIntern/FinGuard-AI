import joblib
import numpy as np
import pandas as pd

from src.data.clean_data import clean_dataframe
from src.features.feature_engineering import build_features
from src.utils.constants import MODEL_FILE, PREPROCESS_FILE


def artifacts_exist() -> bool:
    return MODEL_FILE.exists() and PREPROCESS_FILE.exists()


def load_artifacts():
    if not artifacts_exist():
        from src.modeling.train_model import train_and_save
        train_and_save()
    return joblib.load(PREPROCESS_FILE), joblib.load(MODEL_FILE)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    pre, bundle = load_artifacts()
    feats = bundle["features"]
    d = build_features(clean_dataframe(df, drop_missing_target=False, drop_dupes=False))
    X = d.reindex(columns=feats)
    return pd.DataFrame(pre["imputer"].transform(X), columns=feats)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    _, bundle = load_artifacts()
    X = prepare_features(df)
    return bundle["model"].predict_proba(X)[:, 1]
