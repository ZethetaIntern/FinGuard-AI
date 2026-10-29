from typing import Iterable, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.data.clean_data import clean_dataframe
from src.data.load_data import load_config, load_training_data
from src.features.feature_engineering import build_features, get_feature_columns
from src.modeling.evaluate import evaluate
from src.utils.constants import TARGET


def run_backtest(df: Optional[pd.DataFrame] = None, cutoffs: Iterable[int] = (2015, 2017, 2019)) -> pd.DataFrame:
    """Train on years <= cutoff, test on later years."""
    cfg = load_config()
    df = load_training_data() if df is None else df
    df = build_features(clean_dataframe(df))
    feats = get_feature_columns(df)
    rows = []
    if "year" not in df.columns:
        return pd.DataFrame(columns=["cutoff", "train_rows", "test_rows", "roc_auc", "pr_auc"])
    for c in cutoffs:
        tr, te = df[df["year"] <= c], df[df["year"] > c]
        if len(tr) == 0 or len(te) == 0 or tr[TARGET].nunique() < 2 or te[TARGET].nunique() < 2:
            continue
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tr[feats])
        rf = RandomForestClassifier(n_estimators=150, max_depth=int(cfg["max_depth"]),
                                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=cfg["seed"]).fit(imp.transform(tr[feats]), tr[TARGET])
        m = evaluate(te[TARGET], rf.predict_proba(imp.transform(te[feats]))[:, 1], float(cfg["top_pct"]))
        rows.append({"cutoff": c, "train_rows": len(tr), "test_rows": len(te),
                     "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]})
    return pd.DataFrame(rows, columns=["cutoff", "train_rows", "test_rows", "roc_auc", "pr_auc"])
