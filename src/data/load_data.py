from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.data.download_data import ensure_datasets
from src.utils.constants import CONFIG_PATH, DATA_RAW, INDIAN_FILE, TARGET, TRAIN_FILE, UNIVERSE_FILE

DEFAULT_CONFIG = {
    "seed": 42, "test_size": 0.25, "top_pct": 0.05,
    "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    except FileNotFoundError:
        pass
    return cfg


def _read(path: Path) -> pd.DataFrame:
    ensure_datasets()
    return pd.read_csv(path)


def load_training_data() -> pd.DataFrame:
    return _read(TRAIN_FILE)


def load_indian_data() -> pd.DataFrame:
    return _read(INDIAN_FILE)


def load_universe() -> pd.DataFrame:
    return _read(UNIVERSE_FILE)


def load_real_bankruptcy(folder: str = "taiwanese_bankruptcy") -> Optional[pd.DataFrame]:
    """Load the first CSV found in data/raw/<folder>; returns None if absent.

    Standardises the label column to `bankrupt`. Any dataset with a binary label can be
    passed to train_model.train_and_save(df).
    """
    files = sorted((DATA_RAW / folder).glob("*.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0])
    df.columns = [str(c).strip() for c in df.columns]
    for cand in ("Bankrupt?", "bankrupt", "class", "Class", "target"):
        if cand in df.columns:
            df = df.rename(columns={cand: TARGET})
            break
    return df
