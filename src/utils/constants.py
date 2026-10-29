from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TRAIN_FILE = DATA_PROCESSED / "training_dataset.csv"
INDIAN_FILE = DATA_PROCESSED / "indian_validation_dataset.csv"
UNIVERSE_FILE = DATA_PROCESSED / "company_screening_universe.csv"
MODEL_FILE = MODELS_DIR / "distress_model.joblib"
PREPROCESS_FILE = MODELS_DIR / "preprocessing_pipeline.joblib"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"
RISK_QUEUE_FILE = OUTPUTS_DIR / "risk_queue.csv"
PROFILES_FILE = OUTPUTS_DIR / "company_risk_profiles.csv"

TARGET = "bankrupt"
ID_COLS = ["company_id", "company_name", "sector", "year"]

STATEMENT_COLS = [
    "total_assets", "current_assets", "current_liabilities", "total_liabilities",
    "long_term_debt", "retained_earnings", "ebit", "net_income", "sales", "cogs",
    "receivables", "ppe", "depreciation", "sga", "cfo", "market_value_equity",
    "shares_outstanding",
]
PREV_COLS = [c + "_prev" for c in STATEMENT_COLS]

RATIO_COLS = [
    "wc_ta", "re_ta", "ebit_ta", "equity_tl", "sales_ta",
    "current_ratio", "debt_ratio", "roa", "cfo_ta",
]
TEMPORAL_COLS = ["d_roa", "d_debt_ratio", "d_current_ratio", "sales_growth", "asset_growth"]
