"""Dataset acquisition.

Real datasets (place the CSV files in data/raw/<folder>/):
  - Polish bankruptcy:   https://archive.ics.uci.edu/dataset/365
  - Taiwanese bankruptcy: https://archive.ics.uci.edu/dataset/572
  - Indian companies:    export from a public source (e.g. annual reports, screener exports)

So that the whole project runs end-to-end without manual downloads, this module also
generates a clearly-labelled SYNTHETIC statement-level dataset with the same schema.
"""
import numpy as np
import pandas as pd

from src.utils.constants import (
    DATA_PROCESSED, DATA_RAW, DATA_INTERIM, INDIAN_FILE, MODELS_DIR, OUTPUTS_DIR,
    FIGURES_DIR, STATEMENT_COLS, TARGET, TRAIN_FILE, UNIVERSE_FILE,
)
from src.utils.logging_config import get_logger

log = get_logger(__name__)

SECTORS = ["IT", "Pharma", "Auto", "Energy", "Retail", "Textiles", "Infrastructure", "FMCG"]


def generate_synthetic_financials(n: int, seed: int = 42, distress_shift: float = 0.0,
                                  prefix: str = "CO", year_range=(2012, 2022)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.normal(distress_shift, 1.0, n)  # latent distress

    def noise(s):
        return rng.normal(0, s, n)

    ta_prev = rng.lognormal(8, 0.8, n)
    ta = ta_prev * np.clip(1.05 - 0.04 * d + noise(0.06), 0.5, 2.0)
    ca = ta * np.clip(0.40 - 0.05 * d + noise(0.08), 0.05, 0.85)
    cl = ta * np.clip(0.25 + 0.05 * d + noise(0.06), 0.03, 0.95)
    tl = ta * np.clip(0.45 + 0.08 * d + noise(0.08), 0.10, 1.60)
    ltd = np.maximum(tl - cl, 0.05 * ta)
    re = ta * (0.20 - 0.10 * d + noise(0.05))
    sales = ta * np.clip(1.0 - 0.05 * d + noise(0.20), 0.2, 3.0)
    cogs = sales * np.clip(0.70 + 0.03 * d + noise(0.05), 0.30, 1.10)
    ebit = sales * (0.08 - 0.05 * d + noise(0.02))
    ni = ebit * 0.65 - 0.01 * ta * np.maximum(d, 0)
    ppe = ta * np.clip(0.35 + noise(0.08), 0.05, 0.80)
    dep = ppe * np.clip(0.08 + noise(0.01), 0.02, 0.2)
    sga = sales * np.clip(0.10 + noise(0.02), 0.02, 0.3)
    rec = sales * np.clip(0.12 + 0.02 * d + noise(0.03), 0.01, 0.6)
    cfo = ni + dep + ta * noise(0.02) - 0.01 * ta * np.maximum(d, 0)
    mve = np.maximum(ta - tl, 0.05 * ta) * np.clip(1.2 - 0.2 * d + noise(0.2), 0.1, 3.0)
    shares = 100 * np.clip(1 + 0.02 * np.maximum(d, 0) + noise(0.01), 0.8, 1.5)

    cur = {
        "total_assets": ta, "current_assets": ca, "current_liabilities": cl,
        "total_liabilities": tl, "long_term_debt": ltd, "retained_earnings": re,
        "ebit": ebit, "net_income": ni, "sales": sales, "cogs": cogs,
        "receivables": rec, "ppe": ppe, "depreciation": dep, "sga": sga,
        "cfo": cfo, "market_value_equity": mve, "shares_outstanding": shares,
    }
    out = {
        "company_id": [f"{prefix}{i:05d}" for i in range(n)],
        "company_name": [f"{prefix} Company {i:04d}" for i in range(n)],
        "sector": rng.choice(SECTORS, n),
        "year": rng.integers(year_range[0], year_range[1] + 1, n),
    }
    for c in STATEMENT_COLS:
        out[c] = cur[c]
    for c in STATEMENT_COLS:
        if c == "total_assets":
            out[c + "_prev"] = ta_prev
        else:
            g = np.clip(0.05 - 0.03 * d + noise(0.05), -0.4, 0.8)
            out[c + "_prev"] = cur[c] / (1 + g)
    prob = 1 / (1 + np.exp(-(1.2 * d - 3.2)))
    out[TARGET] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(out)


def print_real_data_instructions() -> None:
    log.info("Real data: put CSVs in data/raw/polish_bankruptcy, data/raw/taiwanese_bankruptcy, "
             "data/raw/indian_companies (see docs/data_sources.md).")


def ensure_datasets(force: bool = False) -> None:
    for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    if force or not TRAIN_FILE.exists():
        generate_synthetic_financials(4000, seed=42).to_csv(TRAIN_FILE, index=False)
        log.info("Wrote %s", TRAIN_FILE)
    if force or not INDIAN_FILE.exists():
        generate_synthetic_financials(800, seed=7, distress_shift=0.25, prefix="IN",
                                      year_range=(2008, 2022)).to_csv(INDIAN_FILE, index=False)
        log.info("Wrote %s", INDIAN_FILE)
    if force or not UNIVERSE_FILE.exists():
        generate_synthetic_financials(1000, seed=11, distress_shift=0.1, prefix="SC",
                                      year_range=(2022, 2022)).to_csv(UNIVERSE_FILE, index=False)
        log.info("Wrote %s", UNIVERSE_FILE)


if __name__ == "__main__":
    print_real_data_instructions()
    ensure_datasets(force=True)
