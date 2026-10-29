import pandas as pd


def sector_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in profiles.columns:
        return pd.DataFrame(columns=["sector", "companies", "median_risk", "queue_count"])
    g = profiles.groupby("sector")
    return pd.DataFrame({
        "companies": g.size(),
        "median_risk": g["risk_probability"].median().round(4),
        "queue_count": g["in_top_queue"].sum().astype(int),
    }).reset_index()


def peer_comparison(profiles: pd.DataFrame, company_id: str) -> dict:
    row = profiles[profiles["company_id"] == company_id]
    if row.empty:
        return {}
    row = row.iloc[0]
    peers = profiles[profiles["sector"] == row["sector"]]
    return {
        "sector": row["sector"],
        "peers": int(len(peers)),
        "company_risk": float(row["risk_probability"]),
        "sector_median": float(peers["risk_probability"].median()),
        "sector_percentile": float((peers["risk_probability"] < row["risk_probability"]).mean()),
    }
