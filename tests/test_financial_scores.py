from src.data.download_data import generate_synthetic_financials
from src.finance.altman_zscore import altman_zscore
from src.finance.beneish_mscore import beneish_mscore
from src.finance.piotroski_fscore import piotroski_fscore
from src.finance.warning_signals import compute_warning_signals


def test_scores():
    df = generate_synthetic_financials(200, seed=3)
    assert len(altman_zscore(df)) == 200
    assert beneish_mscore(df)["m_score"].notna().mean() > 0.9
    f = piotroski_fscore(df)["f_score"]
    assert f.between(0, 9).all()
    w = compute_warning_signals(df)
    assert {"z_zone", "m_flag", "f_label", "warning_count"} <= set(w.columns)
