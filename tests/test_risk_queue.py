import math

from src.data.download_data import generate_synthetic_financials
from src.modeling.risk_queue import generate_queue


def test_queue_size():
    df = generate_synthetic_financials(200, seed=5, prefix="TQ")
    profiles, queue = generate_queue(df, top_pct=0.05, save=False)
    assert len(queue) == math.ceil(200 * 0.05)
    assert queue["risk_probability"].min() >= profiles["risk_probability"].quantile(0.9)
