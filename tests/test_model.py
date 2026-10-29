from src.data.download_data import generate_synthetic_financials
from src.modeling.evaluate import evaluate
from src.modeling.train_model import train_and_save


def test_train_no_save():
    m = train_and_save(generate_synthetic_financials(1500, seed=4), save=False)
    assert m["random_forest"]["roc_auc"] > 0.6


def test_evaluate_keys():
    m = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert m["precision_at_top"] == 1.0
