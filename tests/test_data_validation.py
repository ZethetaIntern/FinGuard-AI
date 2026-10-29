import pandas as pd
import pytest

from src.data.clean_data import clean_dataframe
from src.data.download_data import generate_synthetic_financials
from src.data.validate_data import validate_dataframe


def test_synthetic_valid():
    df = clean_dataframe(generate_synthetic_financials(300, seed=1))
    rep = validate_dataframe(df)
    assert rep["rows"] == 300 and 0 < rep["positive_rate"] < 1


def test_empty_raises():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())


def test_missing_target_raises():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame({"a": [1, 2]}))
