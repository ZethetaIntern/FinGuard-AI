import pandas as pd
import streamlit as st

from src.validation.indian_case_study import run_case_study
from src.validation.temporal_backtest import run_backtest


@st.cache_data(show_spinner=False)
def _case():
    return run_case_study()


@st.cache_data(show_spinner=False)
def _backtest():
    return run_backtest()


def render(ctx: dict) -> None:
    st.header("Validation")
    res = _case()
    table = pd.DataFrame({"Model": res["model"], "Z-score only": res["altman_zscore_only"]}).drop(
        ["top_pct"], errors="ignore")
    st.subheader("Indian case study")
    st.dataframe(table.astype(str))
    st.subheader("Temporal backtest")
    bt = _backtest()
    st.dataframe(bt)
    if not bt.empty:
        st.line_chart(bt.set_index("cutoff")[["roc_auc", "pr_auc"]])
    st.subheader("Model metrics")
    m = ctx["metrics"]
    st.dataframe(pd.DataFrame({k: m[k] for k in ("baseline_logistic_regression", "random_forest")}).astype(str))
