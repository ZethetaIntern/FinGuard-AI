import pandas as pd
import streamlit as st


def render(ctx: dict) -> None:
    st.header("Data")
    st.dataframe(pd.DataFrame([
        {"Source": "Polish bankruptcy (UCI 365)", "Use": "Training", "Status": "Add to data/raw"},
        {"Source": "Taiwanese bankruptcy (UCI 572)", "Use": "Training", "Status": "Add to data/raw"},
        {"Source": "Indian companies", "Use": "Validation", "Status": "Add to data/raw"},
        {"Source": "Synthetic fallback", "Use": "Demo / tests", "Status": "Active"},
    ]))
    st.caption("Synthetic data is for pipeline demonstration only.")
