import streamlit as st

from src.modeling.explain import global_importance
from src.modeling.predict import prepare_features


def render(ctx: dict) -> None:
    st.header("Explainability")
    imp = global_importance(prepare_features(ctx["universe"]))
    st.caption(imp["method"].iloc[0] if len(imp) else "")
    st.bar_chart(imp.set_index("feature")["importance"])
    st.dataframe(imp[["feature", "importance"]])
