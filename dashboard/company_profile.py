import streamlit as st

from src.modeling.explain import explain_row
from src.modeling.predict import prepare_features
from src.validation.peer_comparison import peer_comparison


def render(ctx: dict) -> None:
    profiles = ctx["profiles"]
    st.header("Company")
    name = st.selectbox("Company", profiles["company_name"].tolist())
    row = profiles[profiles["company_name"] == name].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Risk", f"{row['risk_probability']:.3f}")
    b.metric("Rank", int(row["risk_rank"]))
    if "z_score" in profiles.columns:
        c.metric("Z-score", f"{row['z_score']:.2f}", row["z_zone"])
        d.metric("F-score", int(row["f_score"]), row["f_label"])
        st.metric("M-score", f"{row['m_score']:.2f}", row["m_flag"])
        st.write("Warnings:", row["warning_text"] if row["warning_text"] else "None")
    peers = peer_comparison(profiles, row["company_id"]) if "company_id" in profiles.columns else {}
    if peers:
        st.subheader("Sector peers")
        st.write(f"{peers['sector']} | peers {peers['peers']} | median {peers['sector_median']:.3f} | "
                 f"percentile {peers['sector_percentile']:.0%}")
    st.subheader("Drivers")
    one = profiles[profiles["company_name"] == name].head(1)
    contrib = explain_row(prepare_features(one)).head(8)
    st.bar_chart(contrib.set_index("feature")["contribution"])
