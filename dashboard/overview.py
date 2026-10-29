import pandas as pd
import streamlit as st

COLS = ["company_id", "company_name", "sector", "year", "risk_probability", "risk_rank",
        "z_zone", "m_flag", "f_label", "warning_count"]


def render(ctx: dict) -> None:
    profiles, queue, metrics = ctx["profiles"], ctx["queue"], ctx["metrics"]
    st.markdown('<div class="section-title"><h2>Portfolio risk snapshot</h2></div>'
                '<div class="section-subtitle">A ranked view of potential financial distress signals across the screening universe.</div>',
                unsafe_allow_html=True)

    risk = pd.to_numeric(profiles.get("risk_probability", pd.Series(dtype=float)), errors="coerce")
    high_threshold = 0.70
    high_risk = int((risk >= high_threshold).sum()) if len(risk) else 0
    roc = None
    for model_name in ("baseline_logistic_regression", "random_forest"):
        model_metrics = metrics.get(model_name, {})
        if "roc_auc" in model_metrics:
            roc = model_metrics["roc_auc"]
            break

    a, b, c, d = st.columns(4)
    a.metric("Companies screened", f"{len(profiles):,}", help="Number of companies in the current screening universe.")
    b.metric("Priority queue", f"{len(queue):,}", help="Companies included in the configured risk review queue.")
    c.metric("High predicted risk", f"{high_risk:,}", help=f"Risk probability ≥ {high_threshold:.0%}; this is a screening threshold, not a confirmed outcome.")
    d.metric("Model ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A",
             help="Reported evaluation metric. Check the Validation page for the evaluation context.")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-title"><h2>Priority review queue</h2></div>'
                    '<div class="section-subtitle">Companies ranked by model-predicted risk.</div>',
                    unsafe_allow_html=True)
        if not queue.empty:
            display_cols = [c for c in COLS if c in queue.columns]
            show = queue[display_cols].copy()
            if "risk_probability" in show:
                show["risk_probability"] = show["risk_probability"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            st.dataframe(show, use_container_width=True, hide_index=True, height=390)
        else:
            st.info("The priority queue is empty for the current dataset.")
    with right:
        st.markdown('<div class="section-title"><h2>Risk by sector</h2></div>'
                    '<div class="section-subtitle">Number of queued companies in each sector.</div>',
                    unsafe_allow_html=True)
        if "sector" in queue.columns and not queue.empty:
            sector_counts = queue["sector"].fillna("Unknown").value_counts().rename_axis("Sector")
            st.bar_chart(sector_counts, use_container_width=True)
        else:
            st.info("Sector information is not available in the queue.")

    st.markdown('<div class="section-title"><h2>Risk distribution</h2></div>'
                '<div class="section-subtitle">Distribution of model-predicted risk across the full screening universe.</div>',
                unsafe_allow_html=True)
    if len(risk.dropna()):
        hist = pd.cut(risk.dropna(), bins=[-0.001, .2, .4, .6, .8, 1.001],
                      labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]).value_counts(sort=False)
        st.bar_chart(hist, use_container_width=True)
    st.caption("Screening output is decision support, not proof of fraud, bankruptcy, or future failure. Verify data provenance and model calibration before operational use.")
