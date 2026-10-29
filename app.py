import json

import streamlit as st

from dashboard import company_profile, data_sources, explainability, historical_validation, overview
from src.data.download_data import ensure_datasets
from src.data.load_data import load_training_data, load_universe
from src.modeling.predict import artifacts_exist
from src.modeling.risk_queue import generate_queue
from src.modeling.train_model import train_and_save
from src.utils.constants import METRICS_FILE

st.set_page_config(
    page_title="FinGuard AI | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual system inspired by the supplied dark financial intelligence reference.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#07111f; --panel:#0d1b2b; --panel2:#102337; --line:#20374b;
            --teal:#20d6bd; --cyan:#48c8ee; --text:#e8f1f8; --muted:#91a7b9; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(ellipse at 75% 0%, #14283b 0%, #07111f 52%); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(7,17,31,.85); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#091522 0%,#0b1827 100%); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--muted); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#f3f8fc !important; letter-spacing:-.025em; }
    h1 { font-size:2.05rem !important; }
    h2 { font-size:1.45rem !important; }
    h3 { font-size:1.08rem !important; }
    p, label, .stCaption { color:#a8bbca; }
    .brand { display:flex; gap:10px; align-items:center; margin:0 0 5px 0; }
    .brand-icon { width:36px;height:36px;border-radius:11px;display:flex;align-items:center;justify-content:center;
                  background:linear-gradient(135deg,#20d6bd,#168ba8);color:#06131e;font-size:22px;font-weight:800; }
    .brand-name { font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:#f2f8fc; }
    .brand-sub { color:#7894a8;font-size:10px;letter-spacing:.15em;text-transform:uppercase;margin-top:1px; }
    .eyebrow { color:var(--teal);font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px; }
    .page-intro { color:#9bb0c1;font-size:14px;margin-top:-7px;margin-bottom:20px; }
    .topline { display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:22px; }
    .live-pill { border:1px solid #1b655f;background:#0b302f;color:#5de6d1;border-radius:20px;padding:6px 10px;font-size:11px;font-weight:700;white-space:nowrap; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#102337,#0c1a29);border:1px solid #20384c;border-radius:15px;padding:17px 18px;min-height:116px; }
    div[data-testid="stMetricLabel"] { color:#92a9bb !important;font-size:12px; }
    div[data-testid="stMetricValue"] { color:#f4f9fc !important;font-family:'Space Grotesk',sans-serif; }
    div[data-testid="stMetricDelta"] { font-size:11px; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] { border:1px solid #20384c;border-radius:12px;overflow:hidden; }
    .section-title { margin-top:22px;margin-bottom:4px; }
    .section-subtitle { color:#8ca5b8;font-size:12px;margin-bottom:12px; }
    .stButton button { border-radius:9px;border:1px solid #28536a;background:#103449;color:#dffbf7; }
    .stButton button:hover { border-color:var(--teal);color:white; }
    .stSelectbox [data-baseweb="select"] > div, .stTextInput input { background:#0d1b2b;border-color:#294156;color:#edf5fa; }
    hr { border-color:#20374b; }
    footer {visibility:hidden;}
    #MainMenu {visibility:hidden;}
    @media (max-width:800px) { .brand-name{font-size:18px;} }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Preparing data and model...")
def load_context():
    ensure_datasets()
    if not artifacts_exist() or not METRICS_FILE.exists():
        train_and_save(load_training_data())
    universe = load_universe()
    profiles, queue = generate_queue(universe)
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"universe": universe, "profiles": profiles, "queue": queue, "metrics": metrics}

PAGES = {
    "Overview": overview,
    "Company Risk": company_profile,
    "AI Explainability": explainability,
    "Model Validation": historical_validation,
    "Data Sources": data_sources,
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">◈</div><div><div class="brand-name">FinGuard AI</div><div class="brand-sub">Risk intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
        format_func=lambda x: {
            "Overview": "◉   Overview",
            "Company Risk": "▦   Company Risk",
            "AI Explainability": "✳   AI Explainability",
            "Model Validation": "⌁   Model Validation",
            "Data Sources": "▤   Data Sources",
        }[x],
    )
    st.markdown("---")
    st.markdown(
        '<div style="border:1px solid #20384c;background:#0d1b2b;border-radius:12px;padding:13px">'
        '<div style="color:#20d6bd;font-size:11px;font-weight:700;letter-spacing:.12em">SYSTEM STATUS</div>'
        '<div style="margin-top:8px;color:#e7f3f8;font-size:13px">● Risk engine ready</div>'
        '<div style="color:#7f9aaf;font-size:11px;margin-top:4px">Decision support · Not a final diagnosis</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="topline"><div><div class="eyebrow">FINANCIAL INTELLIGENCE PLATFORM</div>'
    f'<h1 style="margin:0">{" ".join(page.split())}</h1>'
    f'<div class="page-intro">Early-warning signals, explainable risk scores, and company-level insights.</div></div>'
    f'<div class="live-pill">● LOCAL WORKSPACE</div></div>',
    unsafe_allow_html=True,
)
PAGES[page].render(load_context())
