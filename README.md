# FinGuard-AI
Early-warning system for corporate financial distress: ML risk ranking, Altman/Beneish/Piotroski scores, SHAP explanations, and a Streamlit dashboard.

## Problem and solution
Analysts cannot manually review thousands of companies. FinGuard AI ranks a screening universe and surfaces the top 5% with explainable drivers.

## Architecture
See `docs/architecture.md`.

## Stack
Python, pandas, scikit-learn, SHAP, Streamlit, pytest.

## Run
```bash
pip install -r requirements.txt
python -m src.data.download_data      # creates datasets (synthetic fallback)
python -m src.modeling.train_model    # trains + writes outputs/metrics.json
python -m src.modeling.risk_queue     # writes outputs/risk_queue.csv
streamlit run app.py                  # dashboard
pytest -q                             # tests
```
`app.py` also runs the steps above automatically on first launch.

## Data
See `docs/data_sources.md`. Do not commit raw datasets or secrets.

## Limitations
See `docs/limitations.md`.

## Team and AI-tool disclosure
Add team contributions and AI tools used here. Transfer the repo to `ZethetaIntern` per the rulebook.
