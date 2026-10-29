# Architecture
Data (raw -> interim -> processed) -> features (ratios, temporal) -> model (RandomForest + LR baseline)
-> risk queue (top 5%) -> Streamlit dashboard.
Altman Z, Beneish M and Piotroski F scores are computed separately in `src/finance` and shown beside, not mixed into, the ML probability.
