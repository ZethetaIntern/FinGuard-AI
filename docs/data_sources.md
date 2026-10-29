# Data sources
- Polish bankruptcy: https://archive.ics.uci.edu/dataset/365 -> data/raw/polish_bankruptcy/
- Taiwanese bankruptcy: https://archive.ics.uci.edu/dataset/572 -> data/raw/taiwanese_bankruptcy/
- Indian companies: public annual-report exports -> data/raw/indian_companies/
- Synthetic fallback (default): `python -m src.data.download_data`. Demo only.
Real UCI files use ratio columns, not full statements: load with `load_real_bankruptcy()` and call `train_and_save(df)`.
