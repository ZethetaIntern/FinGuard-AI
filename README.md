# FinGuard-AI 🛡️
### AI-Powered Financial Risk Analysis and Early Warning Dashboard

FinGuard-AI is a Python-based financial risk analysis project designed to help users identify companies that may require closer financial review. It combines machine learning, financial indicators, and risk prioritization in an interactive dashboard.

## 🎯 Problem Statement

Financial distress can develop through declining profitability, weak liquidity, negative cash flow, and other financial warning signs. Reviewing these indicators manually across many companies can be time-consuming.

FinGuard-AI aims to support earlier risk identification by combining financial data analysis with machine-learning predictions.

## ✨ Key Features

- **Risk prediction:** Estimate financial distress risk using a trained machine-learning model.
- **Risk prioritization:** Rank companies so users can investigate higher-risk cases first.
- **Financial indicators:** Review financial warning signs and relevant metrics.
- **Interactive dashboard:** Explore risk information through a Streamlit interface.
- **Model evaluation:** Evaluate predictive performance using classification metrics.
- **Testing:** Includes a test suite to help validate project functionality.

*Feature availability depends on the current implementation and the data supplied to the application.*

## 🧰 Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Machine learning and financial data analysis
- Git and GitHub
- Pytest

## 🏗️ Project Structure

```text
FinGuard-AI/
├── app.py
├── config/
├── dashboard/
├── data/
├── docs/
├── models/
├── outputs/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/karthik23907/FinGuard-AI.git
cd FinGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

### 5. Run tests

```bash
python -m pytest
```

## 📊 Model Evaluation

Model performance should be reported using measured results from the project's evaluation process. Useful metrics include:

- ROC-AUC
- Precision and recall
- Precision-recall AUC
- Confusion matrix

Results depend on the dataset, target definition, evaluation split, and model configuration.

## ⚠️ Disclaimer

FinGuard-AI is an educational and analytical project. Predictions are estimates, not guarantees of financial distress. The application should not be used as the sole basis for investment, lending, or other consequential financial decisions.

## 👨‍💻 Author

**Karthik**

GitHub: https://github.com/karthik23907