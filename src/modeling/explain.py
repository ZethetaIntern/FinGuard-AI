import numpy as np
import pandas as pd

from src.modeling.predict import load_artifacts

try:  # SHAP is optional at runtime: fall back to impurity importances
    import shap
except Exception:  # pragma: no cover
    shap = None


def _shap_matrix(model, X: pd.DataFrame):
    if shap is None:
        return None
    try:
        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):
            sv = sv[-1]
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        if arr.shape != X.shape:
            return None
        return arr
    except Exception:
        return None


def global_importance(X: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    _, bundle = load_artifacts()
    model = bundle["model"]
    Xs = X.sample(min(max_rows, len(X)), random_state=0) if len(X) else X
    arr = _shap_matrix(model, Xs)
    if arr is not None:
        imp = np.abs(arr).mean(axis=0)
        method = "SHAP"
    else:
        imp = model.feature_importances_
        method = "Importance"
    out = pd.DataFrame({"feature": X.columns, "importance": imp, "method": method})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_row(X_row: pd.DataFrame) -> pd.DataFrame:
    _, bundle = load_artifacts()
    model = bundle["model"]
    arr = _shap_matrix(model, X_row)
    if arr is not None:
        contrib = arr[0]
    else:
        contrib = model.feature_importances_ * 0.0
    out = pd.DataFrame({"feature": X_row.columns, "value": X_row.iloc[0].values, "contribution": contrib})
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
