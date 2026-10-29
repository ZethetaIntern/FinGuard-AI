import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _clean(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def precision_recall_at_top(y_true, scores, top_pct: float = 0.05):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    k = max(1, int(math.ceil(len(s) * top_pct)))
    idx = np.argsort(-s)[:k]
    tp = int(y[idx].sum())
    pos = int(y.sum())
    precision = tp / k
    recall = tp / pos if pos else float("nan")
    return precision, recall, k


def evaluate(y_true, scores, top_pct: float = 0.05) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    both = len(np.unique(y)) == 2
    p, r, k = precision_recall_at_top(y, s, top_pct)
    return {
        "roc_auc": _clean(roc_auc_score(y, s)) if both else None,
        "pr_auc": _clean(average_precision_score(y, s)) if both else None,
        "top_pct": top_pct,
        "top_k": k,
        "precision_at_top": _clean(p),
        "recall_at_top": _clean(r),
        "positives": int(y.sum()),
        "rows": int(len(y)),
    }
