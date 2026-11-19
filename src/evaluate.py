"""Métriques standard du scoring crédit."""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov : écart maximal entre les distributions de score
    des bons et des mauvais payeurs (= max(TPR - FPR))."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def credit_metrics(y_true, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),   # métrique usuelle en banque
        "ks": round(ks_statistic(y_true, y_score), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),  # qualité des probabilités
    }
