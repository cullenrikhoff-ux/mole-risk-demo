from __future__ import annotations

import warnings
from typing import Tuple

from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError as exc:
        warnings.warn(f"AUC undefined for the current labels: {exc}")
        return float("nan")


def compute_confusion(y_true, y_pred) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def sensitivity(tn: int, fp: int, fn: int, tp: int) -> float:
    denom = tp + fn
    return (tp / denom) if denom else 0.0


def specificity(tn: int, fp: int, fn: int, tp: int) -> float:
    denom = tn + fp
    return (tn / denom) if denom else 0.0


def accuracy(tn: int, fp: int, fn: int, tp: int) -> float:
    denom = tn + fp + fn + tp
    return ((tn + tp) / denom) if denom else 0.0
