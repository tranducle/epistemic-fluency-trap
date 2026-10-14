#!/usr/bin/env python3
"""
Statistical and Classification Evaluation Utilities for Multi-Corpus SOC Telemetry.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    """Compute binary classification metrics from continuous scores and threshold."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    y_pred = (y_s > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    try:
        auroc = float(roc_auc_score(y_t, y_s))
    except ValueError:
        auroc = 0.5
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y_t, y_pred, zero_division=0)),
        "recall": float(recall_score(y_t, y_pred, zero_division=0)),
        "f1": float(f1_score(y_t, y_pred, zero_division=0)),
        "fpr": fpr,
        "accuracy": float(accuracy_score(y_t, y_pred)),
        "auroc": auroc,
    }
