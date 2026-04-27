"""Metrics used by family-wise teacher/student and ablation stages."""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist()))
    scores = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def high_usage_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for the 'high usage' bucket (class 2)."""

    mask = y_true == 2
    positives = int(np.sum(mask))
    if positives == 0:
        return 0.0
    return float(np.sum(y_pred[mask] == 2) / positives)


def hamming_distance_rows(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average tuple Hamming distance across rows."""

    if y_true.size == 0:
        return 0.0
    if y_true.ndim == 1:
        return float(np.mean(y_true != y_pred))
    return float(np.mean(np.mean(y_true != y_pred, axis=1)))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_state: np.ndarray | None = None) -> dict[str, float]:
    """Compute family state-classification metrics."""

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "macro_f1": macro_f1(y_true, y_pred),
        "high_usage_recall": high_usage_recall(y_true, y_pred),
    }
