"""Evaluation metrics for phase prediction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if true >= 0 and pred >= 0:
            matrix[true, pred] += 1
    return matrix


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist()))
    scores = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"phase_change_precision": precision, "phase_change_recall": recall, "phase_change_f1": f1}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, change_true: np.ndarray, change_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "macro_f1": macro_f1(y_true, y_pred),
        **binary_prf(change_true, change_pred),
    }


def per_workload_metrics(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray, change_true: np.ndarray, change_pred: np.ndarray) -> list[dict[str, object]]:
    by_workload: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_workload[row.get("workload", "")].append(index)
    output: list[dict[str, object]] = []
    for workload, indices in sorted(by_workload.items()):
        idx = np.asarray(indices, dtype=int)
        item: dict[str, object] = {"workload": workload, "samples": len(indices)}
        item.update(classification_metrics(y_true[idx], y_pred[idx], change_true[idx], change_pred[idx]))
        output.append(item)
    return output
