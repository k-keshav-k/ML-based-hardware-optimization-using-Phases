"""Metrics used by family-wise teacher/student and ablation stages."""

from __future__ import annotations

import numpy as np


def _labels(y_true: np.ndarray, y_pred: np.ndarray) -> list[int]:
    return sorted(set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist()))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = _labels(y_true, y_pred)
    scores = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = _labels(y_true, y_pred)
    numer = 0.0
    denom = 0
    for label in labels:
        support = int(np.sum(y_true == label))
        if support <= 0:
            continue
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        numer += support * f1
        denom += support
    return numer / denom if denom else 0.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = _labels(y_true, y_pred)
    recalls = []
    for label in labels:
        positives = int(np.sum(y_true == label))
        if positives <= 0:
            continue
        recalls.append(float(np.sum((y_true == label) & (y_pred == label)) / positives))
    return float(np.mean(recalls)) if recalls else 0.0


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
        "weighted_f1": weighted_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "high_usage_recall": high_usage_recall(y_true, y_pred),
    }


def transition_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_state: np.ndarray) -> dict[str, float | int]:
    """Score whether a transition occurs, independently of its destination."""

    truth_transition = np.asarray(y_true, dtype=int) != np.asarray(current_state, dtype=int)
    predicted_transition = np.asarray(y_pred, dtype=int) != np.asarray(current_state, dtype=int)
    true_positive = int(np.sum(truth_transition & predicted_transition))
    false_positive = int(np.sum(~truth_transition & predicted_transition))
    false_negative = int(np.sum(truth_transition & ~predicted_transition))
    true_negative = int(np.sum(~truth_transition & ~predicted_transition))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    return {
        "transition_event_precision": precision,
        "transition_event_recall": recall,
        "transition_event_f1": f1,
        "transition_false_alarm_rate": false_alarm_rate,
        "transition_true_positives": true_positive,
        "transition_false_positives": false_positive,
        "transition_false_negatives": false_negative,
        "transition_true_negatives": true_negative,
    }


def grouped_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_state: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 17,
) -> dict[str, float | int]:
    """Return cluster-bootstrap 95% intervals without resampling correlated rows."""

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    current_state = current_state.astype(int)
    group_ids = np.asarray(group_ids, dtype=object)
    if y_true.size == 0:
        return {
            "eval_group_count": 0,
            "bootstrap_resamples": 0,
            "accuracy_ci95_low": 0.0,
            "accuracy_ci95_high": 0.0,
            "macro_f1_ci95_low": 0.0,
            "macro_f1_ci95_high": 0.0,
            "balanced_accuracy_ci95_low": 0.0,
            "balanced_accuracy_ci95_high": 0.0,
            "transition_accuracy_ci95_low": 0.0,
            "transition_accuracy_ci95_high": 0.0,
        }

    normalized_groups = np.asarray(
        [str(value).strip() or f"row_{index}" for index, value in enumerate(group_ids.tolist())],
        dtype=object,
    )
    unique_groups = sorted(set(normalized_groups.tolist()))
    group_index = {group: index for index, group in enumerate(unique_groups)}
    confusion = np.zeros((len(unique_groups), 3, 3), dtype=np.int64)
    transition_correct = np.zeros(len(unique_groups), dtype=np.int64)
    transition_total = np.zeros(len(unique_groups), dtype=np.int64)
    for truth, prediction, current, group in zip(y_true, y_pred, current_state, normalized_groups):
        if truth < 0 or truth > 2 or prediction < 0 or prediction > 2:
            continue
        position = group_index[str(group)]
        confusion[position, truth, prediction] += 1
        if truth != current:
            transition_total[position] += 1
            transition_correct[position] += int(truth == prediction)

    samples = max(1, int(n_resamples))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_groups), size=(samples, len(unique_groups)))
    sampled_confusion = np.sum(confusion[draws], axis=1)
    total = np.sum(sampled_confusion, axis=(1, 2))
    diagonal = np.diagonal(sampled_confusion, axis1=1, axis2=2)
    accuracy = np.divide(np.sum(diagonal, axis=1), total, out=np.zeros(samples, dtype=float), where=total > 0)

    support = np.sum(sampled_confusion, axis=2)
    predicted = np.sum(sampled_confusion, axis=1)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal, dtype=float), where=support > 0)
    balanced = np.divide(
        np.sum(recall, axis=1),
        np.sum(support > 0, axis=1),
        out=np.zeros(samples, dtype=float),
        where=np.sum(support > 0, axis=1) > 0,
    )
    f1_denominator = 2 * diagonal + (predicted - diagonal) + (support - diagonal)
    f1 = np.divide(2 * diagonal, f1_denominator, out=np.zeros_like(diagonal, dtype=float), where=f1_denominator > 0)
    macro_f1_values = np.divide(
        np.sum(f1, axis=1),
        np.sum(f1_denominator > 0, axis=1),
        out=np.zeros(samples, dtype=float),
        where=np.sum(f1_denominator > 0, axis=1) > 0,
    )

    sampled_transition_correct = np.sum(transition_correct[draws], axis=1)
    sampled_transition_total = np.sum(transition_total[draws], axis=1)
    transition_accuracy = np.divide(
        sampled_transition_correct,
        sampled_transition_total,
        out=np.zeros(samples, dtype=float),
        where=sampled_transition_total > 0,
    )

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    accuracy_low, accuracy_high = interval(accuracy)
    macro_low, macro_high = interval(macro_f1_values)
    balanced_low, balanced_high = interval(balanced)
    valid_transition_draws = sampled_transition_total > 0
    transition_low, transition_high = (
        interval(transition_accuracy[valid_transition_draws])
        if np.any(valid_transition_draws)
        else (0.0, 0.0)
    )
    return {
        "eval_group_count": len(unique_groups),
        "bootstrap_resamples": samples,
        "accuracy_ci95_low": accuracy_low,
        "accuracy_ci95_high": accuracy_high,
        "macro_f1_ci95_low": macro_low,
        "macro_f1_ci95_high": macro_high,
        "balanced_accuracy_ci95_low": balanced_low,
        "balanced_accuracy_ci95_high": balanced_high,
        "transition_accuracy_ci95_low": transition_low,
        "transition_accuracy_ci95_high": transition_high,
    }


def grouped_bootstrap_paired_difference(
    y_true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    current_state: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_resamples: int = 5000,
    seed: int = 29,
) -> dict[str, float | int]:
    """Bootstrap paired accuracy differences, sampling complete execution groups."""

    y_true = np.asarray(y_true, dtype=int)
    first_pred = np.asarray(first_pred, dtype=int)
    second_pred = np.asarray(second_pred, dtype=int)
    current_state = np.asarray(current_state, dtype=int)
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    if not unique_groups:
        return {"eval_group_count": 0, "bootstrap_resamples": 0}

    group_index = {group: index for index, group in enumerate(unique_groups)}
    totals = np.zeros(len(unique_groups), dtype=np.int64)
    first_correct = np.zeros(len(unique_groups), dtype=np.int64)
    second_correct = np.zeros(len(unique_groups), dtype=np.int64)
    transition_totals = np.zeros(len(unique_groups), dtype=np.int64)
    first_transition_correct = np.zeros(len(unique_groups), dtype=np.int64)
    second_transition_correct = np.zeros(len(unique_groups), dtype=np.int64)
    for truth, first, second, current, group in zip(y_true, first_pred, second_pred, current_state, groups):
        if truth < 0 or first < 0 or second < 0:
            continue
        position = group_index[str(group)]
        totals[position] += 1
        first_correct[position] += int(first == truth)
        second_correct[position] += int(second == truth)
        if truth != current:
            transition_totals[position] += 1
            first_transition_correct[position] += int(first == truth)
            second_transition_correct[position] += int(second == truth)

    samples = max(1, int(n_resamples))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_groups), size=(samples, len(unique_groups)))
    sampled_totals = np.sum(totals[draws], axis=1)
    first_accuracy = np.divide(
        np.sum(first_correct[draws], axis=1),
        sampled_totals,
        out=np.zeros(samples, dtype=float),
        where=sampled_totals > 0,
    )
    second_accuracy = np.divide(
        np.sum(second_correct[draws], axis=1),
        sampled_totals,
        out=np.zeros(samples, dtype=float),
        where=sampled_totals > 0,
    )
    sampled_transition_totals = np.sum(transition_totals[draws], axis=1)
    first_transition = np.divide(
        np.sum(first_transition_correct[draws], axis=1),
        sampled_transition_totals,
        out=np.zeros(samples, dtype=float),
        where=sampled_transition_totals > 0,
    )
    second_transition = np.divide(
        np.sum(second_transition_correct[draws], axis=1),
        sampled_transition_totals,
        out=np.zeros(samples, dtype=float),
        where=sampled_transition_totals > 0,
    )

    accuracy_difference = first_accuracy - second_accuracy
    valid_transition_draws = sampled_transition_totals > 0
    transition_difference = (first_transition - second_transition)[valid_transition_draws]
    accuracy_low, accuracy_high = np.quantile(accuracy_difference, [0.025, 0.975])
    transition_low, transition_high = (
        np.quantile(transition_difference, [0.025, 0.975])
        if transition_difference.size
        else (0.0, 0.0)
    )
    return {
        "eval_group_count": len(unique_groups),
        "bootstrap_resamples": samples,
        "accuracy_difference": float(np.mean(accuracy_difference)),
        "accuracy_difference_ci95_low": float(accuracy_low),
        "accuracy_difference_ci95_high": float(accuracy_high),
        "accuracy_difference_probability_gt_zero": float(np.mean(accuracy_difference > 0.0)),
        "transition_accuracy_difference": float(np.mean(transition_difference)) if transition_difference.size else 0.0,
        "transition_accuracy_difference_ci95_low": float(transition_low),
        "transition_accuracy_difference_ci95_high": float(transition_high),
        "transition_accuracy_difference_probability_gt_zero": (
            float(np.mean(transition_difference > 0.0)) if transition_difference.size else 0.0
        ),
    }
