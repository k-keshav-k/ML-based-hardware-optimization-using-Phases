"""Train online next-phase predictors against offline clustered phase labels."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import write_csv_rows, write_json

from .config import load_config
from .data import load_scope_family_data, states_matrix
from .hardware_budget import estimate_hardware_budget
from .metrics import classification_metrics, grouped_bootstrap_confidence_intervals, transition_event_metrics
from .orchestration import experiment_dirs, scopes_for_experiment
from .phase_detector_models import (
    run_decision_tree,
    run_hsmm_approx,
    run_last_state_baseline,
    run_majority_baseline,
    run_markov,
    run_rocket_fallback,
    run_rle_markov,
    run_state_conditioned_majority,
    run_tcn,
    run_transformer,
)

TREE_MODEL = "online_phase_history_tree"
ALL_FAMILY_TREE_MODEL = "online_phase_history_tree_all_families"
CURRENT_STATE_TREE_MODEL = "online_current_state_tree_all_families"
GLOBAL_ONCE_MODELS = {
    "baseline_last_state",
    "baseline_majority",
    "baseline_state_conditioned_majority",
    "markov_phase_predictor",
    "rle_markov_phase_predictor",
    "hsmm_duration_phase_predictor",
    CURRENT_STATE_TREE_MODEL,
    ALL_FAMILY_TREE_MODEL,
    "rocket_phase_classifier_all_families",
    "tcn_phase_classifier_all_families",
    "tiny_transformer_phase_classifier_all_families",
}


def _report_mask(split: np.ndarray) -> np.ndarray:
    test = split == "test"
    if np.any(test):
        return test
    val = split == "val"
    if np.any(val):
        return val
    return split != "train"


def _report_split_name(split: np.ndarray) -> str:
    if np.any(split == "test"):
        return "test"
    if np.any(split == "val"):
        return "val"
    return "non_train"


def _counter_selection_split_name(split: np.ndarray) -> str:
    if np.any(split == "val"):
        return "val"
    if np.any(split == "test"):
        return "test"
    return "non_train"


def _fill_train_medians(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col in range(filled.shape[1]):
        values = filled[train_mask, col]
        clean = values[np.isfinite(values)]
        median = float(np.median(clean)) if clean.size else 0.0
        filled[~np.isfinite(filled[:, col]), col] = median
    return filled


def _assert_cluster_state_matrix(matrix: np.ndarray, *, label: str) -> None:
    """Guard that online models only see clustered state IDs, never raw values."""

    if matrix.size == 0:
        return
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    rounded = np.rint(finite)
    if not np.allclose(finite, rounded):
        raise ValueError(f"{label} contains non-integer inputs; expected clustered state IDs only.")
    ints = rounded.astype(int)
    allowed = {-1, 0, 1, 2}
    if any(int(value) not in allowed for value in ints.tolist()):
        raise ValueError(f"{label} contains values outside clustered state IDs {sorted(allowed)}.")


def _fill_and_scale(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col in range(filled.shape[1]):
        train_values = filled[train_mask, col]
        clean = train_values[np.isfinite(train_values)]
        median = float(np.median(clean)) if clean.size else 0.0
        filled[~np.isfinite(filled[:, col]), col] = median
        base = filled[train_mask, col]
        mean = float(np.mean(base)) if base.size else 0.0
        std = float(np.std(base)) if base.size else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        filled[:, col] = (filled[:, col] - mean) / std
    return filled


def _kmeans(train_x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if train_x.shape[0] == 0:
        return np.empty((0, train_x.shape[1]), dtype=float)
    unique = np.unique(train_x, axis=0)
    k = min(max(1, n_clusters), unique.shape[0])
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(unique.shape[0], size=k, replace=False)
    centroids = unique[init_idx].astype(float, copy=True)
    labels = np.zeros(train_x.shape[0], dtype=int)
    for _ in range(100):
        distances = np.sum((train_x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = train_x[new_labels == cluster]
            if members.shape[0] > 0:
                new_centroids[cluster] = np.mean(members, axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return centroids


def _nearest_centroid(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if centroids.shape[0] == 0:
        return np.full(x.shape[0], -1, dtype=int)
    distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


def _cluster_counter_value_states(values: np.ndarray, train_mask: np.ndarray, seed: int) -> np.ndarray:
    matrix = np.full((values.shape[0], 1), np.nan, dtype=float)
    for index, value in enumerate(values):
        if np.isfinite(value):
            matrix[index, 0] = math.log1p(value) if value >= 0.0 else value
    scaled = _fill_and_scale(matrix, train_mask)
    centroids = _kmeans(scaled[train_mask], 3, seed)
    raw_labels = _nearest_centroid(scaled, centroids)
    if centroids.shape[0] == 0:
        return np.full(values.shape[0], -1, dtype=int)
    order = sorted(range(centroids.shape[0]), key=lambda cluster: float(centroids[cluster, 0]))
    remap = {cluster: rank for rank, cluster in enumerate(order)}
    return np.asarray([remap.get(int(label), -1) for label in raw_labels], dtype=int)


def _group_indices(metadata_rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(metadata_rows):
        key = (str(row.get("run_id", "")), str(row.get("core_id", "")))
        grouped.setdefault(key, []).append(index)
    for key in grouped:
        grouped[key].sort(key=lambda idx: float(metadata_rows[idx].get("timestamp_ms", "0") or 0.0))
    return list(grouped.values())


def _history_examples(
    values: np.ndarray,
    phases: np.ndarray,
    split: np.ndarray,
    metadata_rows: list[dict[str, str]],
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    target: list[int] = []
    splits: list[str] = []
    rows: list[dict[str, str]] = []
    row_ids: list[int] = []
    current_phase: list[int] = []
    current_run_length: list[int] = []
    for stream in _group_indices(metadata_rows):
        if len(stream) < history_length + 1:
            continue
        run_lengths = np.ones(len(stream), dtype=int)
        for pos in range(1, len(stream)):
            if int(phases[stream[pos]]) == int(phases[stream[pos - 1]]):
                run_lengths[pos] = run_lengths[pos - 1] + 1
        for pos in range(history_length - 1, len(stream) - 1):
            row_index = stream[pos]
            next_index = stream[pos + 1]
            window = stream[pos - history_length + 1 : pos + 1]
            features.append(values[window].reshape(-1))
            target.append(int(phases[next_index]))
            splits.append(str(split[row_index]))
            rows.append(metadata_rows[row_index])
            row_ids.append(row_index)
            current_phase.append(int(phases[row_index]))
            current_run_length.append(int(run_lengths[pos]))
    if not features:
        width = values.shape[1] * history_length
        return (
            np.empty((0, width), dtype=float),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
            [],
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return (
        np.asarray(features, dtype=float),
        np.asarray(target, dtype=int),
        np.asarray(splits, dtype=object),
        rows,
        np.asarray(row_ids, dtype=int),
        np.asarray(current_phase, dtype=int),
        np.asarray(current_run_length, dtype=int),
    )


def _subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & (y_true >= 0) & (y_pred >= 0)
    if not np.any(usable):
        return 0.0
    return float(np.mean(y_true[usable] == y_pred[usable]))


def _mean_metric(rows: list[dict[str, object]], model: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row.get("model") == model and key in row]
    return float(np.mean(values)) if values else 0.0


def _prepare_phase_detector_data(
    *,
    experiment_dir: Path,
    scope: str,
    horizon: int,
    history_length: int,
) -> list[dict[str, object]]:
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    families, current, _future, split, metadata_rows = states_matrix(family_data, horizon)
    if not families:
        raise SystemExit(
            f"No selected family sequence CSVs found for experiment={experiment_dir.name} scope={scope}. "
            f"Run ablation and rebuild counter sequences first."
        )

    value_matrix = np.full((current.shape[0], len(families)), np.nan, dtype=float)
    selected_counters: dict[str, str] = {}
    for family_index, family in enumerate(families):
        rows = family_data[family].rows[: current.shape[0]]
        for row_index, row in enumerate(rows):
            if row_index == 0:
                selected_counters[family] = str(row.get("selected_counter", row.get("counter_name", "")))
            value = row.get("counter_value", "")
            try:
                value_matrix[row_index, family_index] = float(value)
            except (TypeError, ValueError):
                value_matrix[row_index, family_index] = math.nan

    train_row_mask = split == "train"
    if not np.any(train_row_mask):
        train_row_mask = np.ones(split.shape[0], dtype=bool)
    counter_cluster_states = np.full(value_matrix.shape, -1, dtype=int)
    seed = 17
    for family_index in range(len(families)):
        counter_cluster_states[:, family_index] = _cluster_counter_value_states(
            value_matrix[:, family_index],
            train_row_mask,
            seed + family_index,
        )

    prepared: list[dict[str, object]] = []
    all_counter_map = ";".join(f"{family}:{selected_counters.get(family, '')}" for family in families)
    for family_index, family in enumerate(families):
        phases = current[:, family_index].astype(int)
        if not np.any(phases >= 0):
            continue
        prepared.append(
            {
                "family": family,
                "selected_counter": selected_counters.get(family, ""),
                "selected_counter_map": all_counter_map,
                "local_feature_matrix": counter_cluster_states[:, [family_index]],
                "local_feature_names": [family],
                "all_feature_matrix": counter_cluster_states.copy(),
                "all_feature_names": families,
                "phases": phases,
                "split": split,
                "metadata_rows": metadata_rows,
                "history_length": history_length,
                "experiment": experiment_dir.name,
                "scope": scope,
            }
        )
    if not prepared:
        raise SystemExit(f"No valid phase-detector families for experiment={experiment_dir.name} scope={scope}")
    return prepared


def _prepare_history_dataset(
    *,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    phases: np.ndarray,
    split: np.ndarray,
    metadata_rows: list[dict[str, str]],
    history_length: int,
) -> dict[str, object]:
    _assert_cluster_state_matrix(feature_matrix, label=f"feature_matrix[{','.join(feature_names)}]")
    raw_features, target, example_split, example_rows, row_ids, current_phase, current_run_length = _history_examples(
        feature_matrix.astype(float),
        phases,
        split,
        metadata_rows,
        history_length,
    )
    if raw_features.shape[0] == 0:
        raise SystemExit("No history windows for prepared family detector dataset")
    train_base = example_split == "train"
    if not np.any(train_base):
        train_base = np.ones(example_split.shape[0], dtype=bool)
    eval_base = _report_mask(example_split)
    valid = target >= 0
    hist_features = _fill_train_medians(raw_features, train_base & valid if np.any(train_base & valid) else train_base)
    _assert_cluster_state_matrix(hist_features, label=f"history_features[{','.join(feature_names)}]")
    sequence_features = hist_features.reshape(hist_features.shape[0], history_length, feature_matrix.shape[1])
    _assert_cluster_state_matrix(sequence_features.reshape(sequence_features.shape[0], -1), label=f"sequence_features[{','.join(feature_names)}]")
    return {
        # Align current-state features with the history examples and their targets.
        # Returning the original row matrix here silently shifted prediction exports
        # whenever history_length > 1.
        "features": sequence_features[:, -1, :],
        "feature_names": feature_names,
        "target": target,
        "example_split": example_split,
        "example_rows": example_rows,
        "row_ids": row_ids,
        "current_phase": current_phase,
        "current_run_length": current_run_length,
        "hist_features": hist_features,
        "sequence_features": sequence_features,
        "train_mask": train_base & valid,
        "eval_mask": eval_base & valid,
        "report_split": _report_split_name(example_split),
        "counter_selection_split": _counter_selection_split_name(example_split),
        "history_length": history_length,
    }


def _summary_row(
    *,
    experiment: str,
    scope: str,
    family: str,
    selected_counter: str,
    model: str,
    feature_source: str,
    history_length: int,
    selected_counter_map: str,
    target: np.ndarray,
    pred: np.ndarray,
    current_phase: np.ndarray,
    group_ids: np.ndarray,
    eval_mask: np.ndarray,
    train_rows: int,
    eval_rows: int,
    budget: dict[str, object],
    details: dict[str, object],
    training_seconds: float,
    inference_seconds: float,
    report_split: str,
    counter_selection_split: str,
    bootstrap_resamples: int,
    warning: str = "",
) -> dict[str, object]:
    usable = eval_mask & (pred >= 0) & (target >= 0)
    metrics = classification_metrics(target[usable], pred[usable], current_state=current_phase[usable]) if np.any(usable) else {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "balanced_accuracy": 0.0,
        "high_usage_recall": 0.0,
    }
    stable_mask = usable & (target == current_phase)
    transition_mask = usable & (target != current_phase)
    intervals = grouped_bootstrap_confidence_intervals(
        target[usable],
        pred[usable],
        current_phase[usable],
        group_ids[usable],
        n_resamples=bootstrap_resamples,
    )
    transition_metrics = transition_event_metrics(target[usable], pred[usable], current_phase[usable])
    return {
        "experiment": experiment,
        "scope": scope,
        "family": family,
        "selected_counter": selected_counter,
        "model": model,
        "feature_source": feature_source,
        "phase_label_source": "train_split_kmeans_full_safe_counters",
        "counter_selection_split": counter_selection_split,
        "report_split": report_split,
        "prediction_step": 1,
        "history_length": history_length,
        "prediction_horizon": 1,
        "top1_accuracy": metrics["accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "high_usage_recall": metrics["high_usage_recall"],
        "stable_case_accuracy": _subset_accuracy(target, pred, stable_mask),
        "transition_case_accuracy": _subset_accuracy(target, pred, transition_mask),
        "stable_eval_rows": int(np.sum(stable_mask)),
        "transition_eval_rows": int(np.sum(transition_mask)),
        **transition_metrics,
        **intervals,
        "selected_counter_map": selected_counter_map,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_latency_us": float((inference_seconds / max(1, int(np.sum(usable)))) * 1e6),
        "tree_depth": details.get("tree_depth", ""),
        "tree_max_depth": details.get("tree_max_depth", ""),
        "tree_internal_nodes": details.get("tree_internal_nodes", ""),
        "tree_leaves": details.get("tree_leaves", ""),
        "estimated_storage_bytes": details.get("estimated_storage_bytes", ""),
        "feature_count": details.get("feature_count", ""),
        "parameter_count": budget["parameter_count"],
        "table_entries": budget["table_entries"],
        "stored_parameters_or_entries": budget["stored_parameters_or_entries"],
        "estimated_memory_bytes": budget["estimated_memory_bytes"],
        "model_storage_bytes": budget["model_storage_bytes"],
        "history_storage_bytes": budget["history_storage_bytes"],
        "discretizer_storage_bytes": budget["discretizer_storage_bytes"],
        "approx_operations_per_prediction": budget["approx_operations_per_prediction"],
        "operator_type": budget["operator_type"],
        "hardware_complexity_category": budget["hardware_complexity_category"],
        "deployment_recommendation": budget["deployment_recommendation"],
        "model_warning": warning,
    }


def _prediction_rows(
    *,
    model: str,
    feature_source: str,
    family: str,
    selected_counter: str,
    dataset: dict[str, object],
    pred: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_names = list(dataset["feature_names"])
    features = np.asarray(dataset["features"], dtype=float)
    example_rows = list(dataset["example_rows"])
    row_ids = np.asarray(dataset["row_ids"], dtype=int)
    current_phase = np.asarray(dataset["current_phase"], dtype=int)
    current_run_length = np.asarray(dataset["current_run_length"], dtype=int)
    target = np.asarray(dataset["target"], dtype=int)
    example_split = np.asarray(dataset["example_split"], dtype=object)
    for index, row in enumerate(example_rows):
        item: dict[str, object] = {
            "model": model,
            "family": family,
            "selected_counter": selected_counter,
            "feature_source": feature_source,
            "split": example_split[index],
            "workload": row.get("workload", ""),
            "run_id": row.get("run_id", ""),
            "core_id": row.get("core_id", ""),
            "threads": row.get("threads", ""),
            "process_index": row.get("process_index", ""),
            "process_count": row.get("process_count", ""),
            "concurrent_group_id": row.get("concurrent_group_id", ""),
            "rep": row.get("rep", ""),
            "experiment_set": row.get("experiment_set", ""),
            "requested_input_size": row.get("requested_input_size", ""),
            "co_running_workloads": row.get("co_running_workloads", ""),
            "row_index": int(row.get("row_index", row_ids[index]) or row_ids[index]),
            "current_phase": int(current_phase[index]),
            "current_run_length": int(current_run_length[index]),
            "history_length": int(dataset["history_length"]),
            "y_true_phase_t_plus_1": int(target[index]),
            "y_pred_phase_t_plus_1": int(pred[index]),
        }
        for feature_index, feature_name in enumerate(feature_names):
            item[f"current_cluster_state_{feature_name}"] = int(features[index, feature_index])
        rows.append(item)
    return rows


def _confusion_rows(
    *,
    experiment: str,
    scope: str,
    family: str,
    model: str,
    target: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
) -> list[dict[str, object]]:
    usable = eval_mask & (target >= 0) & (pred >= 0)
    rows: list[dict[str, object]] = []
    for true_label in range(3):
        for pred_label in range(3):
            count = int(np.sum(usable & (target == true_label) & (pred == pred_label)))
            rows.append(
                {
                    "experiment": experiment,
                    "scope": scope,
                    "family": family,
                    "model": model,
                    "true_phase": true_label,
                    "pred_phase": pred_label,
                    "count": count,
                }
            )
    return rows


def _group_accuracy_rows(
    *,
    experiment: str,
    scope: str,
    family: str,
    model: str,
    group_name: str,
    group_values: list[str],
    target: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
) -> list[dict[str, object]]:
    usable = eval_mask & (target >= 0) & (pred >= 0)
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(group_values):
        if usable[index]:
            grouped.setdefault(str(value), []).append(index)
    rows: list[dict[str, object]] = []
    for value, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=int)
        rows.append(
            {
                "experiment": experiment,
                "scope": scope,
                "family": family,
                "model": model,
                group_name: value,
                "rows": int(idx.size),
                "accuracy": float(np.mean(target[idx] == pred[idx])) if idx.size else 0.0,
            }
        )
    return rows


def _thread_process_group(values_threads: list[str], values_processes: list[str]) -> list[str]:
    output: list[str] = []
    for threads, processes in zip(values_threads, values_processes):
        thread_text = str(threads).strip() or "unknown"
        process_text = str(processes).strip() or "unknown"
        output.append(f"threads={thread_text}|processes={process_text}")
    return output


def _phase_behavior_rows(
    *,
    experiment: str,
    scope: str,
    family: str,
    group_name: str,
    group_values: list[str],
    target: np.ndarray,
    current_phase: np.ndarray,
    current_run_length: np.ndarray,
    eval_mask: np.ndarray,
) -> list[dict[str, object]]:
    usable = eval_mask & (target >= 0) & (current_phase >= 0)
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(group_values):
        if usable[index]:
            grouped.setdefault(str(value), []).append(index)
    rows: list[dict[str, object]] = []
    for value, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=int)
        current = current_phase[idx]
        future = target[idx]
        rows.append(
            {
                "experiment": experiment,
                "scope": scope,
                "family": family,
                group_name: value,
                "rows": int(idx.size),
                "transition_rate": float(np.mean(future != current)) if idx.size else 0.0,
                "stable_rate": float(np.mean(future == current)) if idx.size else 0.0,
                "current_state_0_fraction": float(np.mean(current == 0)) if idx.size else 0.0,
                "current_state_1_fraction": float(np.mean(current == 1)) if idx.size else 0.0,
                "current_state_2_fraction": float(np.mean(current == 2)) if idx.size else 0.0,
                "next_state_0_fraction": float(np.mean(future == 0)) if idx.size else 0.0,
                "next_state_1_fraction": float(np.mean(future == 1)) if idx.size else 0.0,
                "next_state_2_fraction": float(np.mean(future == 2)) if idx.size else 0.0,
                "avg_run_length": float(np.mean(current_run_length[idx])) if idx.size else 0.0,
            }
        )
    return rows


def _evaluate_model(
    *,
    experiment: str,
    scope: str,
    family: str,
    selected_counter: str,
    selected_counter_map: str,
    model: str,
    feature_source: str,
    dataset: dict[str, object],
    run,
    bootstrap_resamples: int,
) -> tuple[
    dict[str, object] | None,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if bool(run.details.get("skipped", False)):
        print(f"[phase_detector] skip family={family} model={model} reason={run.warning}", flush=True)
        return None, [], [], [], [], [], [], [], []
    target = np.asarray(dataset["target"], dtype=int)
    pred = np.asarray(run.predictions, dtype=int)
    eval_mask = np.asarray(dataset["eval_mask"], dtype=bool)
    effective_history_length = 1 if model == CURRENT_STATE_TREE_MODEL else int(dataset["history_length"])
    requires_oracle_current_phase = feature_source.startswith("oracle_current_teacher_phase")
    budget = estimate_hardware_budget(
        model,
        {
            **run.details,
            "parameter_count": run.parameter_count,
            "table_entries": run.table_entries,
            "history_length": int(dataset["history_length"]),
            "effective_history_length": effective_history_length,
            "online_family_count": len(list(dataset["feature_names"])),
            "uses_clustered_counter_state": feature_source.startswith("clustered_counter"),
            "requires_oracle_current_phase": requires_oracle_current_phase,
        },
    )
    example_rows = list(dataset["example_rows"])
    group_ids = np.asarray(
        [str(row.get("concurrent_group_id", "")).strip() or str(row.get("run_id", "")).strip() for row in example_rows],
        dtype=object,
    )
    summary = _summary_row(
        experiment=experiment,
        scope=scope,
        family=family,
        selected_counter=selected_counter,
        model=model,
        feature_source=feature_source,
        history_length=int(dataset["history_length"]),
        selected_counter_map=selected_counter_map,
        target=target,
        pred=pred,
        current_phase=np.asarray(dataset["current_phase"], dtype=int),
        group_ids=group_ids,
        eval_mask=eval_mask,
        train_rows=int(np.sum(np.asarray(dataset["train_mask"], dtype=bool))),
        eval_rows=int(np.sum(eval_mask & (target >= 0) & (pred >= 0))),
        budget=budget,
        details=run.details,
        training_seconds=run.training_seconds,
        inference_seconds=run.inference_seconds,
        report_split=str(dataset.get("report_split", "")),
        counter_selection_split=str(dataset.get("counter_selection_split", "")),
        bootstrap_resamples=bootstrap_resamples,
        warning=run.warning,
    )
    summary["requires_oracle_current_phase"] = int(requires_oracle_current_phase)
    predictions = _prediction_rows(
        model=model,
        feature_source=feature_source,
        family=family,
        selected_counter=selected_counter,
        dataset=dataset,
        pred=pred,
    )
    workloads = [str(row.get("workload", "")) for row in example_rows]
    cores = [str(row.get("core_id", "")) for row in example_rows]
    confusion = _confusion_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    workload_accuracy = _group_accuracy_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        group_name="workload",
        group_values=workloads,
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    core_accuracy = _group_accuracy_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        group_name="core_id",
        group_values=cores,
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    thread_values = [str(row.get("threads", "")) for row in example_rows]
    process_values = [str(row.get("process_count", "")) for row in example_rows]
    thread_accuracy = _group_accuracy_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        group_name="threads",
        group_values=thread_values,
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    process_accuracy = _group_accuracy_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        group_name="process_count",
        group_values=process_values,
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    thread_process_accuracy = _group_accuracy_rows(
        experiment=experiment,
        scope=scope,
        family=family,
        model=model,
        group_name="thread_process_config",
        group_values=_thread_process_group(thread_values, process_values),
        target=target,
        pred=pred,
        eval_mask=eval_mask,
    )
    phase_behavior = (
        _phase_behavior_rows(
            experiment=experiment,
            scope=scope,
            family=family,
            group_name="thread_process_config",
            group_values=_thread_process_group(thread_values, process_values),
            target=target,
            current_phase=np.asarray(dataset["current_phase"], dtype=int),
            current_run_length=np.asarray(dataset["current_run_length"], dtype=int),
            eval_mask=eval_mask,
        )
        if model == "baseline_last_state"
        else []
    )
    return (
        summary,
        predictions,
        confusion,
        workload_accuracy,
        core_accuracy,
        thread_accuracy,
        process_accuracy,
        thread_process_accuracy,
        phase_behavior,
    )


def train_phase_detector_for_experiment(
    *,
    experiment_dir: Path,
    scope: str,
    output_dir: Path,
    horizon: int,
    history_length: int,
    prediction_horizon: int,
    tree_max_depth: int,
    tree_min_leaf: int,
    detector_config: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Train per-family next-phase predictors and baselines."""

    if prediction_horizon != 1:
        raise ValueError("Online phase detector currently supports only next-phase prediction (prediction_horizon=1).")

    detector_config = dict(detector_config or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_items = _prepare_phase_detector_data(
        experiment_dir=experiment_dir,
        scope=scope,
        horizon=horizon,
        history_length=history_length,
    )
    print(
        f"[phase_detector] prepared experiment={experiment_dir.name} scope={scope} "
        f"families={len(prepared_items)} history={history_length} horizon=1 inputs=clustered_state_ids_only",
        flush=True,
    )

    bucket_edges = [1, 3, 7, 15]
    rocket_kernels = int(detector_config.get("rocket_kernels", 32))
    neural_epochs = int(detector_config.get("neural_epochs", 3))
    neural_batch_size = int(detector_config.get("neural_batch_size", 128))
    neural_learning_rate = float(detector_config.get("neural_learning_rate", 1e-3))
    tcn_channels = int(detector_config.get("tcn_channels", 16))
    transformer_hidden_dim = int(detector_config.get("transformer_hidden_dim", 32))
    transformer_num_layers = int(detector_config.get("transformer_num_layers", 2))
    transformer_num_heads = int(detector_config.get("transformer_num_heads", 4))
    transformer_ff_dim = int(detector_config.get("transformer_ff_dim", 64))
    bootstrap_resamples = int(detector_config.get("bootstrap_resamples", 1000))

    all_summary_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    workload_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    thread_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []
    thread_process_rows: list[dict[str, object]] = []
    phase_behavior_thread_rows: list[dict[str, object]] = []
    phase_behavior_process_rows: list[dict[str, object]] = []
    phase_behavior_thread_process_rows: list[dict[str, object]] = []
    hardware_rows: list[dict[str, object]] = []
    family_aggregates: list[dict[str, object]] = []
    skipped_models: list[dict[str, str]] = []
    reference_family = str(prepared_items[0]["family"])
    reference_phases = np.asarray(prepared_items[0]["phases"], dtype=int)
    for item in prepared_items[1:]:
        if not np.array_equal(reference_phases, np.asarray(item["phases"], dtype=int)):
            raise ValueError("Selected family streams do not share one global teacher target")

    for prepared in prepared_items:
        family = str(prepared["family"])
        print(
            f"[phase_detector] family={family} experiment={prepared['experiment']} scope={prepared['scope']} "
            f"selected_counter={prepared['selected_counter']}",
            flush=True,
        )
        local_dataset = _prepare_history_dataset(
            feature_matrix=np.asarray(prepared["local_feature_matrix"], dtype=float),
            feature_names=list(prepared["local_feature_names"]),
            phases=np.asarray(prepared["phases"], dtype=int),
            split=np.asarray(prepared["split"], dtype=object),
            metadata_rows=list(prepared["metadata_rows"]),
            history_length=int(prepared["history_length"]),
        )
        all_dataset = _prepare_history_dataset(
            feature_matrix=np.asarray(prepared["all_feature_matrix"], dtype=float),
            feature_names=list(prepared["all_feature_names"]),
            phases=np.asarray(prepared["phases"], dtype=int),
            split=np.asarray(prepared["split"], dtype=object),
            metadata_rows=list(prepared["metadata_rows"]),
            history_length=int(prepared["history_length"]),
        )
        target = np.asarray(local_dataset["target"], dtype=int)
        train_mask = np.asarray(local_dataset["train_mask"], dtype=bool)
        current_phase = np.asarray(local_dataset["current_phase"], dtype=int)
        current_run_length = np.asarray(local_dataset["current_run_length"], dtype=int)
        example_rows = list(local_dataset["example_rows"])
        thread_values = [str(row.get("threads", "")) for row in example_rows]
        process_values = [str(row.get("process_count", "")) for row in example_rows]
        if family == reference_family:
            phase_behavior_thread_rows.extend(
                _phase_behavior_rows(
                    experiment=str(prepared["experiment"]),
                    scope=str(prepared["scope"]),
                    family="__all_families__",
                    group_name="threads",
                    group_values=thread_values,
                    target=target,
                    current_phase=current_phase,
                    current_run_length=current_run_length,
                    eval_mask=np.asarray(local_dataset["eval_mask"], dtype=bool),
                )
            )
            phase_behavior_process_rows.extend(
                _phase_behavior_rows(
                    experiment=str(prepared["experiment"]),
                    scope=str(prepared["scope"]),
                    family="__all_families__",
                    group_name="process_count",
                    group_values=process_values,
                    target=target,
                    current_phase=current_phase,
                    current_run_length=current_run_length,
                    eval_mask=np.asarray(local_dataset["eval_mask"], dtype=bool),
                )
            )
            phase_behavior_thread_process_rows.extend(
                _phase_behavior_rows(
                    experiment=str(prepared["experiment"]),
                    scope=str(prepared["scope"]),
                    family="__all_families__",
                    group_name="thread_process_config",
                    group_values=_thread_process_group(thread_values, process_values),
                    target=target,
                    current_phase=current_phase,
                    current_run_length=current_run_length,
                    eval_mask=np.asarray(local_dataset["eval_mask"], dtype=bool),
                )
            )
        print(
            f"[phase_detector] family={family} local_features={local_dataset['sequence_features'].shape[2]} "
            f"all_features={all_dataset['sequence_features'].shape[2]} train_rows={int(np.sum(train_mask))} "
            f"eval_rows={int(np.sum(np.asarray(local_dataset['eval_mask'], dtype=bool)))}",
            flush=True,
        )

        run_specs = [
            ("baseline_last_state", "oracle_current_teacher_phase", local_dataset, lambda: run_last_state_baseline(current_phase)),
            ("baseline_majority", "train_label_prior_only", local_dataset, lambda: run_majority_baseline(target, train_mask)),
            ("baseline_state_conditioned_majority", "oracle_current_teacher_phase", local_dataset, lambda: run_state_conditioned_majority(current_phase, target, train_mask)),
            ("markov_phase_predictor", "oracle_current_teacher_phase", local_dataset, lambda: run_markov(current_phase, target, train_mask)),
            ("rle_markov_phase_predictor", "oracle_current_teacher_phase_and_run_length", local_dataset, lambda: run_rle_markov(current_phase, current_run_length, target, train_mask, bucket_edges)),
            ("hsmm_duration_phase_predictor", "oracle_current_teacher_phase_and_run_length", local_dataset, lambda: run_hsmm_approx(current_phase, current_run_length, target, train_mask, bucket_edges)),
            (
                CURRENT_STATE_TREE_MODEL,
                "clustered_counter_current_state_all_families",
                all_dataset,
                lambda: run_decision_tree(
                    np.asarray(all_dataset["features"], dtype=float),
                    target,
                    np.asarray(all_dataset["train_mask"], dtype=bool),
                    tree_max_depth,
                    tree_min_leaf,
                ),
            ),
            (
                TREE_MODEL,
                "clustered_counter_history_local",
                local_dataset,
                lambda: run_decision_tree(np.asarray(local_dataset["hist_features"], dtype=float), target, train_mask, tree_max_depth, tree_min_leaf),
            ),
            (
                ALL_FAMILY_TREE_MODEL,
                "clustered_counter_history_all_families",
                all_dataset,
                lambda: run_decision_tree(
                    np.asarray(all_dataset["hist_features"], dtype=float),
                    target,
                    np.asarray(all_dataset["train_mask"], dtype=bool),
                    tree_max_depth,
                    tree_min_leaf,
                ),
            ),
            (
                "rocket_phase_classifier",
                "clustered_counter_history_local",
                local_dataset,
                lambda: run_rocket_fallback(np.asarray(local_dataset["sequence_features"], dtype=float), target, train_mask, rocket_kernels),
            ),
            (
                "rocket_phase_classifier_all_families",
                "clustered_counter_history_all_families",
                all_dataset,
                lambda: run_rocket_fallback(
                    np.asarray(all_dataset["sequence_features"], dtype=float),
                    target,
                    np.asarray(all_dataset["train_mask"], dtype=bool),
                    rocket_kernels,
                ),
            ),
            (
                "tcn_phase_classifier",
                "clustered_counter_history_local",
                local_dataset,
                lambda: run_tcn(
                    np.asarray(local_dataset["sequence_features"], dtype=float),
                    target,
                    train_mask,
                    channels=tcn_channels,
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                ),
            ),
            (
                "tcn_phase_classifier_all_families",
                "clustered_counter_history_all_families",
                all_dataset,
                lambda: run_tcn(
                    np.asarray(all_dataset["sequence_features"], dtype=float),
                    target,
                    np.asarray(all_dataset["train_mask"], dtype=bool),
                    channels=tcn_channels,
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                ),
            ),
            (
                "tiny_transformer_phase_classifier",
                "clustered_counter_history_local",
                local_dataset,
                lambda: run_transformer(
                    np.asarray(local_dataset["sequence_features"], dtype=float),
                    target,
                    train_mask,
                    hidden_dim=transformer_hidden_dim,
                    num_layers=transformer_num_layers,
                    num_heads=transformer_num_heads,
                    ff_dim=transformer_ff_dim,
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                ),
            ),
            (
                "tiny_transformer_phase_classifier_all_families",
                "clustered_counter_history_all_families",
                all_dataset,
                lambda: run_transformer(
                    np.asarray(all_dataset["sequence_features"], dtype=float),
                    target,
                    np.asarray(all_dataset["train_mask"], dtype=bool),
                    hidden_dim=transformer_hidden_dim,
                    num_layers=transformer_num_layers,
                    num_heads=transformer_num_heads,
                    ff_dim=transformer_ff_dim,
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                ),
            ),
        ]
        enabled_models = detector_config.get("enabled_models", [])
        if isinstance(enabled_models, list) and enabled_models:
            enabled = {str(value) for value in enabled_models}
            run_specs = [spec for spec in run_specs if spec[0] in enabled]
        if family != reference_family:
            run_specs = [spec for spec in run_specs if spec[0] not in GLOBAL_ONCE_MODELS]

        summary_by_model: dict[str, dict[str, object]] = {}
        family_model_rows: list[dict[str, object]] = []
        for model_index, (model_name, feature_source, dataset, runner) in enumerate(run_specs, start=1):
            result_family = "__all_families__" if model_name in GLOBAL_ONCE_MODELS else family
            result_selected_counter = (
                str(prepared["selected_counter_map"])
                if model_name in GLOBAL_ONCE_MODELS
                else str(prepared["selected_counter"])
            )
            print(
                f"[phase_detector] family={family} model={model_name} "
                f"feature_source={feature_source} start {model_index}/{len(run_specs)}",
                flush=True,
            )
            run = runner()
            (
                summary,
                predictions,
                confusion,
                workload_accuracy,
                core_accuracy,
                thread_accuracy,
                process_accuracy,
                thread_process_accuracy,
                _phase_behavior_unused,
            ) = _evaluate_model(
                experiment=str(prepared["experiment"]),
                scope=str(prepared["scope"]),
                family=result_family,
                selected_counter=result_selected_counter,
                selected_counter_map=str(prepared["selected_counter_map"]),
                model=model_name,
                feature_source=feature_source,
                dataset=dataset,
                run=run,
                bootstrap_resamples=bootstrap_resamples,
            )
            if summary is None:
                skipped_models.append(
                    {
                        "experiment": str(prepared["experiment"]),
                        "scope": str(prepared["scope"]),
                        "family": family,
                        "model": model_name,
                        "reason": run.warning,
                    }
                )
                continue
            print(
                f"[phase_detector] family={family} model={model_name} done "
                f"accuracy={float(summary['accuracy']):.4f} macro_f1={float(summary['macro_f1']):.4f} "
                f"transition_accuracy={float(summary['transition_case_accuracy']):.4f} "
                f"latency_us={float(summary['inference_latency_us']):.2f}",
                flush=True,
            )
            summary_by_model[model_name] = summary
            family_model_rows.append(summary)
            all_summary_rows.append(summary)
            all_prediction_rows.extend(predictions)
            confusion_rows.extend(confusion)
            workload_rows.extend(workload_accuracy)
            core_rows.extend(core_accuracy)
            thread_rows.extend(thread_accuracy)
            process_rows.extend(process_accuracy)
            thread_process_rows.extend(thread_process_accuracy)
            hardware_rows.append(
                {
                    "experiment": summary["experiment"],
                    "scope": summary["scope"],
                    "family": summary["family"],
                    "model": summary["model"],
                    "feature_source": summary["feature_source"],
                    "stored_parameters_or_entries": summary["stored_parameters_or_entries"],
                    "parameter_count": summary["parameter_count"],
                    "table_entries": summary["table_entries"],
                    "estimated_memory_bytes": summary["estimated_memory_bytes"],
                    "model_storage_bytes": summary["model_storage_bytes"],
                    "history_storage_bytes": summary["history_storage_bytes"],
                    "discretizer_storage_bytes": summary["discretizer_storage_bytes"],
                    "approx_operations_per_prediction": summary["approx_operations_per_prediction"],
                    "operator_type": summary["operator_type"],
                    "hardware_complexity_category": summary["hardware_complexity_category"],
                    "deployment_recommendation": summary["deployment_recommendation"],
                    "requires_oracle_current_phase": summary["requires_oracle_current_phase"],
                }
            )

        last_row = summary_by_model.get("baseline_last_state")
        majority_row = summary_by_model.get("baseline_majority")
        state_row = summary_by_model.get("baseline_state_conditioned_majority")
        for row in family_model_rows:
            if row["model"] in {"baseline_last_state", "baseline_majority", "baseline_state_conditioned_majority"}:
                continue
            if last_row is not None:
                row["accuracy_gain_vs_last_state"] = float(row["accuracy"]) - float(last_row["accuracy"])
                row["macro_f1_gain_vs_last_state"] = float(row["macro_f1"]) - float(last_row["macro_f1"])
                row["transition_accuracy_gain_vs_last_state"] = float(row["transition_case_accuracy"]) - float(last_row["transition_case_accuracy"])
            if majority_row is not None:
                row["accuracy_gain_vs_majority"] = float(row["accuracy"]) - float(majority_row["accuracy"])
                row["macro_f1_gain_vs_majority"] = float(row["macro_f1"]) - float(majority_row["macro_f1"])
            if state_row is not None:
                row["accuracy_gain_vs_state_conditioned_majority"] = float(row["accuracy"]) - float(state_row["accuracy"])
                row["macro_f1_gain_vs_state_conditioned_majority"] = float(row["macro_f1"]) - float(state_row["macro_f1"])

        family_aggregates.append(
            {
                "family": family,
                "selected_counter": str(prepared["selected_counter"]),
                "models": family_model_rows,
            }
        )
        print(
            f"[phase_detector] family={family} complete models={len(family_model_rows)} skipped={len([item for item in skipped_models if item['family'] == family])}",
            flush=True,
        )

    manifest = {
        "experiment": experiment_dir.name,
        "scope": scope,
        "model": "family_phase_detector_bank",
        "history_length": history_length,
        "prediction_horizon": 1,
        "family_count": len(family_aggregates),
        "model_count": len(all_summary_rows),
        "mean_top1_accuracy": _mean_metric(all_summary_rows, TREE_MODEL, "top1_accuracy"),
        "mean_accuracy": _mean_metric(all_summary_rows, TREE_MODEL, "accuracy"),
        "mean_macro_f1": _mean_metric(all_summary_rows, TREE_MODEL, "macro_f1"),
        "selected_counter_map": ";".join(f"{item['family']}:{item['selected_counter']}" for item in family_aggregates),
        "families": family_aggregates,
        "skipped_models": skipped_models,
        "enabled_models": detector_config.get("enabled_models", "all"),
        "graph_tcn_todo": "Dataset does not expose a synchronous multicore tensor for graph baselines; leaving a clean TODO interface for future work.",
    }

    write_csv_rows(output_dir / "phase_detector_predictions.csv", all_prediction_rows)
    write_csv_rows(output_dir / "phase_detector_summary.csv", all_summary_rows)
    write_csv_rows(output_dir / "phase_detector_confusion_matrices.csv", confusion_rows)
    write_csv_rows(output_dir / "phase_detector_per_workload_accuracy.csv", workload_rows)
    write_csv_rows(output_dir / "phase_detector_per_core_accuracy.csv", core_rows)
    write_csv_rows(output_dir / "phase_detector_per_thread_accuracy.csv", thread_rows)
    write_csv_rows(output_dir / "phase_detector_per_process_count_accuracy.csv", process_rows)
    write_csv_rows(output_dir / "phase_detector_per_thread_process_accuracy.csv", thread_process_rows)
    write_csv_rows(output_dir / "phase_detector_phase_behavior_by_thread.csv", phase_behavior_thread_rows)
    write_csv_rows(output_dir / "phase_detector_phase_behavior_by_process_count.csv", phase_behavior_process_rows)
    write_csv_rows(output_dir / "phase_detector_phase_behavior_by_thread_process.csv", phase_behavior_thread_process_rows)
    write_csv_rows(output_dir / "phase_detector_hardware_budget.csv", hardware_rows)
    write_json(output_dir / "phase_detector_manifest.json", manifest)
    print(
        f"[phase_detector] wrote outputs summary={output_dir / 'phase_detector_summary.csv'} "
        f"predictions={output_dir / 'phase_detector_predictions.csv'} "
        f"hardware={output_dir / 'phase_detector_hardware_budget.csv'} "
        f"thread_accuracy={output_dir / 'phase_detector_per_thread_accuracy.csv'} "
        f"phase_behavior={output_dir / 'phase_detector_phase_behavior_by_thread_process.csv'}",
        flush=True,
    )
    return all_summary_rows


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--prediction-horizon", type=int, default=0)
    parser.add_argument("--tree-max-depth", type=int, default=0)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "phase_detector"))
    detector_cfg = dict(config["phase_detector"])
    history_length = args.history_length or int(detector_cfg["history_length"])
    prediction_horizon = args.prediction_horizon or int(detector_cfg["prediction_horizon"])
    tree_max_depth = args.tree_max_depth or int(detector_cfg["decision_tree_max_depth"])
    tree_min_leaf = args.tree_min_samples_leaf or int(detector_cfg["decision_tree_min_samples_leaf"])
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            print(
                f"[phase_detector] train experiment={exp_dir.name} scope={scope} "
                f"depth={tree_max_depth} history={history_length} horizon={prediction_horizon}",
                flush=True,
            )
            rows = train_phase_detector_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                output_dir=output_root / exp_dir.name / scope,
                horizon=int(dataset_cfg["horizon"]),
                history_length=history_length,
                prediction_horizon=prediction_horizon,
                tree_max_depth=tree_max_depth,
                tree_min_leaf=tree_min_leaf,
                detector_config=detector_cfg,
            )
            all_rows.extend(rows)
    write_csv_rows(output_root / "phase_detector_summary_all.csv", all_rows)
    print(f"[phase_detector] wrote_summary={output_root / 'phase_detector_summary_all.csv'} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
