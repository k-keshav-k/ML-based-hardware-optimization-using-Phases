"""Phase label generation for PARSEC interval features."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, safe_float, write_csv_rows, write_json

from .clustering import (
    PCATransformer,
    Standardizer,
    choose_gmm,
    choose_kmeans,
    enforce_min_duration,
    smooth_labels,
)
from .features import BASE_METADATA_COLUMNS, matrix_from_rows
from .splits import run_grouped_split, workload_holdout_split


def fit_predict_labels(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    config: dict[str, object],
    split_by_run: dict[str, str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    # Fit labeling model on train rows only, then predict labels for all intervals.
    matrix = matrix_from_rows(rows, feature_columns)
    train_indices = np.asarray([index for index, row in enumerate(rows) if split_by_run.get(row.get("run_id", ""), "train") == "train"], dtype=int)
    if train_indices.size == 0:
        train_indices = np.arange(matrix.shape[0])
    standardizer = Standardizer.fit(matrix[train_indices])
    transformed = standardizer.transform(matrix)
    train_matrix = transformed[train_indices]
    pca_payload: dict[str, object] = {"enabled": False}
    if bool(config.get("pca_enabled", True)) and train_matrix.shape[1] > 1:
        pca = PCATransformer.fit(train_matrix, float(config.get("pca_variance", 0.95)))
        transformed = pca.transform(transformed)
        train_matrix = transformed[train_indices]
        pca_payload = {
            "enabled": True,
            "mean": pca.mean.tolist(),
            "components": pca.components.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        }
    method = str(config.get("method", "fgmm"))
    k_min = int(config.get("k_min", 2))
    k_max = int(config.get("k_max", 12))
    max_iter = int(config.get("max_iter", 80))
    if method == "2kmeans":
        # Two-level k-means path: first on rows, then on segment centroids.
        first = choose_kmeans(train_matrix, k_min, k_max, seed, max_iter)
        raw_train = first.predict(train_matrix)
        segment_vectors = []
        index = 0
        while index < raw_train.size:
            start = index
            while index < raw_train.size and raw_train[index] == raw_train[start]:
                index += 1
            segment_vectors.append(train_matrix[start:index].mean(axis=0))
        segment_matrix = np.vstack(segment_vectors) if segment_vectors else train_matrix
        model = choose_kmeans(segment_matrix, k_min, min(k_max, len(segment_matrix)), seed + 101, max_iter)
        raw_labels = model.predict(transformed)
        model_payload = {"kind": "2kmeans", "centers": model.centers.tolist()}
    else:
        # Default FGMM path with BIC-based component count selection.
        model = choose_gmm(train_matrix, k_min, k_max, seed, max_iter)
        raw_labels = model.predict(transformed)
        model_payload = {
            "kind": "fgmm",
            "weights": model.weights.tolist(),
            "means": model.means.tolist(),
            "variances": model.variances.tolist(),
        }
    smoothed = raw_labels.copy()
    changed = np.zeros(raw_labels.shape[0], dtype=bool)
    by_unit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_unit[(row.get("run_id", ""), row.get("cpu_or_core_id", ""))].append(index)
    # Temporal cleanup is applied per stream so local phase continuity is respected.
    for indices in by_unit.values():
        indices.sort(key=lambda item: safe_float(rows[item].get("timestamp_ms", "")))
        local = raw_labels[indices]
        local, smooth_changed = smooth_labels(local, int(config.get("smoothing_window", 3)))
        local, min_changed = enforce_min_duration(local, int(config.get("min_phase_duration", 3)))
        smoothed[indices] = local
        changed[indices] = smooth_changed | min_changed
    payload = {
        "feature_columns": feature_columns,
        "standardizer": {
            "medians": standardizer.medians.tolist(),
            "means": standardizer.means.tolist(),
            "scales": standardizer.scales.tolist(),
        },
        "pca": pca_payload,
        "model": model_payload,
    }
    return smoothed.astype(int), changed, payload


def add_interval_targets(rows: list[dict[str, str]], labels: np.ndarray, changed: np.ndarray, split_by_run: dict[str, str], horizon: int) -> list[dict[str, object]]:
    # Convert raw labels into supervised targets (`next_phase_id`, `phase_change`) for training.
    phase_by_interval = {int(row["interval_index"]): int(labels[index]) for index, row in enumerate(rows)}
    by_unit: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_unit[(row.get("run_id", ""), row.get("cpu_or_core_id", ""))].append(index)
    next_by_index: dict[int, int] = {}
    run_length_by_index: dict[int, int] = {}
    for indices in by_unit.values():
        indices.sort(key=lambda item: safe_float(rows[item].get("timestamp_ms", "")))
        current_run = 0
        previous = None
        for local_pos, row_index in enumerate(indices):
            phase = int(labels[row_index])
            current_run = current_run + 1 if phase == previous else 1
            previous = phase
            interval_index = int(rows[row_index]["interval_index"])
            run_length_by_index[interval_index] = current_run
            target_pos = local_pos + horizon
            next_by_index[interval_index] = int(labels[indices[target_pos]]) if target_pos < len(indices) else -1
    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        interval_index = int(row["interval_index"])
        phase = phase_by_interval[interval_index]
        next_phase = next_by_index.get(interval_index, -1)
        item: dict[str, object] = {column: row.get(column, "") for column in BASE_METADATA_COLUMNS if column in row}
        item.update(
            {
                "split": split_by_run.get(row.get("run_id", ""), "train"),
                "phase_id": phase,
                "current_run_length": run_length_by_index.get(interval_index, 1),
                "next_phase_id": next_phase,
                "phase_change": "" if next_phase < 0 else int(next_phase != phase),
                "transition_filtered": int(bool(changed[index])),
            }
        )
        output.append(item)
    return output


def add_window_targets(window_rows: list[dict[str, str]], interval_label_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Window targets are derived by mapping end/target interval indices from metadata.
    by_interval = {int(row["interval_index"]): row for row in interval_label_rows}
    output: list[dict[str, object]] = []
    for row in window_rows:
        current = by_interval.get(int(row["end_interval_index"]))
        target = by_interval.get(int(row["target_interval_index"]))
        if current is None or target is None:
            continue
        phase = int(current["phase_id"])
        next_phase = int(target["phase_id"])
        item: dict[str, object] = dict(row)
        item.update(
            {
                "split": current.get("split", "train"),
                "phase_id": phase,
                "current_run_length": int(current.get("current_run_length", 1)),
                "next_phase_id": next_phase,
                "phase_change": int(next_phase != phase),
            }
        )
        output.append(item)
    return output


def label_dataset(dataset_dir: Path, output_dir: Path, config: dict[str, object], split_config: dict[str, object], seed: int) -> dict[str, object]:
    # Entry for phase labeling stage used by both CLI and Slurm pipelines.
    output_dir = ensure_dir(output_dir)
    manifest = read_json(dataset_dir / "feature_manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("feature_manifest.json must contain a JSON object")
    feature_columns = [str(item) for item in manifest.get("feature_columns", [])]
    rows = load_csv_rows(dataset_dir / "interval_features.csv")
    if str(config.get("split_policy", "run_grouped")) == "workload_holdout":
        split_by_run = workload_holdout_split(rows, str(config.get("holdout_workload", "")) or None)
    else:
        split_by_run = run_grouped_split(
            rows,
            float(split_config.get("train_fraction", 0.70)),
            float(split_config.get("val_fraction", 0.15)),
            seed,
        )
    labels, changed, model_payload = fit_predict_labels(rows, feature_columns, config, split_by_run, seed)
    horizon = int(manifest.get("prediction_horizon", 5))
    interval_label_rows = add_interval_targets(rows, labels, changed, split_by_run, horizon)
    window_label_rows = add_window_targets(load_csv_rows(dataset_dir / "window_metadata.csv"), interval_label_rows)
    write_csv_rows(output_dir / "interval_labels.csv", interval_label_rows)
    write_csv_rows(output_dir / "window_labels.csv", window_label_rows)
    label_payload = {
        "dataset_dir": str(dataset_dir),
        "method": str(config.get("method", "fgmm")),
        "split_policy": str(config.get("split_policy", "run_grouped")),
        "prediction_horizon": horizon,
        "phase_count": int(max(labels) + 1) if labels.size else 0,
        "model": model_payload,
    }
    write_json(output_dir / "label_model.json", label_payload)
    return label_payload
