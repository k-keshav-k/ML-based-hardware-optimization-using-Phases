"""Build interval features and sliding windows for PARSEC phase ML."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, safe_float, write_csv_rows, write_json

from .features import (
    BASE_METADATA_COLUMNS,
    available_feature_columns,
    assert_no_forbidden_features,
    feature_row,
    matrix_from_rows,
)


def parsec_interval_rows(input_csv: Path) -> list[dict[str, str]]:
    rows = load_csv_rows(input_csv)
    return [row for row in rows if row.get("suite", "") == "parsec"]


def build_interval_features(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("run_id", ""),
            row.get("cpu_or_core_id", ""),
            safe_float(row.get("timestamp_ms", "")),
        ),
    )
    broad = [feature_row(row, index) for index, row in enumerate(sorted_rows)]
    feature_columns = available_feature_columns(broad)
    assert_no_forbidden_features(feature_columns)
    narrowed = [feature_row(row, index, feature_columns) for index, row in enumerate(sorted_rows)]
    return narrowed, feature_columns


def group_interval_indices(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (
            str(row.get("run_id", "")),
            str(row.get("cpu_or_core_id", "")),
            str(row.get("collection_unit_type", "")),
        )
        groups[key].append(index)
    for key, indices in groups.items():
        indices.sort(key=lambda index: safe_float(rows[index].get("timestamp_ms", "")))
    return groups


def build_windows(
    rows: list[dict[str, object]],
    feature_columns: list[str],
    sequence_length: int,
    stride: int,
    prediction_horizon: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    matrix = matrix_from_rows(rows, feature_columns)
    windows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    current_indices: list[int] = []
    target_indices: list[int] = []
    for indices in group_interval_indices(rows).values():
        limit = len(indices) - sequence_length - prediction_horizon + 1
        if limit <= 0:
            continue
        for offset in range(0, limit, stride):
            window_indices = indices[offset : offset + sequence_length]
            current_index = window_indices[-1]
            target_index = indices[offset + sequence_length - 1 + prediction_horizon]
            source = rows[current_index]
            target = rows[target_index]
            window_id = len(windows)
            windows.append(matrix[window_indices, :])
            current_indices.append(int(source["interval_index"]))
            target_indices.append(int(target["interval_index"]))
            metadata.append(
                {
                    "window_id": window_id,
                    "suite": source.get("suite", ""),
                    "workload": source.get("workload", ""),
                    "run_id": source.get("run_id", ""),
                    "threads": source.get("threads", ""),
                    "cpu_or_core_id": source.get("cpu_or_core_id", ""),
                    "experiment_set": source.get("experiment_set", ""),
                    "rep": source.get("rep", ""),
                    "concurrent_group_id": source.get("concurrent_group_id", ""),
                    "process_index": source.get("process_index", ""),
                    "process_count": source.get("process_count", ""),
                    "co_running_workloads": source.get("co_running_workloads", ""),
                    "collection_scope": source.get("collection_scope", ""),
                    "collection_unit_type": source.get("collection_unit_type", ""),
                    "physical_core_id": source.get("physical_core_id", ""),
                    "shared_context_scope": source.get("shared_context_scope", ""),
                    "start_interval_index": int(rows[window_indices[0]]["interval_index"]),
                    "end_interval_index": int(source["interval_index"]),
                    "target_interval_index": int(target["interval_index"]),
                    "start_timestamp_ms": rows[window_indices[0]].get("timestamp_ms", ""),
                    "end_timestamp_ms": source.get("timestamp_ms", ""),
                    "target_timestamp_ms": target.get("timestamp_ms", ""),
                }
            )
    if windows:
        x = np.stack(windows).astype(float)
    else:
        x = np.empty((0, sequence_length, len(feature_columns)), dtype=float)
    return x, metadata, np.asarray(current_indices, dtype=int), np.asarray(target_indices, dtype=int)


def build_dataset(
    input_csv: Path,
    output_dir: Path,
    sequence_length: int,
    stride: int,
    prediction_horizon: int,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    parsec_rows = parsec_interval_rows(input_csv)
    interval_rows, feature_columns = build_interval_features(parsec_rows)
    collection_scopes = sorted({str(row.get("collection_scope", "")) for row in interval_rows if row.get("collection_scope", "")})
    shared_context_scopes = sorted({str(row.get("shared_context_scope", "")) for row in interval_rows if row.get("shared_context_scope", "")})
    x, window_metadata, current_indices, target_indices = build_windows(
        interval_rows,
        feature_columns,
        sequence_length,
        stride,
        prediction_horizon,
    )
    interval_columns = BASE_METADATA_COLUMNS + feature_columns
    write_csv_rows(output_dir / "interval_features.csv", interval_rows, interval_columns)
    write_csv_rows(output_dir / "window_metadata.csv", window_metadata)
    np.savez_compressed(
        output_dir / "window_tensors.npz",
        X=x,
        current_interval_indices=current_indices,
        target_interval_indices=target_indices,
        feature_columns=np.asarray(feature_columns),
    )
    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "suite": "parsec",
        "collection_scope": "+".join(collection_scopes) if collection_scopes else "",
        "shared_context_scope": "+".join(shared_context_scopes) if shared_context_scopes else "",
        "interval_rows": len(interval_rows),
        "windows": len(window_metadata),
        "sequence_length": sequence_length,
        "stride": stride,
        "prediction_horizon": prediction_horizon,
        "feature_columns": feature_columns,
        "forbidden_feature_policy": "cycles/time/per-ms/CPI/IPC/stall-derived features excluded",
        "true_per_core_ready": "system_wide_physical_core" in collection_scopes or "system_wide_cpu" in collection_scopes,
    }
    write_json(output_dir / "feature_manifest.json", manifest)
    return manifest
