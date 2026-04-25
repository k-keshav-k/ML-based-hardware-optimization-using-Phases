"""Shared helpers for paper artifact analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


LOCAL_FEATURES = [
    "branch_miss_rate",
    "branch_instructions_per_ki",
    "l1d_loads_per_ki",
    "l1d_stores_per_ki",
    "load_store_ratio",
    "llc_references_per_ki",
    "llc_misses_per_ki",
    "llc_miss_rate",
    "offcore_reads_per_ki",
]

SHARED_FEATURES = [
    "shared_proxy_llc_miss_rate",
    "shared_proxy_offcore_reads_per_ki",
    "shared_imc_read_mib",
    "shared_imc_write_mib",
    "shared_imc_total_mib",
    "shared_imc_write_fraction",
]

FEATURE_GROUPS = {
    "branch": ["branch_miss_rate", "branch_instructions_per_ki"],
    "l1": ["l1d_loads_per_ki", "l1d_stores_per_ki", "load_store_ratio"],
    "llc_offcore": ["llc_references_per_ki", "llc_misses_per_ki", "llc_miss_rate", "offcore_reads_per_ki"],
    "shared": SHARED_FEATURES,
    "local": LOCAL_FEATURES,
    "all": LOCAL_FEATURES + SHARED_FEATURES,
}

FORBIDDEN_FEATURE_PARTS = ["cycle", "elapsed", "per_ms", "ipc", "cpi", "stall", "runtime", "duration"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def feature_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(columns)), np.nan)
    for row_index, row in enumerate(rows):
        for col_index, column in enumerate(columns):
            matrix[row_index, col_index] = safe_float(row.get(column, ""))
    return fill_nan_with_medians(matrix)


def fill_nan_with_medians(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col_index in range(filled.shape[1]):
        col = filled[:, col_index]
        clean = col[np.isfinite(col)]
        fill = float(np.median(clean)) if clean.size else 0.0
        col[~np.isfinite(col)] = fill
        filled[:, col_index] = col
    return filled


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (matrix - mean) / scale


def phase_resource_classes(interval_rows: list[dict[str, str]], window_rows: list[dict[str, str]]) -> dict[int, str]:
    """Map each unsupervised phase to a coarse resource class from centroid features."""
    feature_columns = [column for column in FEATURE_GROUPS["all"] if column in interval_rows[0]]
    by_interval = {int(row["interval_index"]): row for row in interval_rows}
    phase_vectors: dict[int, list[list[float]]] = {}
    for row in window_rows:
        interval = by_interval.get(int(row["end_interval_index"]))
        if interval is None:
            continue
        phase = int(row["phase_id"])
        phase_vectors.setdefault(phase, []).append([safe_float(interval.get(column, "")) for column in feature_columns])
    matrix_all = feature_matrix(interval_rows, feature_columns)
    medians = {column: float(np.median(matrix_all[:, index])) for index, column in enumerate(feature_columns)}
    classes: dict[int, str] = {}
    for phase, vectors in phase_vectors.items():
        centroid = np.asarray(vectors, dtype=float)
        if centroid.size == 0:
            classes[phase] = "balanced"
            continue
        centroid = fill_nan_with_medians(centroid).mean(axis=0)
        values = dict(zip(feature_columns, centroid))
        memory_score = 0
        branch_score = 0
        if values.get("shared_imc_total_mib", 0.0) > medians.get("shared_imc_total_mib", float("inf")):
            memory_score += 2
        if values.get("offcore_reads_per_ki", 0.0) > medians.get("offcore_reads_per_ki", float("inf")):
            memory_score += 1
        if values.get("llc_miss_rate", 0.0) > medians.get("llc_miss_rate", float("inf")):
            memory_score += 1
        if values.get("branch_miss_rate", 0.0) > medians.get("branch_miss_rate", float("inf")):
            branch_score += 2
        if values.get("branch_instructions_per_ki", 0.0) > medians.get("branch_instructions_per_ki", float("inf")):
            branch_score += 1
        if memory_score >= 3:
            classes[phase] = "memory"
        elif branch_score >= 2 and memory_score <= 1:
            classes[phase] = "branch"
        elif memory_score >= 2 and branch_score >= 2:
            classes[phase] = "mixed"
        else:
            classes[phase] = "balanced"
    return classes


def compatibility_conflict(class_a: str, class_b: str) -> bool:
    heavy = {"memory", "mixed"}
    return class_a in heavy and class_b in heavy


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    output = 0.0
    for count in counter.values():
        p = count / total
        output -= p * math.log2(p)
    return output
