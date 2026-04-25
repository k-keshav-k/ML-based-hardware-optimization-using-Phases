"""Feature engineering for PARSEC phase ML without timing-derived features."""

from __future__ import annotations

import math

import numpy as np

from hpc_phase_analysis.io_utils import safe_float

BASE_METADATA_COLUMNS = [
    "interval_index",
    "timestamp_ms",
    "suite",
    "workload",
    "run_id",
    "threads",
    "cpu_or_core_id",
    "experiment_set",
    "rep",
    "concurrent_group_id",
    "process_index",
    "process_count",
    "co_running_workloads",
    "collection_scope",
    "collection_unit_type",
    "physical_core_id",
    "shared_context_scope",
]

LOCAL_FEATURE_COLUMNS = [
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

SHARED_PROXY_COLUMNS = [
    "shared_proxy_llc_miss_rate",
    "shared_proxy_offcore_reads_per_ki",
]

SHARED_UNCORE_COLUMNS = [
    "shared_imc_read_mib",
    "shared_imc_write_mib",
    "shared_imc_total_mib",
    "shared_imc_write_fraction",
]

FORBIDDEN_COLUMN_PARTS = [
    "cycle",
    "per_ms",
    "interval_duration",
    "ipc",
    "cpi",
    "stall",
    "elapsed",
]


def safe_div(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator * scale


def collection_unit_type(row: dict[str, str]) -> str:
    collection_scope = str(row.get("collection_scope", "")).strip()
    if collection_scope == "system_wide_physical_core":
        return "physical_core_system_wide"
    if collection_scope == "system_wide_cpu":
        return "logical_cpu_system_wide"
    try:
        threads = int(str(row.get("threads", "0")).strip())
    except ValueError:
        threads = 0
    cpu_text = str(row.get("cpu_or_core_id", "")).strip()
    cpu_count = len([item for item in cpu_text.split(",") if item.strip()])
    if threads == 1 and cpu_count <= 1:
        return "single_cpu_task_local"
    return "affinity_group_task_local"


def finite_sum(*values: float) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) if clean else math.nan


def derive_feature_values(row: dict[str, str]) -> dict[str, float]:
    inst = safe_float(row.get("counter__instructions_retired", ""))
    branches = safe_float(row.get("counter__branch_instructions", ""))
    branch_miss = safe_float(row.get("counter__branch_mispredictions", ""))
    loads = safe_float(row.get("counter__l1d_loads", ""))
    stores = safe_float(row.get("counter__l1d_stores", ""))
    llc_refs = safe_float(row.get("counter__llc_references", ""))
    llc_miss = safe_float(row.get("counter__llc_misses", ""))
    offcore = safe_float(row.get("counter__offcore_demand_data_reads", ""))
    imc_read = safe_float(row.get("counter__memory_read_bandwidth", ""))
    imc_write = safe_float(row.get("counter__memory_write_bandwidth", ""))
    imc_total = safe_float(row.get("counter__total_memory_bandwidth", ""))
    if math.isnan(imc_total):
        imc_total = finite_sum(imc_read, imc_write)
    llc_miss_rate = safe_div(llc_miss, llc_refs)
    offcore_per_ki = safe_div(offcore, inst, 1000.0)
    return {
        "branch_miss_rate": safe_div(branch_miss, branches),
        "branch_instructions_per_ki": safe_div(branches, inst, 1000.0),
        "l1d_loads_per_ki": safe_div(loads, inst, 1000.0),
        "l1d_stores_per_ki": safe_div(stores, inst, 1000.0),
        "load_store_ratio": safe_div(loads, stores),
        "llc_references_per_ki": safe_div(llc_refs, inst, 1000.0),
        "llc_misses_per_ki": safe_div(llc_miss, inst, 1000.0),
        "llc_miss_rate": llc_miss_rate,
        "offcore_reads_per_ki": offcore_per_ki,
        "shared_proxy_llc_miss_rate": llc_miss_rate,
        "shared_proxy_offcore_reads_per_ki": offcore_per_ki,
        "shared_imc_read_mib": imc_read,
        "shared_imc_write_mib": imc_write,
        "shared_imc_total_mib": imc_total,
        "shared_imc_write_fraction": safe_div(imc_write, imc_total),
    }


def shared_context_scope(values: dict[str, float]) -> str:
    if any(not math.isnan(values[column]) for column in SHARED_UNCORE_COLUMNS):
        return "system_wide_uncore"
    return "task_local_proxy"


def feature_row(source: dict[str, str], interval_index: int, feature_columns: list[str] | None = None) -> dict[str, object]:
    values = derive_feature_values(source)
    selected = feature_columns or list(values.keys())
    item: dict[str, object] = {
        "interval_index": interval_index,
        "timestamp_ms": source.get("timestamp_ms", ""),
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
        "collection_scope": source.get("collection_scope", "task_local") or "task_local",
        "collection_unit_type": collection_unit_type(source),
        "physical_core_id": source.get("physical_core_id", ""),
        "shared_context_scope": shared_context_scope(values),
    }
    for column in selected:
        value = values.get(column, math.nan)
        item[column] = "" if math.isnan(value) else float(value)
    return item


def available_feature_columns(rows: list[dict[str, object]]) -> list[str]:
    candidates = LOCAL_FEATURE_COLUMNS + SHARED_PROXY_COLUMNS + SHARED_UNCORE_COLUMNS
    kept: list[str] = []
    for column in candidates:
        if any(not math.isnan(safe_float(row.get(column, ""))) for row in rows):
            kept.append(column)
    return kept


def assert_no_forbidden_features(columns: list[str]) -> None:
    bad = [column for column in columns if any(part in column.lower() for part in FORBIDDEN_COLUMN_PARTS)]
    if bad:
        raise ValueError("Forbidden timing/cycle/stall features detected: " + ", ".join(bad))


def matrix_from_rows(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            matrix[row_index, column_index] = safe_float(row.get(column, ""))
    return matrix


def finite_column_medians(matrix: np.ndarray) -> np.ndarray:
    medians = np.zeros(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        values = matrix[:, index]
        clean = values[np.isfinite(values)]
        medians[index] = float(np.median(clean)) if clean.size else 0.0
    return medians


def fill_nan_with_medians(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for index in range(filled.shape[1]):
        mask = ~np.isfinite(filled[:, index])
        filled[mask, index] = medians[index]
    return filled
