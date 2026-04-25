"""Dataset cleaning and feature engineering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .constants import DEFAULT_WINSOR_LIMITS, METADATA_COLUMNS
from .events import compute_study_readiness, confident_family_names, generic_fallback_family_names, unsupported_family_names
from .io_utils import load_csv_rows, quantile_clip, safe_div, safe_float, write_csv_rows, write_json


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    if not rows:
        return columns
    for key in rows[0].keys():
        if key in METADATA_COLUMNS:
            continue
        if any(not math.isnan(safe_float(row.get(key, ""))) for row in rows):
            columns.append(key)
    return columns


def rows_to_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            matrix[row_index, column_index] = safe_float(row.get(column, ""))
    return matrix


def matrix_to_rows(rows: list[dict[str, str]], columns: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row_index, source in enumerate(rows):
        item: dict[str, object] = {key: source.get(key, "") for key in METADATA_COLUMNS}
        for column_index, column in enumerate(columns):
            value = matrix[row_index, column_index]
            item[column] = "" if math.isnan(value) else float(value)
        output.append(item)
    return output


def remove_constant_columns(matrix: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    kept_indices: list[int] = []
    removed: list[str] = []
    for index, column in enumerate(columns):
        values = matrix[:, index]
        clean = values[~np.isnan(values)]
        if clean.size == 0:
            removed.append(column)
            continue
        if np.nanmax(clean) - np.nanmin(clean) <= 1e-12:
            removed.append(column)
            continue
        kept_indices.append(index)
    filtered = matrix[:, kept_indices] if kept_indices else np.empty((matrix.shape[0], 0))
    kept_columns = [columns[index] for index in kept_indices]
    return filtered, kept_columns, removed


def fill_missing_with_median(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    filled = matrix.copy()
    medians: list[float] = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        clean = column[~np.isnan(column)]
        median = float(np.median(clean)) if clean.size else math.nan
        medians.append(median)
        mask = np.isnan(filled[:, index])
        filled[mask, index] = median
    return filled, medians


def derive_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    derived: list[dict[str, object]] = []
    for row in rows:
        item = {key: row.get(key, "") for key in METADATA_COLUMNS}
        inst = safe_float(row.get("counter__instructions_retired", ""))
        cycles = safe_float(row.get("counter__cycles", ""))
        branches = safe_float(row.get("counter__branch_instructions", ""))
        branch_miss = safe_float(row.get("counter__branch_mispredictions", ""))
        l2_miss = safe_float(row.get("counter__l2_misses", ""))
        llc_miss = safe_float(row.get("counter__llc_misses", ""))
        offcore = safe_float(row.get("counter__offcore_demand_data_reads", ""))
        fp = safe_float(row.get("counter__fp_arithmetic", ""))
        item["metric__cpi"] = safe_div(cycles, inst)
        item["metric__ipc"] = safe_div(inst, cycles)
        item["metric__branch_mpki"] = safe_div(branch_miss, inst, 1000.0)
        item["metric__branch_per_ki"] = safe_div(branches, inst, 1000.0)
        item["metric__l2_mpki"] = safe_div(l2_miss, inst, 1000.0)
        item["metric__llc_mpki"] = safe_div(llc_miss, inst, 1000.0)
        item["metric__offcore_read_mpki"] = safe_div(offcore, inst, 1000.0)
        item["metric__fp_per_ki"] = safe_div(fp, inst, 1000.0)
        derived.append(item)
    return derived


def normalize_counters(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        item = {key: row.get(key, "") for key in METADATA_COLUMNS}
        inst = safe_float(row.get("counter__instructions_retired", ""))
        duration_ms = safe_float(row.get("interval_duration_ms", ""))
        if math.isnan(duration_ms) or duration_ms <= 0:
            duration_ms = 1.0
        for key, value in row.items():
            if not key.startswith("counter__"):
                continue
            number = safe_float(value)
            if key in {"counter__instructions_retired", "counter__cycles"}:
                item[f"norm__{key[9:]}_per_ms"] = safe_div(number, duration_ms)
            elif "bandwidth" in key:
                item[f"norm__{key[9:]}_per_ms"] = safe_div(number, duration_ms)
            else:
                item[f"norm__{key[9:]}_per_ki"] = safe_div(number, inst, 1000.0)
        normalized.append(item)
    return normalized


def analysis_ready_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    kept: list[dict[str, object]] = []
    dropped_counts = {
        "non_positive_interval_duration_ms": 0,
        "non_positive_instructions_retired": 0,
        "non_positive_cycles": 0,
    }
    for row in rows:
        reasons = []
        interval_duration = safe_float(row.get("interval_duration_ms", ""))
        instructions = safe_float(row.get("counter__instructions_retired", ""))
        cycles = safe_float(row.get("counter__cycles", ""))
        if math.isnan(interval_duration) or interval_duration <= 0:
            reasons.append("non_positive_interval_duration_ms")
        if math.isnan(instructions) or instructions <= 0:
            reasons.append("non_positive_instructions_retired")
        if math.isnan(cycles) or cycles <= 0:
            reasons.append("non_positive_cycles")
        if reasons:
            for reason in reasons:
                dropped_counts[reason] += 1
            continue
        kept.append(row)
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "dropped_reasons": dropped_counts,
    }
    return kept, summary


def winsorize_rows(rows: list[dict[str, object]], limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS) -> list[dict[str, object]]:
    columns = [key for key in rows[0].keys() if key not in METADATA_COLUMNS] if rows else []
    matrix = rows_to_matrix([{k: str(v) for k, v in row.items()} for row in rows], columns)
    clipped = matrix.copy()
    for index in range(matrix.shape[1]):
        clipped[:, index] = quantile_clip(matrix[:, index], limits[0], limits[1])
    return matrix_to_rows([{k: str(v) for k, v in row.items()} for row in rows], columns, clipped)


def preprocess_interval_dataset(
    input_csv: Path,
    output_dir: Path,
    alias_map: dict[str, dict[str, object]] | None = None,
    platform_info: dict[str, object] | None = None,
) -> dict[str, object]:
    rows = load_csv_rows(input_csv)
    observed_counter_columns = [column for column in numeric_columns(rows) if column.startswith("counter__")]
    confident_families = set(confident_family_names(alias_map or {}))
    generic_fallback_families = set(generic_fallback_family_names(alias_map or {}))
    unsupported_families = set(unsupported_family_names(alias_map or {}))
    if alias_map is None:
        raw_counter_columns = observed_counter_columns
    else:
        raw_counter_columns = [
            column for column in observed_counter_columns if column.replace("counter__", "") in confident_families
        ]
    observed_confident_families = sorted(column.replace("counter__", "") for column in raw_counter_columns)
    raw_matrix = rows_to_matrix(rows, raw_counter_columns)
    filtered_matrix, filtered_columns, removed = remove_constant_columns(raw_matrix, raw_counter_columns)
    raw_sparse_rows = matrix_to_rows(rows, filtered_columns, filtered_matrix)
    missingness = {}
    for index, column in enumerate(filtered_columns):
        values = filtered_matrix[:, index]
        missingness[column] = float(np.mean(np.isnan(values))) if values.size else 1.0
    filled_matrix, medians = fill_missing_with_median(filtered_matrix)
    raw_clean_rows = matrix_to_rows(rows, filtered_columns, filled_matrix)
    raw_winsor_rows = winsorize_rows(raw_clean_rows)
    valid_sparse_rows, analysis_filter = analysis_ready_rows(raw_sparse_rows)
    valid_row_ids = {id(row) for row in valid_sparse_rows}
    filtered_analysis_rows = [clean_row for sparse_row, clean_row in zip(raw_sparse_rows, raw_clean_rows) if id(sparse_row) in valid_row_ids]
    normalized_rows = normalize_counters(filtered_analysis_rows)
    normalized_winsor_rows = winsorize_rows(normalized_rows)
    derived_rows = derive_metrics(filtered_analysis_rows)
    derived_winsor_rows = winsorize_rows(derived_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "raw_clean.csv", raw_clean_rows)
    write_csv_rows(output_dir / "raw_winsorized.csv", raw_winsor_rows)
    write_csv_rows(output_dir / "normalized_clean.csv", normalized_rows)
    write_csv_rows(output_dir / "normalized_winsorized.csv", normalized_winsor_rows)
    write_csv_rows(output_dir / "derived_clean.csv", derived_rows)
    write_csv_rows(output_dir / "derived_winsorized.csv", derived_winsor_rows)
    study_readiness = compute_study_readiness(platform_info or {}, alias_map or {}) if alias_map is not None else {}
    observed_uncore_columns = [
        column
        for column in observed_counter_columns
        if column in {"counter__memory_read_bandwidth", "counter__memory_write_bandwidth", "counter__total_memory_bandwidth"}
    ]

    summary = {
        "input_rows": len(rows),
        "raw_counter_columns_initial": observed_counter_columns,
        "raw_counter_columns_kept": filtered_columns,
        "confident_families": observed_confident_families,
        "confident_families_exposed": sorted(confident_families),
        "excluded_generic_fallback_columns": [
            column for column in observed_counter_columns if column.replace("counter__", "") in generic_fallback_families
        ],
        "excluded_unsupported_families": sorted(unsupported_families),
        "removed_constant_columns": removed,
        "observed_system_wide_uncore_columns": observed_uncore_columns,
        "missingness": missingness,
        "medians": {column: medians[index] for index, column in enumerate(filtered_columns)},
        "analysis_row_filter": analysis_filter,
        "study_ready_core": study_readiness.get("study_ready_core", False),
        "study_ready_uncore": study_readiness.get("study_ready_uncore", False),
        "study_ready": study_readiness.get("study_ready", False),
        "uncore_policy_blocked": study_readiness.get("uncore_policy_blocked", False),
        "missing_required_confident_families": study_readiness.get("missing_required_confident_families", []),
        "strict_hardware_study_reason": study_readiness.get("strict_hardware_study_reason", ""),
        "uncore_reason": study_readiness.get("uncore_reason", ""),
        "output_files": {
            "raw_clean": str(output_dir / "raw_clean.csv"),
            "raw_winsorized": str(output_dir / "raw_winsorized.csv"),
            "normalized_clean": str(output_dir / "normalized_clean.csv"),
            "normalized_winsorized": str(output_dir / "normalized_winsorized.csv"),
            "derived_clean": str(output_dir / "derived_clean.csv"),
            "derived_winsorized": str(output_dir / "derived_winsorized.csv"),
        },
    }
    write_json(output_dir / "preprocess_summary.json", summary)
    write_csv_rows(
        output_dir / "missingness_summary.csv",
        [{"column": column, "missing_fraction": value} for column, value in missingness.items()],
        ["column", "missing_fraction"],
    )
    return summary
