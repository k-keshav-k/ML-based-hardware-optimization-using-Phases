"""Counter sequence construction with train-split clustered phase states.

This stage converts raw counter intervals into per-counter value streams plus
discrete states used for ablation scoring.
"""

from __future__ import annotations

import math
from shutil import copyfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, safe_float, write_csv_rows, write_json

from .families import FAMILY_COUNTERS, FORBIDDEN_PARTS, derive_feature_values, derived_columns_for_family, family_counter_availability
from .splits import ExperimentSplit, build_experiment_splits

SelectedCounterMap = dict[tuple[str, str], dict[str, list[str]]]


def parsec_rows(input_csv: Path) -> list[dict[str, str]]:
    """Load only PARSEC rows for the family-LM flow."""

    return [row for row in load_csv_rows(input_csv) if row.get("suite", "") == "parsec"]


def add_core_id(row: dict[str, str]) -> str:
    """Return a normalized core identifier field expected by family artifacts."""

    return str(row.get("cpu_or_core_id", ""))


def interval_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[int]]:
    """Build run/core streams for temporal horizon sequencing."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row.get("run_id", "")), add_core_id(row))
        grouped[key].append(index)
    for key in grouped:
        grouped[key].sort(key=lambda i: safe_float(rows[i].get("timestamp_ms", "")))
    return grouped


def build_derived_matrix(rows: list[dict[str, str]]) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    """Compute derived counter features once and expose column-wise arrays."""

    derived_rows = [derive_feature_values(row) for row in rows]
    by_column: dict[str, np.ndarray] = {}
    if not derived_rows:
        return derived_rows, by_column
    for column in derived_rows[0].keys():
        by_column[column] = np.asarray([safe_float(item.get(column, "")) for item in derived_rows], dtype=float)
    return derived_rows, by_column


def _rowwise_nanmean(stacked: np.ndarray, row_count: int) -> np.ndarray:
    """Return a NaN-safe row-wise mean vector with stable output length."""

    if stacked.size == 0:
        return np.full(row_count, np.nan, dtype=float)
    finite = np.isfinite(stacked)
    counts = finite.sum(axis=1)
    sums = np.nansum(stacked, axis=1)
    values = np.full(stacked.shape[0], np.nan, dtype=float)
    mask = counts > 0
    values[mask] = sums[mask] / counts[mask]
    return values


def _selected_counter_score(rows: list[dict[str, str]], counters: list[str]) -> np.ndarray:
    """Build one usage score from selected raw counters only.

    We compress values with ``log1p`` so mixed-scale counters can still be
    averaged without one very large magnitude counter dominating.
    """

    stacked = np.full((len(rows), len(counters)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, counter in enumerate(counters):
            value = safe_float(row.get(counter, ""))
            if np.isfinite(value) and value >= 0.0:
                stacked[row_index, col_index] = math.log1p(value)
    return _rowwise_nanmean(stacked, len(rows))


def family_usage_scores(rows: list[dict[str, str]], selected_counters_by_family: dict[str, list[str]] | None = None) -> dict[str, np.ndarray]:
    """Aggregate derived features into one interpretable score per family.

    Default behavior: each family score is the nanmean of its derived columns.
    If ``selected_counters_by_family`` is provided for a family, we instead use
    only those selected raw counters to build the score.
    """

    _derived_rows, by_column = build_derived_matrix(rows)
    present_counter_columns = {key for row in rows for key in row.keys() if key.startswith("counter__")}
    output: dict[str, np.ndarray] = {}
    for family in FAMILY_COUNTERS:
        selected = selected_counters_by_family.get(family, []) if selected_counters_by_family else []
        selected_present = [counter for counter in selected if counter in present_counter_columns]
        if selected:
            if not selected_present:
                output[family] = np.full(len(rows), np.nan, dtype=float)
                continue
            output[family] = _selected_counter_score(rows, selected_present)
            continue
        columns = derived_columns_for_family(family)
        present = [column for column in columns if column in by_column]
        if not present:
            output[family] = np.full(len(rows), np.nan, dtype=float)
            continue
        stacked = np.vstack([by_column[column] for column in present]).T
        output[family] = _rowwise_nanmean(stacked, len(rows))
    return output


def _parse_counter_set(counter_set: str) -> list[str]:
    """Split and clean a CSV-style counter set string."""

    return [item.strip() for item in str(counter_set).split(",") if item.strip()]


def _parse_family_counter_map(text: str) -> dict[str, str]:
    """Parse ``family_counter_map`` serialized as ``family:counter;...``."""

    output: dict[str, str] = {}
    for item in str(text).split(";"):
        token = item.strip()
        if not token or ":" not in token:
            continue
        family, counter = token.split(":", 1)
        family_name = family.strip()
        counter_name = counter.strip()
        if family_name and counter_name:
            output[family_name] = counter_name
    return output


def load_ablation_global_one_per_family(ablation_results: Path) -> SelectedCounterMap:
    """Load best global exhaustive one-per-family selections.

    Returns mappings keyed by ``(experiment, scope)`` where each family maps to
    exactly one selected counter (stored as a one-item list for compatibility
    with ``SelectedCounterMap``).
    """

    rows = load_csv_rows(ablation_results)
    output: SelectedCounterMap = {}
    best_meta: dict[tuple[str, str], tuple[float, int]] = {}
    for row in rows:
        if str(row.get("family", "")).strip() != "__global__":
            continue
        if str(row.get("candidate_type", "")).strip() != "one_per_family_exhaustive":
            continue
        mapping = _parse_family_counter_map(str(row.get("family_counter_map", "")))
        if not mapping:
            continue
        experiment = str(row.get("experiment", "")).strip()
        scope = str(row.get("scope", "")).strip()
        score = safe_float(row.get("validation_score", ""))
        if not np.isfinite(score):
            score = -math.inf
        selected_flag = int(row.get("selected", "0") or 0)
        key = (experiment, scope)
        previous = best_meta.get(key, (-math.inf, -1))
        if (score, selected_flag) < previous:
            continue
        best_meta[key] = (score, selected_flag)
        output[key] = {family: [counter] for family, counter in mapping.items()}
    return output


def load_ablation_selected_counters(ablation_results: Path) -> SelectedCounterMap:
    """Load best per-family counter sets from ablation results CSV.

    For each ``(experiment, scope, family)`` key, we pick the best singleton row.
    This fallback preserves the one-counter-per-family contract if a global row
    is unavailable.
    """

    rows = load_csv_rows(ablation_results)
    selected: SelectedCounterMap = {}
    selected_meta: dict[tuple[str, str, str], tuple[float, int]] = {}
    for row in rows:
        family = str(row.get("family", "")).strip()
        if family in {"", "__global__"}:
            continue
        if str(row.get("candidate_type", "")).strip() != "singleton":
            continue
        counters = _parse_counter_set(str(row.get("counter_set", "")))
        if len(counters) != 1:
            continue
        experiment = str(row.get("experiment", "")).strip()
        scope = str(row.get("scope", "")).strip()
        score = safe_float(row.get("validation_score", ""))
        if not np.isfinite(score):
            score = -math.inf
        selected_flag = int(row.get("selected", "0") or 0)
        key = (experiment, scope, family)
        previous = selected_meta.get(key, (-math.inf, -1))
        if (score, selected_flag) < previous:
            continue
        selected_meta[key] = (score, selected_flag)
        selected.setdefault((experiment, scope), {})[family] = counters
    return selected


def resolve_selected_counters(selections: SelectedCounterMap, experiment: str, scope: str) -> dict[str, list[str]]:
    """Resolve selected counter sets with exact match first then fallbacks."""

    for key in [
        (experiment, scope),
        ("", scope),
        (experiment, ""),
        ("", ""),
    ]:
        if key in selections:
            return dict(selections[key])
    return {}


def _tertiles(values: np.ndarray) -> tuple[float, float]:
    """Compute low/high tertile cut points from finite values."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.quantile(finite, 1.0 / 3.0)), float(np.quantile(finite, 2.0 / 3.0))


def _bucketize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map usage scores to 3-class state IDs: 0=low, 1=moderate, 2=high."""

    output = np.full(values.shape[0], -1, dtype=int)
    finite = np.isfinite(values)
    if not np.isfinite(low) or not np.isfinite(high):
        return output
    output[finite & (values < low)] = 0
    output[finite & (values >= low) & (values < high)] = 1
    output[finite & (values >= high)] = 2
    return output


def thresholds_for_family(
    family_values: np.ndarray,
    rows: list[dict[str, str]],
    split_by_run: dict[str, str],
    threshold_mode: str,
) -> dict[str, tuple[float, float]]:
    """Fit train-only thresholds globally and/or per workload."""

    if threshold_mode not in {"global", "per_workload", "both"}:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
    output: dict[str, tuple[float, float]] = {}
    train_mask = np.asarray([split_by_run.get(str(row.get("run_id", "")), "train") == "train" for row in rows], dtype=bool)
    if threshold_mode in {"global", "both"}:
        output["global"] = _tertiles(family_values[train_mask])
    if threshold_mode in {"per_workload", "both"}:
        workloads = sorted({str(row.get("workload", "")) for row in rows})
        for workload in workloads:
            mask = train_mask & np.asarray([str(row.get("workload", "")) == workload for row in rows], dtype=bool)
            output[f"workload::{workload}"] = _tertiles(family_values[mask])
    return output


def states_for_scope(
    family_values: np.ndarray,
    rows: list[dict[str, str]],
    scope: str,
    thresholds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Apply one threshold scope to all rows."""

    if scope == "global":
        low, high = thresholds.get("global", (math.nan, math.nan))
        return _bucketize(family_values, low, high)
    if scope == "per_workload":
        output = np.full(family_values.shape[0], -1, dtype=int)
        for workload in sorted({str(row.get("workload", "")) for row in rows}):
            low, high = thresholds.get(f"workload::{workload}", (math.nan, math.nan))
            mask = np.asarray([str(row.get("workload", "")) == workload for row in rows], dtype=bool)
            output[mask] = _bucketize(family_values[mask], low, high)
        return output
    raise ValueError(f"Unsupported scope: {scope}")


def _future_state_at(stream_states: list[int], pos: int, horizon: int) -> int:
    target_pos = pos + horizon
    if target_pos >= len(stream_states):
        return -1
    return int(stream_states[target_pos])


def build_counter_state_rows(
    rows: list[dict[str, str]],
    states: np.ndarray,
    split_by_run: dict[str, str],
    horizon: int,
) -> list[dict[str, object]]:
    """Build wide current/future state rows for one family/scope."""

    grouped = interval_groups(rows)
    output: list[dict[str, object]] = []
    for _, indices in grouped.items():
        stream_states = [int(states[index]) for index in indices]
        for local_pos, row_index in enumerate(indices):
            current = stream_states[local_pos]
            item: dict[str, object] = {
                "family_state": current,
                "split": split_by_run.get(str(rows[row_index].get("run_id", "")), "train"),
                "workload": rows[row_index].get("workload", ""),
                "run_id": rows[row_index].get("run_id", ""),
                "core_id": add_core_id(rows[row_index]),
                "threads": rows[row_index].get("threads", ""),
                "process_index": rows[row_index].get("process_index", ""),
                "process_count": rows[row_index].get("process_count", ""),
                "concurrent_group_id": rows[row_index].get("concurrent_group_id", ""),
                "rep": rows[row_index].get("rep", ""),
                "experiment_set": rows[row_index].get("experiment_set", ""),
                "requested_input_size": rows[row_index].get("requested_input_size", ""),
                "co_running_workloads": rows[row_index].get("co_running_workloads", ""),
                "collection_scope": rows[row_index].get("collection_scope", ""),
                "row_index": row_index,
                "timestamp_ms": rows[row_index].get("timestamp_ms", ""),
            }
            futures: list[int] = []
            for step in range(1, horizon + 1):
                future = _future_state_at(stream_states, local_pos, step)
                item[f"future_state_{step}"] = future
                futures.append(future)
            valid_changes = [step for step, future in enumerate(futures, start=1) if future >= 0 and current >= 0 and future != current]
            item["will_change_within_horizon"] = 1 if valid_changes else 0
            item["time_to_change"] = valid_changes[0] if valid_changes else -1
            output.append(item)
    return output


def _future_values_by_row(rows: list[dict[str, str]], values: np.ndarray, horizon: int) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for indices in interval_groups(rows).values():
        for local_pos, row_index in enumerate(indices):
            for step in range(1, horizon + 1):
                target_pos = local_pos + step
                output[(row_index, step)] = values[indices[target_pos]] if target_pos < len(indices) else math.nan
    return output


def attach_counter_values(
    label_rows: list[dict[str, object]],
    rows: list[dict[str, str]],
    counter: str,
    horizon: int,
) -> None:
    values = counter_value_series(rows, counter)
    future_values = _future_values_by_row(rows, values, horizon)
    for item in label_rows:
        row_index = int(item.get("row_index", -1))
        current_value = values[row_index] if 0 <= row_index < values.shape[0] else math.nan
        item["selected_counter"] = counter
        item["counter_value"] = "" if not np.isfinite(current_value) else current_value
        for step in range(1, horizon + 1):
            future_value = future_values.get((row_index, step), math.nan)
            item[f"future_counter_value_{step}"] = "" if not np.isfinite(future_value) else future_value


def _scope_output_dir(base_dir: Path, scope: str) -> Path:
    return ensure_dir(base_dir / f"threshold_{scope}")


def _counter_sequence_path(scope_dir: Path, counter: str) -> Path:
    return scope_dir / "counter_value_sequences" / f"counter_sequence_{counter}.csv"


def counter_value_series(rows: list[dict[str, str]], counter: str) -> np.ndarray:
    """Return one finite-or-NaN value array for a raw counter column."""

    values = np.asarray([safe_float(row.get(counter, "")) for row in rows], dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def safe_counter_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return safe raw counter columns used by the offline phase teacher."""

    columns = sorted({key for row in rows for key in row.keys() if key.startswith("counter__")})
    return [column for column in columns if not any(part in column.lower() for part in FORBIDDEN_PARTS)]


def _safe_counter_matrix(rows: list[dict[str, str]], counters: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(counters)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, counter in enumerate(counters):
            value = safe_float(row.get(counter, ""))
            if np.isfinite(value):
                matrix[row_index, col_index] = math.log1p(value) if value >= 0.0 else value
    return matrix


def _fill_and_scale(matrix: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    filled = matrix.copy()
    if filled.size == 0:
        return filled, {"median": [], "mean": [], "std": []}
    medians: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    for col in range(filled.shape[1]):
        train_values = filled[train_mask, col]
        clean = train_values[np.isfinite(train_values)]
        median = float(np.median(clean)) if clean.size else 0.0
        missing = ~np.isfinite(filled[:, col])
        filled[missing, col] = median
        clean_filled = filled[train_mask, col]
        mean = float(np.mean(clean_filled)) if clean_filled.size else 0.0
        std = float(np.std(clean_filled)) if clean_filled.size else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        filled[:, col] = (filled[:, col] - mean) / std
        medians.append(median)
        means.append(mean)
        stds.append(std)
    return filled, {"median": medians, "mean": means, "std": stds}


def _kmeans(train_x: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_x.shape[0] == 0:
        return np.empty((0, train_x.shape[1]), dtype=float), np.empty(0, dtype=int)
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
            labels = new_labels
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return centroids, labels


def _nearest_centroid(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if centroids.shape[0] == 0:
        return np.full(x.shape[0], -1, dtype=int)
    distances = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


def _pressure_feature_indices(counters: list[str]) -> list[int]:
    output: list[int] = []
    for index, counter in enumerate(counters):
        name = counter.lower()
        if "llc_misses" in name or ("memory" in name and "bandwidth" in name) or "branch_mispredictions" in name or "stall" in name:
            output.append(index)
    return output


def _cluster_one_scope(
    rows: list[dict[str, str]],
    counters: list[str],
    split_by_run: dict[str, str],
    row_mask: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix = _safe_counter_matrix(rows, counters)
    train_mask = row_mask & np.asarray([split_by_run.get(str(row.get("run_id", "")), "train") == "train" for row in rows], dtype=bool)
    if not np.any(train_mask):
        train_mask = row_mask.copy()
    if not counters or not np.any(row_mask):
        return np.full(len(rows), -1, dtype=int), {"rows": int(np.sum(row_mask)), "train_rows": int(np.sum(train_mask)), "clusters": 0}
    scaled, scaler = _fill_and_scale(matrix, train_mask)
    centroids, _ = _kmeans(scaled[train_mask], n_clusters, seed)
    raw_labels = _nearest_centroid(scaled, centroids)
    raw_labels[~row_mask] = -1

    pressure_indices = _pressure_feature_indices(counters)
    if pressure_indices:
        pressure = np.sum(scaled[:, pressure_indices], axis=1)
    else:
        pressure = np.linalg.norm(scaled, axis=1)
    cluster_pressure: dict[int, float] = {}
    cluster_sizes: dict[int, int] = {}
    for cluster in range(centroids.shape[0]):
        members = train_mask & (raw_labels == cluster)
        cluster_sizes[cluster] = int(np.sum(raw_labels[row_mask] == cluster))
        cluster_pressure[cluster] = float(np.mean(pressure[members])) if np.any(members) else math.inf
    ordered_clusters = sorted(range(centroids.shape[0]), key=lambda item: cluster_pressure.get(item, math.inf))
    remap = {cluster: rank for rank, cluster in enumerate(ordered_clusters)}
    states = np.full(len(rows), -1, dtype=int)
    for cluster, state in remap.items():
        states[row_mask & (raw_labels == cluster)] = state

    train_distances = np.sqrt(np.sum((scaled[train_mask] - centroids[raw_labels[train_mask]]) ** 2, axis=1))
    centroid_distances = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(centroids.shape[0])
        for j in range(i + 1, centroids.shape[0])
    ]
    summary = {
        "rows": int(np.sum(row_mask)),
        "train_rows": int(np.sum(train_mask)),
        "clusters": int(centroids.shape[0]),
        "counter_count": len(counters),
        "pressure_counters": ",".join(counters[index] for index in pressure_indices),
        "state_order": ";".join(f"{remap[cluster]}:{cluster_pressure[cluster]:.6g}" for cluster in ordered_clusters),
        "cluster_sizes": ";".join(f"{remap[cluster]}:{cluster_sizes[cluster]}" for cluster in ordered_clusters),
        "mean_train_distance": float(np.mean(train_distances)) if train_distances.size else 0.0,
        "min_centroid_distance": min(centroid_distances) if centroid_distances else 0.0,
        "centroids": centroids.tolist(),
        "scaler": scaler,
    }
    return states, summary


def clustered_phase_states(
    rows: list[dict[str, str]],
    split_by_run: dict[str, str],
    scope: str,
    n_clusters: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Fit train-only k-means phase labels and assign every interval."""

    counters = safe_counter_columns(rows)
    all_mask = np.ones(len(rows), dtype=bool)
    if scope == "global":
        states, summary = _cluster_one_scope(rows, counters, split_by_run, all_mask, n_clusters, seed)
        summary.update({"scope": scope, "workload": ""})
        return states, [summary]
    if scope == "per_workload":
        output = np.full(len(rows), -1, dtype=int)
        summaries: list[dict[str, object]] = []
        for workload in sorted({str(row.get("workload", "")) for row in rows}):
            mask = np.asarray([str(row.get("workload", "")) == workload for row in rows], dtype=bool)
            states, summary = _cluster_one_scope(rows, counters, split_by_run, mask, n_clusters, seed)
            output[mask] = states[mask]
            summary.update({"scope": scope, "workload": workload})
            summaries.append(summary)
        return output, summaries
    raise ValueError(f"Unsupported scope: {scope}")


def build_raw_counter_sequence_rows(
    rows: list[dict[str, str]],
    split: ExperimentSplit,
    scope: str,
    horizon: int,
    output_dir: Path,
    phase_states: np.ndarray,
) -> int:
    """Write per-counter value sequence datasets for LM-style counter prediction."""

    scope_dir = _scope_output_dir(output_dir, scope)
    counter_dir = ensure_dir(scope_dir / "counter_value_sequences")
    count = 0
    for family, counters in FAMILY_COUNTERS.items():
        for counter in counters:
            values = counter_value_series(rows, counter)
            label_rows = build_counter_state_rows(rows, phase_states, split.split_by_run, horizon)
            future_values = _future_values_by_row(rows, values, horizon)
            for item in label_rows:
                row_index = int(item.get("row_index", -1))
                item["counter_name"] = counter
                item["family"] = family
                item["phase_label_source"] = "train_split_kmeans_full_safe_counters"
                current_value = values[row_index] if 0 <= row_index < values.shape[0] else math.nan
                item["counter_value"] = "" if not np.isfinite(current_value) else current_value
                for step in range(1, horizon + 1):
                    future_value = future_values.get((row_index, step), math.nan)
                    item[f"future_counter_value_{step}"] = "" if not np.isfinite(future_value) else future_value
            write_csv_rows(counter_dir / f"counter_sequence_{counter}.csv", label_rows)
            count += 1
    return count


def build_counter_sequences_for_split(
    rows: list[dict[str, str]],
    split: ExperimentSplit,
    output_dir: Path,
    horizon: int,
    threshold_mode: str,
    seed: int,
    selected_counters_by_scope: dict[str, dict[str, list[str]]] | None = None,
    write_family_sequences: bool = True,
) -> dict[str, object]:
    """Generate counter sequence artifacts for all families for one experiment split."""

    output_dir = ensure_dir(output_dir)
    availability = family_counter_availability(rows)
    scopes = ["global", "per_workload"] if threshold_mode == "both" else [threshold_mode]
    summary_rows: list[dict[str, object]] = []
    counter_dataset_rows: list[dict[str, object]] = []
    cluster_summary_rows: list[dict[str, object]] = []
    for scope in scopes:
        phase_states, cluster_summaries = clustered_phase_states(rows, split.split_by_run, scope, 3, seed)
        cluster_summary_rows.extend(cluster_summaries)
        counter_files_written = build_raw_counter_sequence_rows(rows, split, scope, horizon, output_dir, phase_states)
        counter_dataset_rows.append(
            {
                "experiment": split.name,
                "mode": split.mode,
                "scope": scope,
                "counter_files_written": counter_files_written,
                "phase_label_source": "train_split_kmeans_full_safe_counters",
            }
        )
        scope_dir = _scope_output_dir(output_dir, scope)
        if not write_family_sequences:
            for family in FAMILY_COUNTERS:
                stale_path = scope_dir / f"counter_sequence_{family}.csv"
                if stale_path.exists():
                    stale_path.unlink()
            continue
        selected_for_scope = selected_counters_by_scope.get(scope, {}) if selected_counters_by_scope else {}
        for family in FAMILY_COUNTERS:
            selected_counter_set = selected_for_scope.get(family, [])
            family_path = scope_dir / f"counter_sequence_{family}.csv"
            source = "missing_single_ablation_selection"
            label_rows: list[dict[str, str]] = []
            low = math.nan
            high = math.nan
            if len(selected_counter_set) == 1:
                selected_counter = selected_counter_set[0]
                source_path = _counter_sequence_path(scope_dir, selected_counter)
                if source_path.exists():
                    copyfile(source_path, family_path)
                    label_rows = load_csv_rows(family_path)
                    if scope == "global":
                        values = counter_value_series(rows, selected_counter)
                        low, high = thresholds_for_family(values, rows, split.split_by_run, scope).get("global", (math.nan, math.nan))
                    source = "ablation_selected"
                else:
                    source = "selected_counter_sequence_missing"
                    if family_path.exists():
                        family_path.unlink()
            elif family_path.exists():
                family_path.unlink()
            summary_rows.append(
                {
                    "experiment": split.name,
                    "mode": split.mode,
                    "family": family,
                    "scope": scope,
                    "rows": len(label_rows),
                    "available_counters": ",".join(availability.get(family, [])),
                    "available": int(bool(availability.get(family, []))),
                    "selected_counters": ",".join(selected_counter_set),
                    "sequence_source": source,
                    "phase_label_source": "train_split_kmeans_full_safe_counters",
                    "counter_sequence_files_written": counter_files_written,
                    "global_low_threshold": "" if not np.isfinite(low) else low,
                    "global_high_threshold": "" if not np.isfinite(high) else high,
                }
            )
    write_csv_rows(output_dir / "counter_sequence_summary.csv", summary_rows)
    write_csv_rows(output_dir / "counter_sequence_file_summary.csv", counter_dataset_rows)
    write_csv_rows(
        output_dir / "phase_cluster_summary.csv",
        [{key: value for key, value in row.items() if key not in {"centroids", "scaler"}} for row in cluster_summary_rows],
    )
    write_json(
        output_dir / "counter_sequence_manifest.json",
        {
            "experiment": split.name,
            "mode": split.mode,
            "threshold_mode": threshold_mode,
            "phase_label_method": "train_split_kmeans_full_safe_counters",
            "phase_clusters": 3,
            "horizon": horizon,
            "rows": len(rows),
            "family_sequences_written": write_family_sequences,
            "family_availability": availability,
            "safe_counter_columns": safe_counter_columns(rows),
            "phase_cluster_models": cluster_summary_rows,
            "selected_counters_by_scope": selected_counters_by_scope or {},
        },
    )
    return {
        "experiment": split.name,
        "mode": split.mode,
        "rows": len(rows),
        "families": len(FAMILY_COUNTERS),
    }


def build_counter_sequences(
    input_csv: Path,
    output_root: Path,
    horizon: int,
    threshold_mode: str,
    experiment_mode: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    ablation_results: Path | None = None,
    require_ablation_coverage: bool = False,
    write_family_sequences: bool | None = None,
) -> list[dict[str, object]]:
    """Top-level entrypoint used by the CLI and orchestration pipeline."""

    rows = parsec_rows(input_csv)
    splits = build_experiment_splits(rows, experiment_mode, train_fraction, val_fraction, seed)
    should_write_family_sequences = bool(ablation_results and ablation_results.exists()) if write_family_sequences is None else write_family_sequences
    selections: SelectedCounterMap = {}
    global_one_per_family: SelectedCounterMap = {}
    if ablation_results and ablation_results.exists():
        selections = load_ablation_selected_counters(ablation_results)
        global_one_per_family = load_ablation_global_one_per_family(ablation_results)
    summaries: list[dict[str, object]] = []
    for split in splits:
        split_output = ensure_dir(output_root / split.name)
        scopes = ["global", "per_workload"] if threshold_mode == "both" else [threshold_mode]
        selected_counters_by_scope: dict[str, dict[str, list[str]]] = {}
        for scope in scopes:
            global_selected = resolve_selected_counters(global_one_per_family, split.name, scope)
            selected_counters_by_scope[scope] = global_selected if global_selected else resolve_selected_counters(selections, split.name, scope)
        if ablation_results and ablation_results.exists():
            for scope in scopes:
                for family in FAMILY_COUNTERS:
                    if selected_counters_by_scope.get(scope, {}).get(family, []):
                        continue
                    print(
                        f"[sequences] experiment={split.name} scope={scope} family={family} no single counter selected from ablation; no family sequence will be written",
                        flush=True,
                    )
        if require_ablation_coverage and ablation_results and ablation_results.exists():
            for scope in scopes:
                missing = [family for family in FAMILY_COUNTERS if len(selected_counters_by_scope.get(scope, {}).get(family, [])) != 1]
                if missing:
                    missing_text = ", ".join(sorted(missing))
                    raise ValueError(
                        f"Ablation coverage check failed for experiment={split.name} scope={scope}: "
                        f"missing exactly one selected counter for families [{missing_text}] in {ablation_results}"
                    )
        summaries.append(
            build_counter_sequences_for_split(
                rows,
                split,
                split_output,
                horizon,
                threshold_mode,
                seed,
                selected_counters_by_scope=selected_counters_by_scope,
                write_family_sequences=should_write_family_sequences,
            )
        )
    write_csv_rows(output_root / "counter_sequence_runs.csv", summaries)
    return summaries
