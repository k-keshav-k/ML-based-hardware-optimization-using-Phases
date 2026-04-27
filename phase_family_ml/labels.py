"""Family-state labeling with train-split tertile thresholds.

This stage converts raw counter intervals into discrete family states and writes
one label file per family in the requested wide multi-horizon schema.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, safe_float, write_csv_rows, write_json

from .families import FAMILY_COUNTERS, derive_feature_values, derived_columns_for_family, family_counter_availability
from .splits import ExperimentSplit, build_experiment_splits

SelectedCounterMap = dict[tuple[str, str], dict[str, list[str]]]


def parsec_rows(input_csv: Path) -> list[dict[str, str]]:
    """Load only PARSEC rows so behavior matches the existing phase_ml policy."""

    return [row for row in load_csv_rows(input_csv) if row.get("suite", "") == "parsec"]


def add_core_id(row: dict[str, str]) -> str:
    """Return a normalized core identifier field expected by new artifacts."""

    return str(row.get("cpu_or_core_id", ""))


def interval_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[int]]:
    """Build run/core streams for temporal horizon labeling."""

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


def load_ablation_selected_counters(ablation_results: Path) -> SelectedCounterMap:
    """Load selected per-family counter sets from ablation results CSV."""

    rows = load_csv_rows(ablation_results)
    selected: SelectedCounterMap = {}
    selected_scores: dict[tuple[str, str, str], float] = {}
    for row in rows:
        family = str(row.get("family", "")).strip()
        if family in {"", "__global__"}:
            continue
        if int(row.get("selected", "0") or 0) != 1:
            continue
        if str(row.get("candidate_type", "")).strip() == "unavailable":
            continue
        counters = _parse_counter_set(str(row.get("counter_set", "")))
        if not counters:
            continue
        experiment = str(row.get("experiment", "")).strip()
        scope = str(row.get("scope", "")).strip()
        score = safe_float(row.get("validation_score", ""))
        key = (experiment, scope, family)
        previous = selected_scores.get(key, -math.inf)
        if score < previous:
            continue
        selected_scores[key] = score
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


def build_family_label_rows(
    rows: list[dict[str, str]],
    states: np.ndarray,
    split_by_run: dict[str, str],
    horizon: int,
) -> list[dict[str, object]]:
    """Build the requested wide family label rows for one family/scope."""

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


def _scope_output_dir(base_dir: Path, scope: str) -> Path:
    return ensure_dir(base_dir / f"threshold_{scope}")


def build_family_labels_for_split(
    rows: list[dict[str, str]],
    split: ExperimentSplit,
    output_dir: Path,
    horizon: int,
    threshold_mode: str,
    selected_counters_by_scope: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, object]:
    """Generate labels for all families for one experiment split."""

    output_dir = ensure_dir(output_dir)
    availability = family_counter_availability(rows)
    scopes = ["global", "per_workload"] if threshold_mode == "both" else [threshold_mode]
    summary_rows: list[dict[str, object]] = []
    for scope in scopes:
        selected_for_scope = selected_counters_by_scope.get(scope, {}) if selected_counters_by_scope else {}
        usage = family_usage_scores(rows, selected_for_scope)
        for family in FAMILY_COUNTERS:
            values = usage.get(family, np.full(len(rows), np.nan, dtype=float))
            thresholds = thresholds_for_family(values, rows, split.split_by_run, scope)
            states = states_for_scope(values, rows, scope, thresholds)
            label_rows = build_family_label_rows(rows, states, split.split_by_run, horizon)
            scope_dir = _scope_output_dir(output_dir, scope)
            write_csv_rows(scope_dir / f"family_labels_{family}.csv", label_rows)
            low, high = thresholds.get("global", (math.nan, math.nan)) if scope == "global" else (math.nan, math.nan)
            selected_counter_set = selected_for_scope.get(family, [])
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
                    "label_source": "ablation_selected" if selected_counter_set else "family_default",
                    "global_low_threshold": "" if not np.isfinite(low) else low,
                    "global_high_threshold": "" if not np.isfinite(high) else high,
                }
            )
    write_csv_rows(output_dir / "family_label_summary.csv", summary_rows)
    write_json(
        output_dir / "family_label_manifest.json",
        {
            "experiment": split.name,
            "mode": split.mode,
            "threshold_mode": threshold_mode,
            "horizon": horizon,
            "rows": len(rows),
            "family_availability": availability,
            "selected_counters_by_scope": selected_counters_by_scope or {},
        },
    )
    return {
        "experiment": split.name,
        "mode": split.mode,
        "rows": len(rows),
        "families": len(FAMILY_COUNTERS),
    }


def build_family_labels(
    input_csv: Path,
    output_root: Path,
    horizon: int,
    threshold_mode: str,
    experiment_mode: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    ablation_results: Path | None = None,
) -> list[dict[str, object]]:
    """Top-level entrypoint used by the CLI and orchestration pipeline."""

    rows = parsec_rows(input_csv)
    splits = build_experiment_splits(rows, experiment_mode, train_fraction, val_fraction, seed)
    selections: SelectedCounterMap = {}
    if ablation_results and ablation_results.exists():
        selections = load_ablation_selected_counters(ablation_results)
    summaries: list[dict[str, object]] = []
    for split in splits:
        split_output = ensure_dir(output_root / split.name)
        scopes = ["global", "per_workload"] if threshold_mode == "both" else [threshold_mode]
        selected_counters_by_scope = {scope: resolve_selected_counters(selections, split.name, scope) for scope in scopes}
        summaries.append(
            build_family_labels_for_split(
                rows,
                split,
                split_output,
                horizon,
                threshold_mode,
                selected_counters_by_scope=selected_counters_by_scope,
            )
        )
    write_csv_rows(output_root / "family_label_runs.csv", summaries)
    return summaries
