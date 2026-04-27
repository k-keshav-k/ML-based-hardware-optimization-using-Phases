"""Counter ablation runners for family-wise phase labeling.

Implemented ablations:
- per-family single-counter sweeps plus all-counters upper bound
- cross-family exhaustive search selecting exactly one counter per family
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, safe_float, write_csv_rows, write_json
from phase_ml.baselines import DecisionTree

from .data import load_scope_family_data
from .families import FAMILY_COUNTERS, FORBIDDEN_PARTS
from .metrics import classification_metrics


def parsec_rows(input_csv: Path) -> list[dict[str, str]]:
    return [row for row in load_csv_rows(input_csv) if row.get("suite", "") == "parsec"]


def _valid_counter_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = sorted({column for row in rows for column in row.keys() if column.startswith("counter__")})
    return [column for column in columns if not any(part in column.lower() for part in FORBIDDEN_PARTS)]


def _counter_matrix(rows: list[dict[str, str]], counters: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(counters)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, counter in enumerate(counters):
            matrix[row_index, col_index] = safe_float(row.get(counter, ""))
    return matrix


def _fill_train_medians(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    if matrix.size == 0:
        return filled
    for col in range(matrix.shape[1]):
        train_values = matrix[train_mask, col]
        clean = train_values[np.isfinite(train_values)]
        median = float(np.median(clean)) if clean.size else 0.0
        mask = ~np.isfinite(filled[:, col])
        filled[mask, col] = median
    return filled


def _eval_mask(split: np.ndarray) -> np.ndarray:
    val = split == "val"
    if np.any(val):
        return val
    test = split == "test"
    if np.any(test):
        return test
    return split != "train"


def _weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return (
        float(weights.get("accuracy", 0.0)) * float(metrics.get("accuracy", 0.0))
        + float(weights.get("phase_change_f1", 0.0)) * float(metrics.get("phase_change_f1", 0.0))
        + float(weights.get("high_usage_recall", 0.0)) * float(metrics.get("high_usage_recall", 0.0))
    )


def _row_indices(label_rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row.get("row_index", "0") or 0) for row in label_rows], dtype=int)


def _fit_predict_tree(x: np.ndarray, y: np.ndarray, split: np.ndarray, max_depth: int, min_leaf: int) -> np.ndarray:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return np.empty(0, dtype=int)
    train = split == "train"
    if not np.any(train):
        train = np.ones(split.shape[0], dtype=bool)
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_leaf)
    tree.fit(x[train], y[train])
    return tree.predict(x)


def run_ablation(
    input_csv: Path,
    experiment_dir: Path,
    scope: str,
    horizon: int,
    output_csv: Path,
    weights: dict[str, float],
    tree_max_depth: int,
    tree_min_samples_leaf: int,
    run_global_exhaustive: bool,
) -> list[dict[str, object]]:
    """Execute ablation sweeps and return all result rows."""

    rows = parsec_rows(input_csv)
    valid_counters = set(_valid_counter_columns(rows))
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    output_rows: list[dict[str, object]] = []

    # Per-family sweep: every singleton + all-counters upper bound.
    for family in sorted(family_data.keys()):
        labels = family_data[family]
        row_indices = _row_indices(labels.rows)
        if row_indices.size == 0 or labels.future_states.shape[0] == 0:
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "unavailable",
                    "counter_set": "",
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "phase_change_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                    "selected": 1,
                    "note": "no_label_rows",
                }
            )
            continue

        # Guard against stale/mismatched row_index values when labels were
        # produced from a different merged CSV than the one passed to ablation.
        aligned = (row_indices >= 0) & (row_indices < len(rows))
        if not np.any(aligned):
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "unavailable",
                    "counter_set": "",
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "phase_change_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                    "selected": 1,
                    "note": "row_index_mismatch",
                }
            )
            continue
        row_indices = row_indices[aligned]
        targets = labels.future_states[aligned, 0]
        current = labels.family_state[aligned]
        split = labels.split[aligned]

        # We need at least one valid (known) target to score this family.
        valid_target = targets >= 0
        if not np.any(valid_target):
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "unavailable",
                    "counter_set": "",
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "phase_change_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                    "selected": 1,
                    "note": "no_valid_targets",
                }
            )
            continue
        eval_mask = _eval_mask(split)
        candidates = [counter for counter in FAMILY_COUNTERS.get(family, []) if counter in valid_counters]
        if not candidates:
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "unavailable",
                    "counter_set": "",
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "phase_change_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                    "selected": 1,
                }
            )
            continue
        candidate_sets: list[tuple[str, list[str]]] = []
        for counter in candidates:
            candidate_sets.append(("singleton", [counter]))
        if len(candidates) > 1:
            candidate_sets.append(("all_counters", list(candidates)))
        best_score = -1.0
        best_key = ""
        for candidate_type, selected_counters in candidate_sets:
            matrix = _counter_matrix(rows, selected_counters)
            x = matrix[row_indices]
            train_mask = split == "train"
            x = _fill_train_medians(x, train_mask)
            pred = _fit_predict_tree(x, targets, split, tree_max_depth, tree_min_samples_leaf)
            usable = eval_mask & valid_target
            if not np.any(usable):
                usable = valid_target
            metrics = classification_metrics(targets[usable], pred[usable], current_state=current[usable])
            score = _weighted_score(metrics, weights)
            key = ",".join(selected_counters)
            if score > best_score:
                best_score = score
                best_key = key
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": candidate_type,
                    "counter_set": key,
                    **metrics,
                    "validation_score": score,
                    "selected": 0,
                }
            )
        for row in output_rows:
            if row.get("family") == family and row.get("counter_set") == best_key and row.get("scope") == scope:
                row["selected"] = 1

    # Cross-family exhaustive sweep: exactly one counter per family.
    if run_global_exhaustive:
        families = []
        for family in sorted(family_data.keys()):
            labels = family_data[family]
            row_indices = _row_indices(labels.rows)
            has_rows = row_indices.size > 0 and labels.future_states.shape[0] > 0
            has_counters = any(counter in valid_counters for counter in FAMILY_COUNTERS.get(family, []))
            if has_rows and has_counters:
                families.append(family)
        family_counter_lists = []
        for family in families:
            options = [counter for counter in FAMILY_COUNTERS.get(family, []) if counter in valid_counters]
            if options:
                family_counter_lists.append(options)
        best_combo = ""
        best_score = -1.0
        for combo in itertools.product(*family_counter_lists) if family_counter_lists else []:
            selected = list(combo)
            combo_metrics = []
            for family in families:
                labels = family_data[family]
                row_indices = _row_indices(labels.rows)
                aligned = (row_indices >= 0) & (row_indices < len(rows))
                if not np.any(aligned):
                    continue
                row_indices = row_indices[aligned]
                split = labels.split
                split = split[aligned]
                eval_mask = _eval_mask(split)
                x = _counter_matrix(rows, selected)[row_indices]
                x = _fill_train_medians(x, split == "train")
                target = labels.future_states[aligned, 0]
                valid_target = target >= 0
                if not np.any(valid_target):
                    continue
                pred = _fit_predict_tree(x, target, split, tree_max_depth, tree_min_samples_leaf)
                current_state = labels.family_state[aligned]
                usable = eval_mask & valid_target
                if not np.any(usable):
                    usable = valid_target
                metrics = classification_metrics(target[usable], pred[usable], current_state=current_state[usable])
                combo_metrics.append(_weighted_score(metrics, weights))
            aggregate = float(np.mean(combo_metrics)) if combo_metrics else 0.0
            combo_key = ",".join(selected)
            if aggregate > best_score:
                best_score = aggregate
                best_combo = combo_key
            output_rows.append(
                {
                    "family": "__global__",
                    "scope": scope,
                    "candidate_type": "one_per_family_exhaustive",
                    "counter_set": combo_key,
                    "accuracy": "",
                    "macro_f1": "",
                    "phase_change_f1": "",
                    "high_usage_recall": "",
                    "validation_score": aggregate,
                    "selected": 0,
                }
            )
        for row in output_rows:
            if row.get("family") == "__global__" and row.get("counter_set") == best_combo and row.get("scope") == scope:
                row["selected"] = 1

    write_csv_rows(output_csv, output_rows)
    write_json(
        output_csv.with_suffix(".json"),
        {
            "scope": scope,
            "horizon": horizon,
            "rows": len(output_rows),
            "selected_rows": [row for row in output_rows if int(row.get("selected", 0)) == 1],
        },
    )
    return output_rows
