"""Counter ablation runners for family-wise counter sequences.

Implemented ablations:
- per-family single-counter sweeps from per-counter state sequences
- cross-family exhaustive search selecting exactly one counter per family
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, safe_float, write_csv_rows, write_json

from .data import CounterSequenceData, load_counter_sequence
from .families import FAMILY_COUNTERS, FORBIDDEN_PARTS
from .metrics import classification_metrics
from .tree import DecisionTree


def parsec_rows(input_csv: Path) -> list[dict[str, str]]:
    return [row for row in load_csv_rows(input_csv) if row.get("suite", "") == "parsec"]


def _valid_counter_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    columns = sorted({column for row in rows for column in row.keys() if column.startswith("counter__")})
    return [column for column in columns if not any(part in column.lower() for part in FORBIDDEN_PARTS)]


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


def _selection_mask(split: np.ndarray) -> np.ndarray:
    val = split == "val"
    if np.any(val):
        return val
    test = split == "test"
    if np.any(test):
        return test
    return split != "train"


def _report_mask(split: np.ndarray) -> np.ndarray:
    test = split == "test"
    if np.any(test):
        return test
    val = split == "val"
    if np.any(val):
        return val
    return split != "train"


def _selection_split_name(split: np.ndarray) -> str:
    if np.any(split == "val"):
        return "val"
    if np.any(split == "test"):
        return "test"
    return "non_train"


def _report_split_name(split: np.ndarray) -> str:
    if np.any(split == "test"):
        return "test"
    if np.any(split == "val"):
        return "val"
    return "non_train"


def _weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return (
        float(weights.get("accuracy", 0.0)) * float(metrics.get("accuracy", 0.0))
        + float(weights.get("high_usage_recall", 0.0)) * float(metrics.get("high_usage_recall", 0.0))
    )


def _counter_sequence_path(experiment_dir: Path, scope: str, counter: str) -> Path:
    return experiment_dir / f"threshold_{scope}" / "counter_value_sequences" / f"counter_sequence_{counter}.csv"


def _load_family_counter_sequences(experiment_dir: Path, scope: str, horizon: int) -> dict[str, dict[str, CounterSequenceData]]:
    output: dict[str, dict[str, CounterSequenceData]] = {}
    for family, counters in FAMILY_COUNTERS.items():
        for counter in counters:
            path = _counter_sequence_path(experiment_dir, scope, counter)
            if path.exists():
                output.setdefault(family, {})[counter] = load_counter_sequence(path, horizon)
    return output


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
    family_counter_data = _load_family_counter_sequences(experiment_dir, scope, horizon)
    output_rows: list[dict[str, object]] = []
    singleton_scores: dict[tuple[str, str], float] = {}
    singleton_report_scores: dict[tuple[str, str], float] = {}
    selection_split = ""
    report_split = ""

    # Per-family sweep: score each counter against its own state sequence.
    for family in sorted(FAMILY_COUNTERS):
        counter_data = {
            counter: payload
            for counter, payload in family_counter_data.get(family, {}).items()
            if counter in valid_counters
        }
        if not counter_data:
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "unavailable",
                    "counter_set": "",
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                    "selected": 0,
                    "note": "no_counter_sequence_rows",
                }
            )
            continue
        best_score = -1.0
        best_key = ""
        for counter, sequences in counter_data.items():
            targets = sequences.future_states[:, 0]
            current = sequences.family_state
            split = sequences.split
            values = np.asarray([safe_float(row.get("counter_value", "")) for row in sequences.rows], dtype=float)
            x = values.reshape(-1, 1)
            train_mask = split == "train"
            x = _fill_train_medians(x, train_mask)
            valid_target = targets >= 0
            if not np.any(valid_target):
                selection_metrics = {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "high_usage_recall": 0.0,
                }
                report_metrics = dict(selection_metrics)
                score = 0.0
                report_score = 0.0
            else:
                selection_split = _selection_split_name(split)
                report_split = _report_split_name(split)
                selection_mask = _selection_mask(split)
                report_mask = _report_mask(split)
                pred = _fit_predict_tree(x, targets, split, tree_max_depth, tree_min_samples_leaf)
                selection_usable = selection_mask & valid_target
                if not np.any(selection_usable):
                    selection_usable = valid_target
                report_usable = report_mask & valid_target
                if not np.any(report_usable):
                    report_usable = selection_usable
                selection_metrics = classification_metrics(targets[selection_usable], pred[selection_usable], current_state=current[selection_usable])
                report_metrics = classification_metrics(targets[report_usable], pred[report_usable], current_state=current[report_usable])
                score = _weighted_score(selection_metrics, weights)
                report_score = _weighted_score(report_metrics, weights)
            key = counter
            singleton_scores[(family, counter)] = score
            singleton_report_scores[(family, counter)] = report_score
            if score > best_score:
                best_score = score
                best_key = key
            output_rows.append(
                {
                    "family": family,
                    "scope": scope,
                    "candidate_type": "singleton",
                    "counter_set": key,
                    "selection_split": selection_split,
                    "report_split": report_split,
                    "selection_accuracy": selection_metrics["accuracy"],
                    "selection_macro_f1": selection_metrics["macro_f1"],
                    "selection_high_usage_recall": selection_metrics["high_usage_recall"],
                    "report_accuracy": report_metrics["accuracy"],
                    "report_macro_f1": report_metrics["macro_f1"],
                    "report_high_usage_recall": report_metrics["high_usage_recall"],
                    "accuracy": report_metrics["accuracy"],
                    "macro_f1": report_metrics["macro_f1"],
                    "high_usage_recall": report_metrics["high_usage_recall"],
                    "validation_score": score,
                    "report_score": report_score,
                    "selected": 0,
                    "target_source": "counter_sequence",
                }
            )
        for row in output_rows:
            if row.get("family") == family and row.get("counter_set") == best_key and row.get("scope") == scope:
                row["selected"] = 1

    # Cross-family exhaustive sweep: exactly one counter per family.
    if run_global_exhaustive:
        families = []
        for family in sorted(FAMILY_COUNTERS):
            options = [counter for counter in FAMILY_COUNTERS.get(family, []) if (family, counter) in singleton_scores]
            if options:
                families.append(family)
        family_counter_lists = []
        for family in families:
            options = [counter for counter in FAMILY_COUNTERS.get(family, []) if (family, counter) in singleton_scores]
            if options:
                family_counter_lists.append(options)
        best_combo = ""
        best_score = -1.0
        for combo in itertools.product(*family_counter_lists) if family_counter_lists else []:
            selected = list(combo)
            family_counter_map = {family: counter for family, counter in zip(families, selected)}
            combo_scores = [singleton_scores[(family, counter)] for family, counter in family_counter_map.items()]
            combo_report_scores = [singleton_report_scores[(family, counter)] for family, counter in family_counter_map.items()]
            aggregate = float(np.mean(combo_scores)) if combo_scores else 0.0
            report_aggregate = float(np.mean(combo_report_scores)) if combo_report_scores else 0.0
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
                    "family_counter_map": ";".join(f"{family}:{counter}" for family, counter in family_counter_map.items()),
                    "selection_split": selection_split,
                    "report_split": report_split,
                    "accuracy": "",
                    "macro_f1": "",
                    "high_usage_recall": "",
                    "validation_score": aggregate,
                    "report_score": report_aggregate,
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
