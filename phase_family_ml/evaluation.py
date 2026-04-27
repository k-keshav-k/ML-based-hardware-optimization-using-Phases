"""Evaluation utilities for factorized family-wise teacher/student outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows

from .metrics import classification_metrics, hamming_distance_rows


def _eval_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    val = [row for row in rows if row.get("split", "") == "val"]
    if val:
        return val
    test = [row for row in rows if row.get("split", "") == "test"]
    if test:
        return test
    return [row for row in rows if row.get("split", "") != "train"] or rows


def _family_metrics(rows: list[dict[str, str]], model_name: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", ""))].append(row)
    for family, family_rows in sorted(by_family.items()):
        eval_rows = _eval_rows(family_rows)
        y_true = np.asarray([int(row.get("y_true_future_state_1", "-1") or -1) for row in eval_rows], dtype=int)
        y_pred = np.asarray([int(row.get("y_pred_future_state_1", "-1") or -1) for row in eval_rows], dtype=int)
        current = np.asarray([int(row.get("family_state", "-1") or -1) for row in eval_rows], dtype=int)
        metrics = classification_metrics(y_true, y_pred, current_state=current)
        output.append({"model": model_name, "family": family, "samples": len(eval_rows), **metrics})
    return output


def _tuple_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Build tuple maps keyed by row identity and family name."""

    tuples: dict[str, dict[str, int]] = defaultdict(dict)
    for row in _eval_rows(rows):
        key = "|".join(
            [
                str(row.get("run_id", "")),
                str(row.get("core_id", "")),
                str(row.get("row_index", "")),
                str(row.get("split", "")),
            ]
        )
        family = str(row.get("family", ""))
        tuples[f"true::{key}"][family] = int(row.get("y_true_future_state_1", "-1") or -1)
        tuples[f"pred::{key}"][family] = int(row.get("y_pred_future_state_1", "-1") or -1)
    return tuples


def _tuple_metrics(rows: list[dict[str, str]], model: str) -> dict[str, object]:
    tuples = _tuple_rows(rows)
    true_keys = sorted([key for key in tuples if key.startswith("true::")])
    if not true_keys:
        return {
            "model": model,
            "samples": 0,
            "families_per_tuple": 0,
            "exact_tuple_accuracy": 0.0,
            "tuple_hamming_distance": 0.0,
        }
    families = sorted({family for key in true_keys for family in tuples[key].keys()})
    y_true = np.full((len(true_keys), len(families)), -1, dtype=int)
    y_pred = np.full((len(true_keys), len(families)), -1, dtype=int)
    for row_index, true_key in enumerate(true_keys):
        pred_key = true_key.replace("true::", "pred::", 1)
        for family_index, family in enumerate(families):
            y_true[row_index, family_index] = tuples[true_key].get(family, -1)
            y_pred[row_index, family_index] = tuples.get(pred_key, {}).get(family, -1)
    exact = float(np.mean(np.all(y_true == y_pred, axis=1))) if y_true.size else 0.0
    return {
        "model": model,
        "samples": y_true.shape[0],
        "families_per_tuple": y_true.shape[1],
        "exact_tuple_accuracy": exact,
        "tuple_hamming_distance": hamming_distance_rows(y_true, y_pred),
    }


def _hardware_cost_rows(
    family_rows: list[dict[str, object]],
    student_summary_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    selected = [row for row in ablation_rows if int(row.get("selected", "0") or 0) == 1 and row.get("family", "") not in {"", "__global__"}]
    counter_map: dict[str, int] = {}
    for row in selected:
        counters = [item for item in str(row.get("counter_set", "")).split(",") if item]
        counter_map[str(row.get("family", ""))] = len(counters)
    lookup_map: dict[str, int] = {}
    tree_depth_map: dict[str, int] = {}
    for row in student_summary_rows:
        family = str(row.get("family", ""))
        if row.get("model") == "lookup_rle_student":
            lookup_map[family] = int(float(str(row.get("lookup_entries", "0") or 0)))
        if row.get("model") == "decision_tree_student":
            tree_depth_map[family] = int(float(str(row.get("tree_depth", "0") or 0)))
    output: list[dict[str, object]] = []
    for row in family_rows:
        family = str(row.get("family", ""))
        if family == "":
            continue
        counter_count = int(counter_map.get(family, 0))
        tree_depth = int(tree_depth_map.get(family, 0))
        lookup_entries = int(lookup_map.get(family, 0))
        # Crude but comparable storage estimate in bytes.
        est_storage = counter_count * 8 + tree_depth * 32 + lookup_entries * 12
        output.append(
            {
                "family": family,
                "counter_count": counter_count,
                "tree_depth": tree_depth,
                "lookup_table_entries": lookup_entries,
                "estimated_storage_bytes": est_storage,
            }
        )
    return output


def evaluate_outputs(
    teacher_predictions: Path,
    student_predictions: Path,
    student_summary: Path,
    ablation_results: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Write required evaluation artifacts from produced prediction tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_rows = load_csv_rows(teacher_predictions)
    student_rows = load_csv_rows(student_predictions)
    student_summary_rows = load_csv_rows(student_summary) if student_summary.exists() else []
    ablation_rows = load_csv_rows(ablation_results) if ablation_results.exists() else []

    comparison_rows: list[dict[str, object]] = []
    comparison_rows.extend(_family_metrics(teacher_rows, "teacher_transformer"))
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in student_rows:
        by_model[str(row.get("model", ""))].append(row)
    for model_name, rows in sorted(by_model.items()):
        comparison_rows.extend(_family_metrics(rows, model_name))

    # Teacher-vs-student retention at horizon 1.
    teacher_by_key = {
        (row.get("family", ""), row.get("run_id", ""), row.get("core_id", ""), row.get("row_index", "")): int(row.get("y_pred_future_state_1", "-1") or -1)
        for row in teacher_rows
    }
    retention_rows: list[dict[str, object]] = []
    for model_name, rows in sorted(by_model.items()):
        by_family: dict[str, list[float]] = defaultdict(list)
        for row in _eval_rows(rows):
            key = (row.get("family", ""), row.get("run_id", ""), row.get("core_id", ""), row.get("row_index", ""))
            teacher = teacher_by_key.get(key, -1)
            pred = int(row.get("y_pred_future_state_1", "-1") or -1)
            if teacher >= 0:
                by_family[str(row.get("family", ""))].append(1.0 if pred == teacher else 0.0)
        for family, values in by_family.items():
            retention_rows.append(
                {
                    "model": model_name,
                    "family": family,
                    "teacher_retention": float(np.mean(values)) if values else 0.0,
                }
            )

    # Merge retention back into comparison rows.
    retention_map = {(row["model"], row["family"]): float(row["teacher_retention"]) for row in retention_rows}
    for row in comparison_rows:
        if row["model"] == "teacher_transformer":
            row["teacher_retention"] = 1.0
        else:
            row["teacher_retention"] = retention_map.get((row["model"], row["family"]), 0.0)

    write_csv_rows(output_dir / "family_teacher_student_comparison.csv", comparison_rows)

    tuple_rows = [
        _tuple_metrics(teacher_rows, "teacher_transformer"),
    ]
    for model_name, rows in sorted(by_model.items()):
        tuple_rows.append(_tuple_metrics(rows, model_name))
    write_csv_rows(output_dir / "tuple_phase_prediction_results.csv", tuple_rows)

    hw_rows = _hardware_cost_rows(comparison_rows, student_summary_rows, ablation_rows)
    write_csv_rows(output_dir / "hardware_cost_summary.csv", hw_rows)

    return {
        "family_rows": len(comparison_rows),
        "tuple_rows": len(tuple_rows),
        "hardware_rows": len(hw_rows),
    }
