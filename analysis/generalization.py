"""Generalization checks for workload and experiment-setting transfer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phase_ml.metrics import classification_metrics

from .ablation import window_features
from .common import FEATURE_GROUPS, load_rows, write_json, write_rows
from .detector import CentroidDetector


def evaluate_transfer(rows: list[dict[str, str]], x: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> dict[str, float]:
    y = np.asarray([int(row["next_phase_id"]) for row in rows], dtype=int)
    current = np.asarray([int(row["phase_id"]) for row in rows], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in rows], dtype=int)
    if not np.any(train_mask) or not np.any(test_mask):
        return {"accuracy": 0.0, "macro_f1": 0.0, "phase_change_precision": 0.0, "phase_change_recall": 0.0, "phase_change_f1": 0.0}
    detector = CentroidDetector.fit(x[train_mask], y[train_mask], metric="manhattan", max_centroids=16)
    pred = detector.predict(x)
    pred_change = (pred != current).astype(int)
    return classification_metrics(y[test_mask], pred[test_mask], change[test_mask], pred_change[test_mask])


def run_generalization(dataset_dir: Path, label_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = load_rows(label_dir / "window_labels.csv")
    x, _ = window_features(dataset_dir, FEATURE_GROUPS["all"])
    output_rows: list[dict[str, object]] = []

    workloads = sorted({row["workload"] for row in rows})
    for workload in workloads:
        train = np.asarray([row["workload"] != workload for row in rows])
        test = ~train
        item: dict[str, object] = {"experiment": "leave_one_workload_out", "train": "all_except_" + workload, "test": workload}
        item.update(evaluate_transfer(rows, x, train, test))
        output_rows.append(item)

    settings = sorted({row["experiment_set"] for row in rows})
    for setting in settings:
        train = np.asarray([row["experiment_set"] == setting for row in rows])
        test = ~train
        item = {"experiment": "train_one_setting_test_others", "train": setting, "test": "not_" + setting}
        item.update(evaluate_transfer(rows, x, train, test))
        output_rows.append(item)
    for setting in settings:
        train = np.asarray([row["experiment_set"] != setting for row in rows])
        test = ~train
        item = {"experiment": "train_other_settings_test_one", "train": "not_" + setting, "test": setting}
        item.update(evaluate_transfer(rows, x, train, test))
        output_rows.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "train", "test", "accuracy", "macro_f1", "phase_change_precision", "phase_change_recall", "phase_change_f1"]
    write_rows(output_dir / "generalization.csv", output_rows, fieldnames)
    write_json(
        output_dir / "generalization_summary.json",
        {
            "analysis_type": "nearest_centroid_transfer",
            "best_phase_change_f1": max(output_rows, key=lambda row: float(row["phase_change_f1"])),
            "worst_phase_change_f1": min(output_rows, key=lambda row: float(row["phase_change_f1"])),
            "rows": output_rows,
        },
    )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = run_generalization(Path(args.dataset_dir), Path(args.label_dir), Path(args.output_dir))
    print(f"Wrote {len(rows)} generalization rows.")


if __name__ == "__main__":
    main()
