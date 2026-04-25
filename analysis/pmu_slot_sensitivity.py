"""Evaluate detector quality under limited PMU counter-slot budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phase_ml.metrics import classification_metrics

from .ablation import window_features
from .common import load_rows, write_json, write_rows
from .detector import CentroidDetector


SLOT_FEATURES = {
    "2": ["branch_miss_rate", "offcore_reads_per_ki"],
    "4": ["branch_miss_rate", "offcore_reads_per_ki", "llc_miss_rate", "l1d_loads_per_ki"],
    "6": ["branch_miss_rate", "offcore_reads_per_ki", "llc_miss_rate", "l1d_loads_per_ki", "branch_instructions_per_ki", "llc_references_per_ki"],
    "8": [
        "branch_miss_rate",
        "offcore_reads_per_ki",
        "llc_miss_rate",
        "l1d_loads_per_ki",
        "branch_instructions_per_ki",
        "llc_references_per_ki",
        "l1d_stores_per_ki",
        "load_store_ratio",
    ],
}


def run_slot_sensitivity(dataset_dir: Path, label_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = load_rows(label_dir / "window_labels.csv")
    y = np.asarray([int(row["next_phase_id"]) for row in rows], dtype=int)
    current = np.asarray([int(row["phase_id"]) for row in rows], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in rows], dtype=int)
    split = np.asarray([row["split"] for row in rows])
    train = split == "train"
    eval_mask = split != "train"
    output = []
    for slots, features in SLOT_FEATURES.items():
        x, used = window_features(dataset_dir, features)
        detector = CentroidDetector.fit(x[train], y[train], metric="manhattan", max_centroids=16)
        pred = detector.predict(x)
        pred_change = (pred != current).astype(int)
        metrics = classification_metrics(y[eval_mask], pred[eval_mask], change[eval_mask], pred_change[eval_mask])
        output.append({"counter_slots": slots, "feature_count": len(used), "features": ",".join(used), **metrics})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "pmu_slot_sensitivity.csv",
        output,
        ["counter_slots", "feature_count", "features", "accuracy", "macro_f1", "phase_change_precision", "phase_change_recall", "phase_change_f1"],
    )
    write_json(
        output_dir / "pmu_slot_sensitivity_summary.json",
        {
            "analysis_type": "offline_counter_slot_sensitivity",
            "best_phase_change_f1": max(output, key=lambda row: float(row["phase_change_f1"])),
            "rows": output,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = run_slot_sensitivity(Path(args.dataset_dir), Path(args.label_dir), Path(args.output_dir))
    print(f"Wrote {len(rows)} PMU-slot sensitivity rows.")


if __name__ == "__main__":
    main()

