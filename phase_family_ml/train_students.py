"""CLI for distilled student model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import write_csv_rows

from .config import load_config
from .orchestration import experiment_dirs, scopes_for_experiment
from .students import train_students_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--teacher-root", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    teacher_root = Path(args.teacher_root or (Path(dataset_cfg["output_dir"]) / "teacher"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "students"))
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            teacher_predictions = teacher_root / exp_dir.name / scope / "teacher_predictions.csv"
            if not teacher_predictions.exists():
                continue
            scope_output = output_root / exp_dir.name / scope
            rows = train_students_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                teacher_predictions_path=teacher_predictions,
                output_dir=scope_output,
                horizon=int(dataset_cfg["horizon"]),
                blend_alpha=float(config["student"]["blend_alpha"]),
                tree_max_depth=int(config["student"]["decision_tree_max_depth"]),
                tree_min_leaf=int(config["student"]["decision_tree_min_samples_leaf"]),
                run_length_buckets=[int(item) for item in config["student"]["run_length_buckets"]],
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                all_rows.append(item)
    write_csv_rows(output_root / "student_summary_all.csv", all_rows)
    print(f"Wrote student summaries for {len(all_rows)} family/model runs.")


if __name__ == "__main__":
    main()
