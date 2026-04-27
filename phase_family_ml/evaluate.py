"""CLI for family-wise teacher/student evaluation aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows

from .config import load_config
from .evaluation import evaluate_outputs
from .orchestration import experiment_dirs, scopes_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--labels-root", default="")
    parser.add_argument("--teacher-root", default="")
    parser.add_argument("--students-root", default="")
    parser.add_argument("--ablation-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    labels_root = Path(args.labels_root or (Path(dataset_cfg["output_dir"]) / "family_labels"))
    teacher_root = Path(args.teacher_root or (Path(dataset_cfg["output_dir"]) / "teacher"))
    students_root = Path(args.students_root or (Path(dataset_cfg["output_dir"]) / "students"))
    ablation_dir = Path(args.ablation_dir or (Path(dataset_cfg["output_dir"]) / "ablation"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "evaluation"))
    output_root.mkdir(parents=True, exist_ok=True)

    family_rows: list[dict[str, str]] = []
    tuple_rows: list[dict[str, str]] = []
    hw_rows: list[dict[str, str]] = []
    ablation_path = ablation_dir / "family_ablation_results.csv"

    for exp_dir in experiment_dirs(labels_root):
        for scope in scopes_for_experiment(exp_dir):
            teacher_predictions = teacher_root / exp_dir.name / scope / "teacher_predictions.csv"
            student_predictions = students_root / exp_dir.name / scope / "student_predictions.csv"
            student_summary = students_root / exp_dir.name / scope / "student_summary.csv"
            if not teacher_predictions.exists() or not student_predictions.exists():
                continue
            stage_dir = output_root / exp_dir.name / scope
            evaluate_outputs(
                teacher_predictions=teacher_predictions,
                student_predictions=student_predictions,
                student_summary=student_summary,
                ablation_results=ablation_path,
                output_dir=stage_dir,
            )
            for row in load_csv_rows(stage_dir / "family_teacher_student_comparison.csv"):
                row["experiment"] = exp_dir.name
                row["scope"] = scope
                family_rows.append(row)
            for row in load_csv_rows(stage_dir / "tuple_phase_prediction_results.csv"):
                row["experiment"] = exp_dir.name
                row["scope"] = scope
                tuple_rows.append(row)
            for row in load_csv_rows(stage_dir / "hardware_cost_summary.csv"):
                row["experiment"] = exp_dir.name
                row["scope"] = scope
                hw_rows.append(row)

    write_csv_rows(output_root / "family_teacher_student_comparison.csv", family_rows)
    write_csv_rows(output_root / "tuple_phase_prediction_results.csv", tuple_rows)
    write_csv_rows(output_root / "hardware_cost_summary.csv", hw_rows)
    if ablation_path.exists():
        write_csv_rows(output_root / "family_ablation_results.csv", load_csv_rows(ablation_path))
    print(f"Aggregated evaluation rows: family={len(family_rows)}, tuple={len(tuple_rows)}, hardware={len(hw_rows)}")


if __name__ == "__main__":
    main()
