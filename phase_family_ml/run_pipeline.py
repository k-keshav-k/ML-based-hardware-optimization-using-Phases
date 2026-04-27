"""End-to-end orchestration for factorized family-wise phase modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows

from .ablation import run_ablation
from .config import apply_runtime_profile, load_config
from .evaluation import evaluate_outputs
from .labels import build_family_labels
from .orchestration import experiment_dirs, scopes_for_experiment
from .students import train_students_for_experiment
from .teacher import train_teachers_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--experiment-mode", choices=["per_workload_holdout", "pooled_run_group", "leave_one_workload_out", "all"], default="")
    parser.add_argument("--threshold-mode", choices=["global", "per_workload", "both"], default="")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = apply_runtime_profile(load_config(args.config or None), full=args.full)
    dataset_cfg = config["dataset"]
    input_csv = Path(args.input or dataset_cfg["input_csv"])
    root = Path(args.output_dir or dataset_cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)

    labels_root = root / "family_labels"
    ablation_root = root / "ablation"
    teacher_root = root / "teacher"
    students_root = root / "students"
    eval_root = root / "evaluation"

    build_family_labels(
        input_csv=input_csv,
        output_root=labels_root,
        horizon=int(dataset_cfg["horizon"]),
        threshold_mode=str(args.threshold_mode or config["families"]["threshold_mode"]),
        experiment_mode=str(args.experiment_mode or config["experiments"]["default_mode"]),
        train_fraction=float(config["splits"]["train_fraction"]),
        val_fraction=float(config["splits"]["val_fraction"]),
        seed=int(config["random_seed"]),
    )

    ablation_root.mkdir(parents=True, exist_ok=True)
    ablation_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(labels_root):
        for scope in scopes_for_experiment(exp_dir):
            rows = run_ablation(
                input_csv=input_csv,
                experiment_dir=exp_dir,
                scope=scope,
                horizon=int(dataset_cfg["horizon"]),
                output_csv=ablation_root / f"ablation_{exp_dir.name}_{scope}.csv",
                weights=dict(config["ablation"]["score_weights"]),
                tree_max_depth=int(config["ablation"]["tree_max_depth"]),
                tree_min_samples_leaf=int(config["ablation"]["tree_min_samples_leaf"]),
                run_global_exhaustive=bool(config["ablation"].get("global_exhaustive_one_per_family", True)),
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                ablation_rows.append(item)
    write_csv_rows(ablation_root / "family_ablation_results.csv", ablation_rows)

    # Rebuild labels using the selected per-family counter sets from ablation so
    # teacher training is grounded in the final chosen counters.
    build_family_labels(
        input_csv=input_csv,
        output_root=labels_root,
        horizon=int(dataset_cfg["horizon"]),
        threshold_mode=str(args.threshold_mode or config["families"]["threshold_mode"]),
        experiment_mode=str(args.experiment_mode or config["experiments"]["default_mode"]),
        train_fraction=float(config["splits"]["train_fraction"]),
        val_fraction=float(config["splits"]["val_fraction"]),
        seed=int(config["random_seed"]),
        ablation_results=ablation_root / "family_ablation_results.csv",
    )

    for exp_dir in experiment_dirs(labels_root):
        for scope in scopes_for_experiment(exp_dir):
            teacher_dir = teacher_root / exp_dir.name / scope
            train_teachers_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                output_dir=teacher_dir,
                horizon=int(dataset_cfg["horizon"]),
                history_length=int(dataset_cfg["history_length"]),
                teacher_config=dict(config["teacher"]),
                seed=int(config["random_seed"]),
            )
            train_students_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                teacher_predictions_path=teacher_dir / "teacher_predictions.csv",
                output_dir=students_root / exp_dir.name / scope,
                horizon=int(dataset_cfg["horizon"]),
                blend_alpha=float(config["student"]["blend_alpha"]),
                tree_max_depth=int(config["student"]["decision_tree_max_depth"]),
                tree_min_leaf=int(config["student"]["decision_tree_min_samples_leaf"]),
                run_length_buckets=[int(item) for item in config["student"]["run_length_buckets"]],
            )

    eval_root.mkdir(parents=True, exist_ok=True)
    family_rows: list[dict[str, str]] = []
    tuple_rows: list[dict[str, str]] = []
    hw_rows: list[dict[str, str]] = []
    for exp_dir in experiment_dirs(labels_root):
        for scope in scopes_for_experiment(exp_dir):
            stage_dir = eval_root / exp_dir.name / scope
            evaluate_outputs(
                teacher_predictions=teacher_root / exp_dir.name / scope / "teacher_predictions.csv",
                student_predictions=students_root / exp_dir.name / scope / "student_predictions.csv",
                student_summary=students_root / exp_dir.name / scope / "student_summary.csv",
                ablation_results=ablation_root / "family_ablation_results.csv",
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

    write_csv_rows(eval_root / "family_teacher_student_comparison.csv", family_rows)
    write_csv_rows(eval_root / "tuple_phase_prediction_results.csv", tuple_rows)
    write_csv_rows(eval_root / "hardware_cost_summary.csv", hw_rows)
    write_csv_rows(eval_root / "family_ablation_results.csv", load_csv_rows(ablation_root / "family_ablation_results.csv"))

    print(
        f"Pipeline complete: ablation={len(ablation_rows)} family_metrics={len(family_rows)} tuple_metrics={len(tuple_rows)} hardware_rows={len(hw_rows)}"
    )


if __name__ == "__main__":
    main()
