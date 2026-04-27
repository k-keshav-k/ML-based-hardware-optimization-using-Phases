"""CLI for family-wise transformer teacher training."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import write_csv_rows

from .config import apply_runtime_profile, load_config
from .labels import build_family_labels
from .orchestration import experiment_dirs, scopes_for_experiment
from .teacher import train_teachers_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--labels-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ablation-results", default="")
    parser.add_argument("--threshold-mode", choices=["global", "per_workload", "both"], default="")
    parser.add_argument("--experiment-mode", choices=["per_workload_holdout", "pooled_run_group", "leave_one_workload_out", "all"], default="")
    parser.add_argument("--skip-label-refresh", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = apply_runtime_profile(load_config(args.config or None), full=args.full)
    dataset_cfg = config["dataset"]
    input_csv = Path(args.input or dataset_cfg["input_csv"])
    labels_root = Path(args.labels_root or (Path(dataset_cfg["output_dir"]) / "family_labels"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "teacher"))
    ablation_results = Path(args.ablation_results or (Path(dataset_cfg["output_dir"]) / "ablation" / "family_ablation_results.csv"))
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_label_refresh and ablation_results.exists():
        build_family_labels(
            input_csv=input_csv,
            output_root=labels_root,
            horizon=int(dataset_cfg["horizon"]),
            threshold_mode=str(args.threshold_mode or config["families"]["threshold_mode"]),
            experiment_mode=str(args.experiment_mode or config["experiments"]["default_mode"]),
            train_fraction=float(config["splits"]["train_fraction"]),
            val_fraction=float(config["splits"]["val_fraction"]),
            seed=int(config["random_seed"]),
            ablation_results=ablation_results,
        )
        print(f"Refreshed labels from ablation selections: {ablation_results}")

    all_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(labels_root):
        for scope in scopes_for_experiment(exp_dir):
            scope_output = output_root / exp_dir.name / scope
            rows = train_teachers_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                output_dir=scope_output,
                horizon=int(dataset_cfg["horizon"]),
                history_length=int(dataset_cfg["history_length"]),
                teacher_config=dict(config["teacher"]),
                seed=int(config["random_seed"]),
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                all_rows.append(item)
    write_csv_rows(output_root / "teacher_summary_all.csv", all_rows)
    print(f"Wrote teacher summaries for {len(all_rows)} family/context runs.")


if __name__ == "__main__":
    main()
