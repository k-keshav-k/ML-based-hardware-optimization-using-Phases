"""End-to-end orchestration for factorized family-wise phase modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import write_csv_rows

from .ablation import run_ablation
from .config import load_config
from .labels import build_counter_sequences
from .orchestration import experiment_dirs, scopes_for_experiment
from .train_phase_detector import train_phase_detector_for_experiment


def run_pipeline_for_dataset(
    *,
    input_csv: Path,
    output_dir: Path,
    config: dict[str, object],
    experiment_mode: str,
    threshold_mode: str,
) -> dict[str, int]:
    """Run the full family-ML pipeline for one input dataset."""

    dataset_cfg = config["dataset"]
    detector_cfg = dict(config["phase_detector"])
    sequence_horizon = max(int(dataset_cfg["horizon"]), int(detector_cfg["prediction_horizon"]))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    sequences_root = root / "counter_sequences"
    ablation_root = root / "ablation"
    detector_root = root / "phase_detector"

    build_counter_sequences(
        input_csv=input_csv,
        output_root=sequences_root,
        horizon=sequence_horizon,
        threshold_mode=threshold_mode,
        experiment_mode=experiment_mode,
        train_fraction=float(config["splits"]["train_fraction"]),
        val_fraction=float(config["splits"]["val_fraction"]),
        seed=int(config["random_seed"]),
    )

    ablation_root.mkdir(parents=True, exist_ok=True)
    ablation_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            rows = run_ablation(
                input_csv=input_csv,
                experiment_dir=exp_dir,
                scope=scope,
                horizon=sequence_horizon,
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

    # Refresh streams with the selected per-family counters from ablation so
    # the hardware detector uses exactly one representative counter per family.
    build_counter_sequences(
        input_csv=input_csv,
        output_root=sequences_root,
        horizon=sequence_horizon,
        threshold_mode=threshold_mode,
        experiment_mode=experiment_mode,
        train_fraction=float(config["splits"]["train_fraction"]),
        val_fraction=float(config["splits"]["val_fraction"]),
        seed=int(config["random_seed"]),
        ablation_results=ablation_root / "family_ablation_results.csv",
    )

    detector_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            detector_rows.extend(
                train_phase_detector_for_experiment(
                    experiment_dir=exp_dir,
                    scope=scope,
                    output_dir=detector_root / exp_dir.name / scope,
                    horizon=sequence_horizon,
                    history_length=int(detector_cfg["history_length"]),
                    prediction_horizon=int(detector_cfg["prediction_horizon"]),
                    tree_max_depth=int(detector_cfg["decision_tree_max_depth"]),
                    tree_min_leaf=int(detector_cfg["decision_tree_min_samples_leaf"]),
                    detector_config=detector_cfg,
                )
            )
    write_csv_rows(detector_root / "phase_detector_summary_all.csv", detector_rows)

    summary = {
        "ablation_rows": len(ablation_rows),
        "phase_detector_rows": len(detector_rows),
    }
    print(
        f"Pipeline complete: ablation={summary['ablation_rows']} "
        f"phase_detector_rows={summary['phase_detector_rows']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--experiment-mode", choices=["per_workload_holdout", "pooled_run_group", "config_group_holdout", "leave_one_workload_out", "all"], default="")
    parser.add_argument("--threshold-mode", choices=["global", "per_workload", "both"], default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    run_pipeline_for_dataset(
        input_csv=Path(args.input or dataset_cfg["input_csv"]),
        output_dir=Path(args.output_dir or dataset_cfg["output_dir"]),
        config=config,
        experiment_mode=str(args.experiment_mode or config["experiments"]["default_mode"]),
        threshold_mode=str(args.threshold_mode or config["families"]["threshold_mode"]),
    )


if __name__ == "__main__":
    main()
