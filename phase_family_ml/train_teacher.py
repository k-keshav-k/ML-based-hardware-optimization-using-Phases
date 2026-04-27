"""CLI for family-wise transformer teacher training."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import write_csv_rows

from .config import apply_runtime_profile, load_config
from .orchestration import experiment_dirs, scopes_for_experiment
from .teacher import train_teachers_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--labels-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = apply_runtime_profile(load_config(args.config or None), full=args.full)
    dataset_cfg = config["dataset"]
    labels_root = Path(args.labels_root or (Path(dataset_cfg["output_dir"]) / "family_labels"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "teacher"))
    output_root.mkdir(parents=True, exist_ok=True)

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
