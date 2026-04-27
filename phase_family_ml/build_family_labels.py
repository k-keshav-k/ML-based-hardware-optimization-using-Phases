"""CLI for train-split tertile family label generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .labels import build_family_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--threshold-mode", choices=["global", "per_workload", "both"], default="")
    parser.add_argument("--experiment-mode", choices=["per_workload_holdout", "pooled_run_group", "leave_one_workload_out", "all"], default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "family_labels"))
    summaries = build_family_labels(
        input_csv=Path(args.input or dataset_cfg["input_csv"]),
        output_root=output_root,
        horizon=int(args.horizon or dataset_cfg["horizon"]),
        threshold_mode=str(args.threshold_mode or config["families"]["threshold_mode"]),
        experiment_mode=str(args.experiment_mode or config["experiments"]["default_mode"]),
        train_fraction=float(config["splits"]["train_fraction"]),
        val_fraction=float(config["splits"]["val_fraction"]),
        seed=int(config["random_seed"]),
    )
    print(f"Built family labels for {len(summaries)} experiment split(s).")


if __name__ == "__main__":
    main()
