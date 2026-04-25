"""CLI for classical phase prediction baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from .baselines import train_baselines
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--include-linear-svm", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config or None)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["output_dir"])
    label_dir = Path(args.label_dir or dataset_dir.parent / "labels")
    output_dir = Path(args.output_dir or dataset_dir.parent / "baselines")
    baseline_config = dict(config["baselines"])
    if args.include_linear_svm:
        baseline_config["include_linear_svm"] = True
    summary = train_baselines(dataset_dir, label_dir, output_dir, baseline_config)
    print(f"Trained {len(summary)} baselines.")


if __name__ == "__main__":
    main()
