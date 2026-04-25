"""CLI for unsupervised phase label generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .labeling import label_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--method", choices=["fgmm", "2kmeans"], default="")
    parser.add_argument("--split-policy", choices=["run_grouped", "workload_holdout"], default="")
    parser.add_argument("--holdout-workload", default="")
    parser.add_argument("--no-pca", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config or None)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["output_dir"])
    output_dir = Path(args.output_dir or dataset_dir.parent / "labels")
    label_config = dict(config["labeling"])
    if args.method:
        label_config["method"] = args.method
    if args.split_policy:
        label_config["split_policy"] = args.split_policy
    if args.holdout_workload:
        label_config["holdout_workload"] = args.holdout_workload
    if args.no_pca:
        label_config["pca_enabled"] = False
    summary = label_dataset(dataset_dir, output_dir, label_config, config["splits"], int(config["random_seed"]))
    print(f"Labeled phases with {summary['method']} into {summary['phase_count']} phase IDs.")


if __name__ == "__main__":
    main()
