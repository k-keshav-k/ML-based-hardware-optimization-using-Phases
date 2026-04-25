"""CLI for building PARSEC phase-ML datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--prediction-horizon", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config or None)
    dataset_config = config["dataset"]
    summary = build_dataset(
        Path(args.input or dataset_config["input_csv"]),
        Path(args.output_dir or dataset_config["output_dir"]),
        args.sequence_length or int(dataset_config["sequence_length"]),
        args.stride or int(dataset_config["stride"]),
        args.prediction_horizon or int(dataset_config["prediction_horizon"]),
    )
    print(f"Built {summary['windows']} windows from {summary['interval_rows']} PARSEC intervals.")


if __name__ == "__main__":
    main()
