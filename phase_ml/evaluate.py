"""CLI for phase-ML evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_predictions
from hpc_phase_analysis.io_utils import listify_csv_argument


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="results/phase_ml/baselines")
    parser.add_argument("--transformer-predictions", default="")
    parser.add_argument("--extra-predictions", default="")
    parser.add_argument("--output-dir", default="results/phase_ml/evaluation")
    args = parser.parse_args()
    files = sorted(Path(args.prediction_dir).glob("*_predictions.csv"))
    if args.transformer_predictions:
        files.append(Path(args.transformer_predictions))
    for item in listify_csv_argument(args.extra_predictions):
        files.append(Path(item))
    summary = evaluate_predictions(files, Path(args.output_dir))
    print(f"Evaluated {len(summary)} prediction files.")


if __name__ == "__main__":
    main()
