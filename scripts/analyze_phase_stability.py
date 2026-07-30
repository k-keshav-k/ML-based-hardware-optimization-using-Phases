#!/usr/bin/env python3
"""Write k/seed sensitivity for the train-only global phase labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.io_utils import write_csv_rows
from phase_family_ml.labels import parsec_rows
from phase_family_ml.robustness import cluster_stability_rows
from phase_family_ml.splits import build_experiment_splits


def integer_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiment-mode",
        choices=["per_workload_holdout", "pooled_run_group", "config_group_holdout", "leave_one_workload_out", "all"],
        default="config_group_holdout",
    )
    parser.add_argument("--k-values", default="2,3,4,5")
    parser.add_argument("--seeds", default="3,7,11,17,23")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17)
    args = parser.parse_args()

    rows = parsec_rows(Path(args.input))
    output_rows: list[dict[str, object]] = []
    for split in build_experiment_splits(
        rows,
        args.experiment_mode,
        args.train_fraction,
        args.val_fraction,
        args.split_seed,
    ):
        output_rows.extend(
            cluster_stability_rows(
                rows,
                split,
                k_values=integer_list(args.k_values),
                seeds=integer_list(args.seeds),
            )
        )
    write_csv_rows(Path(args.output), output_rows)
    print(f"Wrote {len(output_rows)} clustering-stability rows to {args.output}")


if __name__ == "__main__":
    main()
