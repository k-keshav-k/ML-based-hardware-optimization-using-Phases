#!/usr/bin/env python3
"""Paired execution-group bootstrap comparison for two phase predictors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.io_utils import load_csv_rows, write_json
from phase_family_ml.metrics import grouped_bootstrap_paired_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--first-model", required=True)
    parser.add_argument("--second-model", required=True)
    parser.add_argument("--family", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()

    rows = load_csv_rows(Path(args.predictions))
    available_families = sorted({row.get("family", "") for row in rows})
    default_family = (
        "__all_families__"
        if "__all_families__" in available_families
        else (available_families[0] if available_families else "")
    )
    family = args.family or default_family
    candidate_rows = [
        row
        for row in rows
        if row.get("family", "") == family and row.get("model", "") in {args.first_model, args.second_model}
    ]
    report_split = "test" if any(row.get("split", "") == "test" for row in candidate_rows) else "val"

    def index_model(model: str) -> dict[tuple[str, str, str], dict[str, str]]:
        return {
            (row.get("run_id", ""), row.get("core_id", ""), row.get("row_index", "")): row
            for row in rows
            if row.get("model", "") == model
            and row.get("family", "") == family
            and row.get("split", "") == report_split
        }

    first = index_model(args.first_model)
    second = index_model(args.second_model)
    keys = sorted(set(first) & set(second))
    if not keys:
        raise SystemExit("No aligned prediction rows for the requested model/family comparison")
    reference = [first[key] for key in keys]
    result = grouped_bootstrap_paired_difference(
        np.asarray([int(row["y_true_phase_t_plus_1"]) for row in reference]),
        np.asarray([int(first[key]["y_pred_phase_t_plus_1"]) for key in keys]),
        np.asarray([int(second[key]["y_pred_phase_t_plus_1"]) for key in keys]),
        np.asarray([int(row["current_phase"]) for row in reference]),
        np.asarray(
            [row.get("concurrent_group_id", "").strip() or row.get("run_id", "") for row in reference],
            dtype=object,
        ),
        n_resamples=args.resamples,
    )
    payload = {
        "family": family,
        "first_model": args.first_model,
        "second_model": args.second_model,
        "report_split": report_split,
        "aligned_rows": len(keys),
        **result,
    }
    write_json(Path(args.output), payload)
    print(payload)


if __name__ == "__main__":
    main()
