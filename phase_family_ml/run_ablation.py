"""CLI for family counter ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows

from .ablation import run_ablation
from .config import apply_runtime_profile, load_config
from .orchestration import experiment_dirs, scopes_for_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    config = apply_runtime_profile(load_config(args.config or None), full=args.full)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    output_dir = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            temp_csv = output_dir / f"ablation_{exp_dir.name}_{scope}.csv"
            rows = run_ablation(
                input_csv=Path(args.input or dataset_cfg["input_csv"]),
                experiment_dir=exp_dir,
                scope=scope,
                horizon=int(dataset_cfg["horizon"]),
                output_csv=temp_csv,
                weights=dict(config["ablation"]["score_weights"]),
                tree_max_depth=int(config["ablation"]["tree_max_depth"]),
                tree_min_samples_leaf=int(config["ablation"]["tree_min_samples_leaf"]),
                run_global_exhaustive=bool(config["ablation"].get("global_exhaustive_one_per_family", True)),
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                all_rows.append(item)
    write_csv_rows(output_dir / "family_ablation_results.csv", all_rows)
    print(f"Wrote {len(all_rows)} ablation rows.")


if __name__ == "__main__":
    main()
