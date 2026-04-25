"""Validation checks for the phase-signature paper artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import FORBIDDEN_FEATURE_PARTS, load_rows, read_json, write_json


def validate(processed_dir: Path, dataset_dir: Path, label_dir: Path, eval_dir: Path, output_path: Path) -> dict[str, object]:
    manifest = read_json(dataset_dir / "feature_manifest.json")
    merge_summary = read_json(processed_dir / "merge_summary.json")
    interval_rows = load_rows(dataset_dir / "interval_features.csv")
    window_rows = load_rows(dataset_dir / "window_metadata.csv")
    label_rows = load_rows(label_dir / "window_labels.csv")
    metrics_rows = load_rows(eval_dir / "model_comparison.csv")
    tensors = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    feature_columns = [str(item) for item in manifest["feature_columns"]]  # type: ignore[index]

    split_by_run: dict[str, set[str]] = defaultdict(set)
    for row in label_rows:
        split_by_run[row["run_id"]].add(row["split"])
    leaking_runs = sorted(run_id for run_id, splits in split_by_run.items() if len(splits) > 1)
    forbidden = [column for column in feature_columns if any(part in column.lower() for part in FORBIDDEN_FEATURE_PARTS)]
    label_window_ids = {int(row["window_id"]) for row in label_rows}
    metadata_window_ids = {int(row["window_id"]) for row in window_rows}
    repetition_counts = Counter(row.get("rep", "") for row in interval_rows if row.get("rep", ""))
    split_counts = Counter(row["split"] for row in label_rows)
    checks = {
        "manifest_interval_rows": int(manifest["interval_rows"]),  # type: ignore[index]
        "actual_interval_rows": len(interval_rows),
        "manifest_windows": int(manifest["windows"]),  # type: ignore[index]
        "actual_window_metadata_rows": len(window_rows),
        "actual_window_label_rows": len(label_rows),
        "tensor_windows": int(tensors["X"].shape[0]),
        "feature_count": len(feature_columns),
        "forbidden_features": forbidden,
        "run_count_with_windows": len(split_by_run),
        "split_counts": dict(split_counts),
        "split_leakage_run_count": len(leaking_runs),
        "split_leakage_runs": leaking_runs[:20],
        "window_id_mismatch_count": len(label_window_ids.symmetric_difference(metadata_window_ids)),
        "merge_manifest_run_count": int(merge_summary.get("manifest_run_count", 0)),  # type: ignore[union-attr]
        "merge_merged_run_count": int(merge_summary.get("merged_run_count", 0)),  # type: ignore[union-attr]
        "repetition_counts": dict(sorted(repetition_counts.items())),
        "model_count": len(metrics_rows),
        "models": [row["model"] for row in metrics_rows],
    }
    checks["passed"] = (
        checks["manifest_interval_rows"] == checks["actual_interval_rows"]
        and checks["manifest_windows"] == checks["actual_window_metadata_rows"]
        and checks["actual_window_metadata_rows"] == checks["tensor_windows"]
        and checks["actual_window_label_rows"] <= checks["actual_window_metadata_rows"]
        and checks["split_leakage_run_count"] == 0
        and checks["window_id_mismatch_count"] == 0
        and not checks["forbidden_features"]
        and checks["merge_manifest_run_count"] == checks["merge_merged_run_count"]
    )
    write_json(output_path, checks)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = validate(Path(args.processed_dir), Path(args.dataset_dir), Path(args.label_dir), Path(args.eval_dir), Path(args.output))
    if not payload["passed"]:
        raise SystemExit(f"Artifact validation failed: {args.output}")
    print(f"Artifact validation passed: {args.output}")


if __name__ == "__main__":
    main()
