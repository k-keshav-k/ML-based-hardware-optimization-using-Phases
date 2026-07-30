#!/usr/bin/env python3
"""Validate a complete MLForSys real-data run and write a hashed evidence manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


DEPLOYABLE_MODELS = {
    "online_current_state_tree_all_families",
    "online_phase_history_tree_all_families",
}


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_counts(workloads: list[str], reps: int, set1_threads: list[int]) -> dict[str, tuple[int, int]]:
    pair_count = len(list(itertools.combinations(workloads, 2)))
    return {
        "set1": (len(workloads) * len(set1_threads) * reps, len(workloads) * len(set1_threads) * reps),
        "set2": (pair_count * 2 * reps, pair_count * reps),
        "set3": (pair_count * 2 * reps, pair_count * reps),
    }


def validate_set(
    data_root: Path,
    set_key: str,
    workloads: list[str],
    reps: int,
    set1_threads: list[int],
) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    set_root = data_root / set_key
    raw_root = set_root / "raw"
    processed_root = set_root / "processed"
    results_root = set_root / "results" / "phase_family_ml"
    manifest_path = raw_root / "phase_family_ml_experiment_manifest.json"
    collection_path = raw_root / "phase_family_ml_experiment_results.json"
    merged_path = processed_root / "merged_interval_dataset.csv"
    quality_path = processed_root / "pmu_collection_quality.csv"
    summary_path = results_root / "phase_detector" / "phase_detector_summary_all.csv"
    required = [manifest_path, collection_path, merged_path, quality_path, summary_path]
    for path in required:
        if not path.is_file():
            errors.append(f"{set_key}: missing {path}")
    if errors:
        return {"set": set_key}, errors

    manifest = read_json(manifest_path)
    collection = read_json(collection_path)
    if not isinstance(manifest, list) or not isinstance(collection, list):
        errors.append(f"{set_key}: collection manifest/results must be JSON arrays")
        return {"set": set_key}, errors

    expected_tasks, expected_groups = expected_counts(workloads, reps, set1_threads)[set_key]
    group_ids = {str(row.get("concurrent_group_id", "")) for row in manifest if isinstance(row, dict)}
    if len(manifest) != expected_tasks:
        errors.append(f"{set_key}: expected {expected_tasks} tasks, found {len(manifest)}")
    if len(group_ids) != expected_groups:
        errors.append(f"{set_key}: expected {expected_groups} execution groups, found {len(group_ids)}")
    if len(collection) != expected_tasks:
        errors.append(f"{set_key}: expected {expected_tasks} collection results, found {len(collection)}")
    failed = [row for row in collection if not isinstance(row, dict) or row.get("returncode") != 0]
    if failed:
        errors.append(f"{set_key}: {len(failed)} tasks have a nonzero or missing perf return code")
    if any(str(row.get("pairing_mode", "")) != "all_combinations" for row in manifest if isinstance(row, dict)):
        errors.append(f"{set_key}: manifest contains a non-publication pairing mode")

    if set_key in {"set2", "set3"}:
        expected_pairs = {tuple(sorted(pair)) for pair in itertools.combinations(workloads, 2)}
        for rep in range(1, reps + 1):
            observed_pairs = {
                tuple(sorted(str(row.get("co_running_workloads", "")).split(",")))
                for row in manifest
                if isinstance(row, dict) and int(row.get("rep", 0)) == rep
            }
            if observed_pairs != expected_pairs:
                errors.append(
                    f"{set_key}: repetition {rep} has {len(observed_pairs)}/{len(expected_pairs)} workload pairs"
                )

    merged = csv_rows(merged_path)
    quality = csv_rows(quality_path)
    summary = csv_rows(summary_path)
    if not merged:
        errors.append(f"{set_key}: merged interval dataset is empty")
    if not quality:
        errors.append(f"{set_key}: PMU collection-quality table is empty")
    if merged:
        required_columns = {"run_id", "concurrent_group_id", "workload", "rep", "pairing_mode"}
        missing_columns = required_columns - set(merged[0])
        if missing_columns:
            errors.append(f"{set_key}: merged dataset misses columns {sorted(missing_columns)}")
        if not any(column.startswith("counter__") for column in merged[0]):
            errors.append(f"{set_key}: merged dataset contains no counter features")

    config_rows = [row for row in summary if row.get("experiment") == "config_group_holdout"]
    config_models = {row.get("model", "") for row in config_rows if row.get("family") == "__all_families__"}
    missing_models = DEPLOYABLE_MODELS - config_models
    if missing_models:
        errors.append(f"{set_key}: grouped results miss models {sorted(missing_models)}")
    loo_experiments = {
        row.get("experiment", "")
        for row in summary
        if row.get("experiment", "").startswith("leave_one_workload_out__")
        and row.get("family") == "__all_families__"
        and row.get("model") in DEPLOYABLE_MODELS
    }
    expected_loo = {f"leave_one_workload_out__{workload}" for workload in workloads}
    if loo_experiments != expected_loo:
        errors.append(f"{set_key}: expected {len(expected_loo)} workload-holdout experiments, found {len(loo_experiments)}")

    paired_files = sorted((set_root / "analysis" / "paired").glob("*.json"))
    expected_paired = 3 + len(workloads)
    if len(paired_files) != expected_paired:
        errors.append(f"{set_key}: expected {expected_paired} paired comparisons, found {len(paired_files)}")

    return {
        "set": set_key,
        "tasks": len(manifest),
        "execution_groups": len(group_ids),
        "merged_intervals": len(merged),
        "quality_rows": len(quality),
        "summary_rows": len(summary),
        "paired_comparisons": len(paired_files),
        "artifacts": {
            str(path.relative_to(data_root)): sha256(path)
            for path in required
        },
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--set1-threads", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    set1_threads = [int(item.strip()) for item in args.set1_threads.split(",") if item.strip()]
    all_errors: list[str] = []
    sets: list[dict[str, object]] = []
    for set_key in ("set1", "set2", "set3"):
        payload, errors = validate_set(data_root, set_key, workloads, args.reps, set1_threads)
        sets.append(payload)
        all_errors.extend(errors)

    report = {
        "status": "valid" if not all_errors else "invalid",
        "data_root": str(data_root),
        "workloads": workloads,
        "repetitions": args.reps,
        "set1_threads": set1_threads,
        "sets": sets,
        "errors": all_errors,
    }
    output = data_root / "publication_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if all_errors:
        raise SystemExit(f"Publication run validation failed with {len(all_errors)} error(s); see {output}")


if __name__ == "__main__":
    main()
