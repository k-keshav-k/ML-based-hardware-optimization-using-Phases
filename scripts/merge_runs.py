#!/usr/bin/env python3
"""Merge raw run directories into wide datasets."""

from __future__ import annotations

import argparse
import bisect
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import METADATA_COLUMNS, PROCESSED_RESULTS_DIR, RAW_RESULTS_DIR, RESULTS_DIR
from hpc_phase_analysis.events import normalize_event_name, reverse_alias_map
from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, safe_float, write_csv_rows, write_json

EXPERIMENT_METADATA_COLUMNS = [
    "experiment_set",
    "rep",
    "requested_input_size",
    "concurrent_group_id",
    "process_index",
    "process_count",
    "co_running_workloads",
    "collection_scope",
    "core_collection_scope",
]


def attach_experiment_metadata(item: dict[str, object], metadata: dict[str, object]) -> None:
    # Preserve experiment-set fields so downstream training can stratify by scenario.
    for column in EXPERIMENT_METADATA_COLUMNS:
        if column in metadata:
            item[column] = metadata.get(column, "")


def load_phase_rows(path: Path) -> list[dict[str, str]]:
    return load_csv_rows(path) if path.exists() else []


def phase_for_timestamp(phase_rows: list[dict[str, str]], timestamp_ms: float) -> str:
    for row in phase_rows:
        start = safe_float(row.get("start_ms", ""))
        end = safe_float(row.get("end_ms", ""))
        if start <= timestamp_ms < end:
            return row.get("phase_label", "")
    return ""


def interval_durations(rows: list[dict[str, object]]) -> list[float]:
    # Infer interval duration from timestamp deltas; fallback to median positive delta.
    timestamps = [safe_float(row.get("timestamp_ms", "")) for row in rows]
    deltas = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
    positive = [delta for delta in deltas if delta > 0]
    fallback = float(statistics.median(positive)) if positive else math.nan
    durations: list[float] = []
    for index, timestamp in enumerate(timestamps):
        if index == 0:
            durations.append(fallback)
            continue
        delta = timestamp - timestamps[index - 1]
        if delta <= 0 and not math.isnan(fallback):
            delta = fallback
        durations.append(delta)
    return durations


def physical_core_id_for_cpu(metadata: dict[str, object], cpu_or_core_id: str) -> str:
    topology = metadata.get("cpu_topology", {})
    if isinstance(topology, dict):
        cpu_info = topology.get(str(cpu_or_core_id), {})
        if isinstance(cpu_info, dict):
            value = str(cpu_info.get("physical_core_id", "")).strip()
            if value:
                return value
    return str(cpu_or_core_id)


def logical_cpu_row_identity(row: dict[str, str], metadata: dict[str, object]) -> tuple[str, str, str]:
    # Normalize row identity depending on task-local vs system-wide collection scope.
    row_cpu = str(row.get("cpu_or_core_id", "")).strip()
    if not row_cpu:
        return str(metadata["cpu_or_core_id"]), "", "task_local"
    physical_core_id = physical_core_id_for_cpu(metadata, row_cpu)
    mode = str(metadata.get("core_collection_scope", "system_wide_cpu"))
    if mode == "system_wide_physical_core":
        return physical_core_id, physical_core_id, mode
    return row_cpu, physical_core_id, mode


def assign_interval_durations(rows: list[dict[str, object]]) -> None:
    by_unit: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = str(row.get("cpu_or_core_id", ""))
        by_unit.setdefault(key, []).append(row)
    for unit_rows in by_unit.values():
        unit_rows.sort(key=lambda item: safe_float(item.get("timestamp_ms", "")))
        for row, duration in zip(unit_rows, interval_durations(unit_rows)):
            row["interval_duration_ms"] = "" if math.isnan(duration) else duration


def aggregate_uncore_interval_rows(run_dir: Path, metadata: dict[str, object]) -> list[dict[str, object]]:
    # Build a timestamp-aligned uncore table keyed by normalized family names.
    uncore_specs = metadata.get("uncore_events", [])
    if not isinstance(uncore_specs, list):
        return []
    event_to_family = {}
    for spec in uncore_specs:
        if not isinstance(spec, dict):
            continue
        event_name = str(spec.get("event_name", "")).strip()
        family = str(spec.get("family", "")).strip()
        if event_name and family:
            event_to_family[normalize_event_name(event_name)] = family
    if not event_to_family:
        return []
    perf_path = run_dir / "uncore_interval_perf.csv"
    if not perf_path.exists():
        return []
    grouped: dict[str, dict[str, object]] = {}
    for row in load_csv_rows(perf_path):
        timestamp = safe_float(row.get("timestamp_ms", ""))
        event_name = normalize_event_name(row.get("event_name", ""))
        family = event_to_family.get(event_name)
        if not family:
            continue
        key = f"{timestamp:.3f}"
        item = grouped.setdefault(key, {"timestamp_ms": timestamp})
        column = f"counter__{family}"
        current = safe_float(item.get(column, ""))
        value = safe_float(row.get("value", ""))
        if math.isnan(value):
            continue
        if math.isnan(current):
            item[column] = value
        else:
            item[column] = current + value
    rows = sorted(grouped.values(), key=lambda item: safe_float(item.get("timestamp_ms", "")))
    for row in rows:
        read_value = safe_float(row.get("counter__memory_read_bandwidth", ""))
        write_value = safe_float(row.get("counter__memory_write_bandwidth", ""))
        if not math.isnan(read_value) or not math.isnan(write_value):
            row["counter__total_memory_bandwidth"] = (0.0 if math.isnan(read_value) else read_value) + (
                0.0 if math.isnan(write_value) else write_value
            )
    return rows


def align_uncore_rows(
    core_rows: list[dict[str, object]],
    uncore_rows: list[dict[str, object]],
) -> int:
    # Attach nearest uncore interval sample to each core row within tolerance.
    if not core_rows or not uncore_rows:
        return 0
    uncore_timestamps = [safe_float(row.get("timestamp_ms", "")) for row in uncore_rows]
    positive = [timestamp for timestamp in uncore_timestamps if not math.isnan(timestamp)]
    if not positive:
        return 0
    positive_deltas = [positive[index] - positive[index - 1] for index in range(1, len(positive)) if positive[index] > positive[index - 1]]
    tolerance = float(statistics.median(positive_deltas)) if positive_deltas else 15.0
    attached = 0
    for core_row in core_rows:
        core_timestamp = safe_float(core_row.get("timestamp_ms", ""))
        if math.isnan(core_timestamp):
            continue
        index = bisect.bisect_left(uncore_timestamps, core_timestamp)
        candidates = []
        if index < len(uncore_rows):
            candidates.append(uncore_rows[index])
        if index > 0:
            candidates.append(uncore_rows[index - 1])
        if not candidates:
            continue
        closest = min(
            candidates,
            key=lambda item: abs(safe_float(item.get("timestamp_ms", "")) - core_timestamp),
        )
        closest_timestamp = safe_float(closest.get("timestamp_ms", ""))
        if math.isnan(closest_timestamp) or abs(closest_timestamp - core_timestamp) > tolerance:
            continue
        for key, value in closest.items():
            if key.startswith("counter__"):
                core_row[key] = value
        attached += 1
    return attached


def merge_interval_rows(run_dir: Path) -> tuple[list[dict[str, object]], int]:
    # Core merge path: parse interval perf rows into one wide row per (timestamp, unit).
    metadata = read_json(run_dir / "metadata.json")
    alias_map = metadata["alias_map"]
    reverse = reverse_alias_map(alias_map)
    phase_rows = load_phase_rows(run_dir / "phase_log.csv")
    perf_rows = load_csv_rows(run_dir / "interval_perf.csv")
    grouped: dict[str, dict[str, object]] = {}
    for row in perf_rows:
        timestamp = safe_float(row.get("timestamp_ms", ""))
        unit_id, physical_core_id, core_collection_scope = logical_cpu_row_identity(row, metadata)
        key = f"{timestamp:.3f}|{unit_id}"
        item = grouped.setdefault(
            key,
            {
                "timestamp_ms": timestamp,
                "interval_duration_ms": "",
                "workload": metadata["workload"],
                "suite": metadata["suite"],
                "run_id": metadata["run_id"],
                "threads": metadata["threads"],
                "cpu_or_core_id": unit_id,
                "physical_core_id": physical_core_id,
                "collection_scope": core_collection_scope,
                "phase_label": phase_for_timestamp(phase_rows, timestamp),
            },
        )
        attach_experiment_metadata(item, metadata)
        item["collection_scope"] = core_collection_scope
        item["physical_core_id"] = physical_core_id
        family = reverse.get(normalize_event_name(row.get("event_name", "")))
        if family:
            column = f"counter__{family}"
            current = safe_float(item.get(column, ""))
            value = safe_float(row.get("value", ""))
            if math.isnan(value):
                continue
            item[column] = value if math.isnan(current) else current + value
    merged_rows = sorted(grouped.values(), key=lambda item: (safe_float(item.get("timestamp_ms", "")), str(item.get("cpu_or_core_id", ""))))
    assign_interval_durations(merged_rows)
    attached_uncore_rows = align_uncore_rows(merged_rows, aggregate_uncore_interval_rows(run_dir, metadata))
    return merged_rows, attached_uncore_rows


def merge_aggregate_rows(run_dir: Path) -> list[dict[str, object]]:
    metadata = read_json(run_dir / "metadata.json")
    alias_map = metadata["alias_map"]
    reverse = reverse_alias_map(alias_map)
    perf_path = run_dir / "aggregate_perf.csv"
    if not perf_path.exists():
        return []
    grouped: dict[str, dict[str, object]] = {}
    for row in load_csv_rows(perf_path):
        unit_id, physical_core_id, core_collection_scope = logical_cpu_row_identity(row, metadata)
        item = grouped.setdefault(
            unit_id,
            {
                "timestamp_ms": "",
                "interval_duration_ms": "",
                "workload": metadata["workload"],
                "suite": metadata["suite"],
                "run_id": metadata["run_id"],
                "threads": metadata["threads"],
                "cpu_or_core_id": unit_id,
                "physical_core_id": physical_core_id,
                "collection_scope": core_collection_scope,
                "phase_label": "",
            },
        )
        attach_experiment_metadata(item, metadata)
        item["collection_scope"] = core_collection_scope
        item["physical_core_id"] = physical_core_id
        family = reverse.get(normalize_event_name(row.get("event_name", "")))
        if family:
            column = f"counter__{family}"
            current = safe_float(item.get(column, ""))
            value = safe_float(row.get("value", ""))
            if math.isnan(value):
                continue
            item[column] = value if math.isnan(current) else current + value
    return sorted(grouped.values(), key=lambda item: str(item.get("cpu_or_core_id", "")))


def resolve_run_directories(input_dir: Path, manifest_path: Path, include_all_raw_runs: bool) -> tuple[list[Path], dict[str, object]]:
    # Prefer manifest-driven merging to avoid stale directories from prior runs.
    all_run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if include_all_raw_runs or not manifest_path.exists():
        return all_run_dirs, {
            "manifest_path": str(manifest_path),
            "manifest_run_count": 0,
            "merged_run_count": len(all_run_dirs),
            "filtered_stale_run_count": 0,
            "manifest_used": False,
        }

    manifest = read_json(manifest_path)
    manifest_entries = manifest if isinstance(manifest, list) else []
    requested_dirs = []
    for entry in manifest_entries:
        run_dir = Path(str(entry.get("run_dir", "")))
        if run_dir.exists() and run_dir.is_dir():
            requested_dirs.append(run_dir)
    deduped = sorted({path.resolve(): path for path in requested_dirs}.values())
    all_resolved = {path.resolve() for path in all_run_dirs}
    filtered = [path for path in deduped if path.resolve() in all_resolved]
    return filtered, {
        "manifest_path": str(manifest_path),
        "manifest_run_count": len(manifest_entries),
        "merged_run_count": len(filtered),
        "filtered_stale_run_count": max(0, len(all_run_dirs) - len(filtered)),
        "manifest_used": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(RAW_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_RESULTS_DIR))
    parser.add_argument("--manifest-path", default=str(RESULTS_DIR / "run_manifest.json"))
    parser.add_argument("--include-all-raw-runs", action="store_true")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    # Stage order: select run dirs -> merge interval rows -> merge aggregate rows -> write summary.
    interval_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    uncore_aligned_rows = 0
    run_dirs, provenance = resolve_run_directories(input_dir, Path(args.manifest_path), args.include_all_raw_runs)
    for run_dir in run_dirs:
        merged_rows, attached = merge_interval_rows(run_dir)
        interval_rows.extend(merged_rows)
        uncore_aligned_rows += attached
        aggregate_rows.extend(merge_aggregate_rows(run_dir))

    write_csv_rows(output_dir / "merged_interval_dataset.csv", interval_rows)
    write_csv_rows(output_dir / "merged_aggregate_dataset.csv", aggregate_rows)
    summary = {
        "run_count": len(run_dirs),
        "manifest_run_count": provenance["manifest_run_count"],
        "merged_run_count": provenance["merged_run_count"],
        "manifest_path": provenance["manifest_path"],
        "filtered_stale_run_count": provenance["filtered_stale_run_count"],
        "manifest_used": provenance["manifest_used"],
        "interval_rows": len(interval_rows),
        "aggregate_rows": len(aggregate_rows),
        "uncore_aligned_interval_rows": uncore_aligned_rows,
        "observed_system_wide_uncore_columns": sorted(
            {
                key
                for row in interval_rows
                for key in row.keys()
                if key.startswith("counter__memory_") or key == "counter__total_memory_bandwidth"
            }
        ),
        "metadata_columns": METADATA_COLUMNS,
        "interval_dataset": str(output_dir / "merged_interval_dataset.csv"),
        "aggregate_dataset": str(output_dir / "merged_aggregate_dataset.csv"),
    }
    write_json(output_dir / "merge_summary.json", summary)
    print(f"Merged {len(run_dirs)} runs into {summary['interval_rows']} interval rows.")


if __name__ == "__main__":
    main()
