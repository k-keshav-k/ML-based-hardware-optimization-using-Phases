#!/usr/bin/env python3
"""Run workloads and collect perf counter data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.collector import run_workload_capture
from hpc_phase_analysis.constants import (
    DEFAULT_INTERVAL_MS,
    DEFAULT_REPETITIONS,
    DEFAULT_SYNTHETIC_DURATION_MS,
    DEFAULT_SYNTHETIC_PHASE_MS,
    DEFAULT_THREADS,
    PROJECT_ROOT,
    RAW_RESULTS_DIR,
    RESULTS_DIR,
)
from hpc_phase_analysis.detect import detect_platform
from hpc_phase_analysis.events import (
    build_alias_map,
    compute_study_readiness,
    discover_perf_list_output,
    extract_event_aliases,
    supported_event_names,
    supported_uncore_event_specs,
)
from hpc_phase_analysis.io_utils import ensure_dir, listify_csv_argument, utc_now_token, write_json
from hpc_phase_analysis.workloads import (
    build_parsec_command,
    build_spec_command,
    cpu_affinity_for_threads,
    default_synthetic_workloads,
    pick_thread_counts,
    synthetic_binary_path,
    synthetic_command,
)

DEFAULT_PARSEC_STRICT_WORKLOADS = ["blackscholes", "bodytrack", "canneal", "fluidanimate", "freqmine"]
PHASE_ML_COUNTER_FAMILIES = [
    "instructions_retired",
    "branch_instructions",
    "branch_mispredictions",
    "l1d_loads",
    "l1d_stores",
    "llc_references",
    "llc_misses",
    "offcore_demand_data_reads",
]


def parse_modes(value: str) -> tuple[bool, bool]:
    parts = set(listify_csv_argument(value))
    return ("interval" in parts or "both" in parts, "aggregate" in parts or "both" in parts)


def events_for_profile(alias_map: dict[str, dict[str, object]], profile: str) -> list[str]:
    if profile == "default":
        return supported_event_names(alias_map)
    events: list[str] = []
    for family in PHASE_ML_COUNTER_FAMILIES:
        metadata = alias_map.get(family, {})
        if metadata.get("supported") and metadata.get("collection_scope") == "task_local":
            selected = str(metadata.get("selected_event", "")).strip()
            if selected:
                events.append(selected)
    return events


def phase_ml_readiness(alias_map: dict[str, dict[str, object]]) -> tuple[bool, str]:
    confident = {
        family
        for family in PHASE_ML_COUNTER_FAMILIES
        if alias_map.get(family, {}).get("analysis_confident") and alias_map.get(family, {}).get("collection_scope") == "task_local"
    }
    if "instructions_retired" not in confident:
        return False, "Phase-ML profile requires a confident instructions-retired counter for instruction-normalized features."
    behavior_families = confident - {"instructions_retired"}
    if not behavior_families:
        return False, "Phase-ML profile requires at least one confident non-timing behavior counter."
    return True, ""


def ensure_synthetic_binary() -> Path:
    binary = synthetic_binary_path(PROJECT_ROOT)
    if binary.exists():
        return binary
    subprocess.run(["make", "synthetic"], cwd=str(PROJECT_ROOT), check=True)
    return binary


def metadata_for_run(
    run_id: str,
    suite: str,
    workload: str,
    threads: int,
    affinity: list[int],
    alias_map: dict[str, dict[str, object]],
    command: list[str],
    run_dir: Path,
    interval_ms: int,
    collect_interval: bool,
    collect_aggregate: bool,
    duration_ms: int | None,
    phase_ms: int | None,
    input_size: str | None,
    study_readiness: dict[str, object],
    uncore_event_specs: list[dict[str, str]],
    collect_uncore: bool,
    core_collection_scope: str,
    cpu_topology: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "suite": suite,
        "workload": workload,
        "threads": threads,
        "cpu_or_core_id": ",".join(str(cpu) for cpu in affinity),
        "affinity": affinity,
        "command": command,
        "run_dir": str(run_dir),
        "alias_map": alias_map,
        "collection_interval_ms": interval_ms,
        "collect_interval": collect_interval,
        "collect_aggregate": collect_aggregate,
        "requested_duration_ms": duration_ms,
        "requested_phase_ms": phase_ms,
        "requested_input_size": input_size,
        "pmu_study_readiness": study_readiness,
        "uncore_events": uncore_event_specs,
        "collect_uncore": collect_uncore,
        "collection_scope": core_collection_scope,
        "core_collection_scope": core_collection_scope,
        "cpu_topology": cpu_topology,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="synthetic", choices=["synthetic", "parsec", "spec2017"])
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--duration-ms", type=int, default=DEFAULT_SYNTHETIC_DURATION_MS)
    parser.add_argument("--phase-ms", type=int, default=DEFAULT_SYNTHETIC_PHASE_MS)
    parser.add_argument("--modes", default="interval")
    parser.add_argument("--output-dir", default=str(RAW_RESULTS_DIR))
    parser.add_argument("--workloads", default="")
    parser.add_argument("--parsec-input", default="test")
    parser.add_argument("--spec-config", default="")
    parser.add_argument("--pcm-command-template", default="")
    parser.add_argument("--require-confident-counters", action="store_true")
    parser.add_argument("--event-profile", default="default", choices=["default", "phase_ml"])
    parser.add_argument("--core-collection-scope", default="task_local", choices=["task_local", "system_wide_cpu", "system_wide_physical_core"])
    parser.add_argument("--collect-uncore", dest="collect_uncore", action="store_true")
    parser.add_argument("--no-collect-uncore", dest="collect_uncore", action="store_false")
    parser.set_defaults(collect_uncore=None)
    args = parser.parse_args()

    platform = detect_platform()
    perf_list_output = discover_perf_list_output()
    alias_map = build_alias_map(platform, extract_event_aliases(str(perf_list_output["stderr"]) + str(perf_list_output["stdout"])))
    study_readiness = compute_study_readiness(platform, alias_map)
    events = events_for_profile(alias_map, args.event_profile)
    if not events:
        raise SystemExit(f"No supported events found for event profile '{args.event_profile}'. Run scripts/discover_events.py to inspect the environment.")
    if args.require_confident_counters:
        if args.event_profile == "phase_ml":
            ready, reason = phase_ml_readiness(alias_map)
            if not ready:
                raise SystemExit(reason)
        elif not study_readiness["study_ready_core"]:
            raise SystemExit(str(study_readiness["strict_hardware_study_reason"]))

    requested_threads = [int(item) for item in listify_csv_argument(args.threads)] if args.threads else DEFAULT_THREADS
    thread_counts = pick_thread_counts(requested_threads, platform["online_cpus"])
    collect_interval, collect_aggregate = parse_modes(args.modes)
    output_dir = ensure_dir(Path(args.output_dir))
    manifest = []
    default_collect_uncore = bool(study_readiness["study_ready_uncore"])
    collect_uncore = default_collect_uncore if args.collect_uncore is None else bool(args.collect_uncore)
    uncore_event_specs = supported_uncore_event_specs(platform, alias_map) if collect_uncore else []
    if collect_uncore and not study_readiness["study_ready_uncore"]:
        raise SystemExit(str(study_readiness["uncore_reason"]))

    suite = args.suite
    selected_workloads = listify_csv_argument(args.workloads) if args.workloads else []
    if suite == "synthetic":
        binary = ensure_synthetic_binary()
        workloads = selected_workloads or default_synthetic_workloads()
    elif suite == "parsec":
        parsec = platform["parsec"]
        if not parsec["available"]:
            raise SystemExit("PARSEC was not detected on this machine.")
        workloads = selected_workloads or [workload for workload in DEFAULT_PARSEC_STRICT_WORKLOADS if workload in parsec["benchmarks"]]
        if not workloads:
            raise SystemExit("No PARSEC workloads were selected or detected for the strict-study subset.")
    else:
        spec = platform["spec2017"]
        if not spec["available"]:
            raise SystemExit("SPEC CPU2017 was not detected on this machine.")
        if not args.spec_config:
            raise SystemExit("SPEC CPU2017 requires --spec-config.")
        workloads = selected_workloads or spec["benchmarks"][:4]

    for workload in workloads:
        for threads in thread_counts:
            affinity = cpu_affinity_for_threads(threads, platform["online_cpus"])
            affinity_text = ",".join(str(cpu) for cpu in affinity)
            for rep in range(1, args.reps + 1):
                run_id = f"{suite}_{workload}_t{threads}_r{rep}_{utc_now_token()}"
                run_dir = ensure_dir(output_dir / run_id)
                phase_log = run_dir / "phase_log.csv"
                if suite == "synthetic":
                    base_command = synthetic_command(binary, workload, threads, args.duration_ms, args.phase_ms, phase_log if workload == "mixed" else None)
                elif suite == "parsec":
                    base_command = build_parsec_command(platform["parsec"]["parsecmgmt"], workload, threads, args.parsec_input)
                else:
                    base_command = build_spec_command(platform["spec2017"]["runcpu"], workload, threads, args.spec_config)
                command = ["taskset", "-c", affinity_text] + base_command
                requested_duration_ms = args.duration_ms if suite == "synthetic" else None
                requested_phase_ms = args.phase_ms if suite == "synthetic" and workload == "mixed" else None
                metadata = metadata_for_run(
                    run_id,
                    suite,
                    workload,
                    threads,
                    affinity,
                    alias_map,
                    command,
                    run_dir,
                    args.interval_ms,
                    collect_interval,
                    collect_aggregate,
                    requested_duration_ms,
                    requested_phase_ms,
                    args.parsec_input if suite == "parsec" else None,
                    study_readiness,
                    uncore_event_specs,
                    collect_uncore,
                    args.core_collection_scope,
                    dict(platform.get("cpu_topology", {})),
                )
                metadata["event_profile"] = args.event_profile
                metadata["events"] = events
                write_json(run_dir / "metadata.json", metadata)
                run_env = dict(os.environ)
                run_cwd = PROJECT_ROOT
                if suite == "parsec":
                    run_env["PARSECDIR"] = str(platform["parsec"]["root"])
                    run_cwd = Path(platform["parsec"]["root"])
                results = run_workload_capture(
                    run_dir=run_dir,
                    workload_command=command,
                    events=events,
                    interval_ms=args.interval_ms,
                    collect_interval=collect_interval,
                    collect_aggregate=collect_aggregate,
                    cwd=run_cwd,
                    env=run_env,
                    pcm_template=args.pcm_command_template,
                    uncore_events=[spec["event_name"] for spec in uncore_event_specs] if collect_uncore else [],
                    core_collection_scope=args.core_collection_scope,
                    system_wide_cpus=affinity if args.core_collection_scope != "task_local" else None,
                )
                write_json(run_dir / "collection_results.json", results)
                manifest.append(
                    {
                        "run_id": run_id,
                        "suite": suite,
                        "workload": workload,
                        "threads": threads,
                        "run_dir": str(run_dir),
                        "interval": collect_interval,
                        "aggregate": collect_aggregate,
                        "collection_interval_ms": args.interval_ms,
                        "requested_duration_ms": requested_duration_ms,
                        "requested_phase_ms": requested_phase_ms,
                        "requested_input_size": args.parsec_input if suite == "parsec" else None,
                        "affinity": affinity,
                        "collect_uncore": collect_uncore,
                        "core_collection_scope": args.core_collection_scope,
                        "event_profile": args.event_profile,
                        "events": events,
                        "pmu_study_readiness": study_readiness,
                    }
                )

    write_json(Path(RESULTS_DIR) / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "runs": len(manifest),
                "events": events,
                "collect_uncore": collect_uncore,
                "study_ready_core": study_readiness["study_ready_core"],
                "study_ready_uncore": study_readiness["study_ready_uncore"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
