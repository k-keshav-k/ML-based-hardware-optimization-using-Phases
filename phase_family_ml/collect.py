"""Collect PARSEC experiment sets for the factorized family-wise pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hpc_phase_analysis.collector import run_workload_capture
from hpc_phase_analysis.constants import DEFAULT_INTERVAL_MS, PROJECT_ROOT
from hpc_phase_analysis.detect import detect_platform
from hpc_phase_analysis.events import (
    build_alias_map,
    compute_study_readiness,
    discover_perf_list_output,
    extract_event_aliases,
    supported_uncore_event_specs,
)
from hpc_phase_analysis.io_utils import ensure_dir, listify_csv_argument, utc_now_token, write_json
from hpc_phase_analysis.workloads import build_parsec_command

EXPERIMENT_SET_NAMES = {
    "set1": "set1_single_process_multithread",
    "set2": "set2_multi_process_single_thread",
    "set3": "set3_hybrid_multi_process_multithread",
}

DEFAULT_PARSEC_STRICT_WORKLOADS = ["blackscholes", "bodytrack", "canneal", "fluidanimate", "freqmine"]
DEFAULT_FINALS_DATASET_ROOT = Path("/scratch/kk6081/finals_dataset")

FAMILY_LM_COUNTER_FAMILIES = [
    "instructions_retired",
    "branch_instructions",
    "branch_mispredictions",
    "l1d_loads",
    "l1d_stores",
    "l2_misses",
    "llc_references",
    "llc_misses",
    "offcore_demand_data_reads",
    "fp_arithmetic",
]


def family_lm_events(alias_map: dict[str, dict[str, object]]) -> list[str]:
    """Select expanded family counters while respecting host support."""

    events: list[str] = []
    for family in FAMILY_LM_COUNTER_FAMILIES:
        metadata = alias_map.get(family, {})
        if metadata.get("supported") and metadata.get("collection_scope") == "task_local":
            selected = str(metadata.get("selected_event", "")).strip()
            if selected:
                events.append(selected)
    return events


def family_lm_readiness(alias_map: dict[str, dict[str, object]]) -> tuple[bool, str]:
    """Require enough counters to build at least one family state robustly."""

    confident = {
        family
        for family in FAMILY_LM_COUNTER_FAMILIES
        if alias_map.get(family, {}).get("analysis_confident") and alias_map.get(family, {}).get("collection_scope") == "task_local"
    }
    if "instructions_retired" not in confident:
        return False, "Family-LM profile requires instructions-retired for normalized features."
    behavior = confident - {"instructions_retired"}
    if not behavior:
        return False, "Family-LM profile requires at least one non-instruction behavior counter."
    return True, ""


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in listify_csv_argument(value)]


def chunked(values: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    if not values:
        return []
    if len(values) == 1:
        return [[values[0]] * max(2, size)]
    groups = [values[index : index + size] for index in range(0, len(values), size)]
    if len(values) > 1 and groups and len(groups[-1]) == 1:
        groups[-1] = groups[-1] + values[: size - 1]
    return groups


def affinity_groups(available_cpus: list[int], thread_counts: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    offset = 0
    fallback = available_cpus or list(range(max(sum(thread_counts), 1)))
    for threads in thread_counts:
        if offset + threads > len(fallback):
            raise ValueError("Not enough online CPUs for the requested concurrent experiment group.")
        groups.append(fallback[offset : offset + threads])
        offset += threads
    return groups


def selected_workloads(platform: dict[str, object], requested: str) -> list[str]:
    parsec = platform["parsec"]
    benchmarks = set(parsec["benchmarks"])
    workloads = listify_csv_argument(requested) if requested else DEFAULT_PARSEC_STRICT_WORKLOADS
    return [workload for workload in workloads if workload in benchmarks]


def build_common_context(args: argparse.Namespace) -> dict[str, object]:
    """Discover PMU context once and reuse for all set tasks."""

    platform = detect_platform()
    if not platform["parsec"]["available"]:
        raise SystemExit("PARSEC was not detected on this machine.")
    perf_list_output = discover_perf_list_output()
    alias_map = build_alias_map(platform, extract_event_aliases(str(perf_list_output["stderr"]) + str(perf_list_output["stdout"])))
    ready, reason = family_lm_readiness(alias_map)
    if not ready:
        raise SystemExit(reason)
    study_readiness = compute_study_readiness(platform, alias_map)
    collect_uncore = bool(study_readiness["study_ready_uncore"]) if args.collect_uncore is None else bool(args.collect_uncore)
    if args.collect_uncore and not collect_uncore:
        raise SystemExit(str(study_readiness["uncore_reason"]))
    return {
        "platform": platform,
        "alias_map": alias_map,
        "events": family_lm_events(alias_map),
        "study_readiness": study_readiness,
        "uncore_event_specs": supported_uncore_event_specs(platform, alias_map) if collect_uncore else [],
        "collect_uncore": collect_uncore,
        "core_collection_scope": args.core_collection_scope,
    }


def make_task(
    *,
    context: dict[str, object],
    output_dir: Path,
    experiment_set: str,
    concurrent_group_id: str,
    process_index: int,
    process_count: int,
    co_running_workloads: list[str],
    workload: str,
    threads: int,
    affinity: list[int],
    parsec_input: str,
    interval_ms: int,
    rep: int,
) -> dict[str, object]:
    """Materialize one executable collection task with metadata."""

    platform = context["platform"]
    unique_suffix = uuid.uuid4().hex[:10]
    run_id = f"parsec_family_{experiment_set}_r{rep}_{concurrent_group_id}_{workload}_p{process_index}_t{threads}_{utc_now_token()}_{unique_suffix}"
    run_dir = output_dir / run_id
    affinity_text = ",".join(str(cpu) for cpu in affinity)
    base_command = build_parsec_command(str(platform["parsec"]["parsecmgmt"]), workload, threads, parsec_input)
    command = ["taskset", "-c", affinity_text] + base_command
    metadata = {
        "run_id": run_id,
        "suite": "parsec",
        "workload": workload,
        "threads": threads,
        "cpu_or_core_id": affinity_text,
        "affinity": affinity,
        "command": command,
        "run_dir": str(run_dir),
        "collection_interval_ms": interval_ms,
        "collect_interval": True,
        "collect_aggregate": False,
        "requested_input_size": parsec_input,
        "pmu_study_readiness": context["study_readiness"],
        "alias_map": context["alias_map"],
        "events": context["events"],
        "event_profile": "phase_family_lm",
        "uncore_events": context["uncore_event_specs"],
        "collect_uncore": context["collect_uncore"],
        "experiment_set": experiment_set,
        "set_key": next((key for key, value in EXPERIMENT_SET_NAMES.items() if value == experiment_set), ""),
        "rep": rep,
        "concurrent_group_id": concurrent_group_id,
        "process_index": process_index,
        "process_count": process_count,
        "co_running_workloads": ",".join(co_running_workloads),
        "collection_scope": context["core_collection_scope"],
        "core_collection_scope": context["core_collection_scope"],
        "cpu_topology": platform.get("cpu_topology", {}),
}
    env = dict(os.environ)
    env["PARSECDIR"] = str(platform["parsec"]["root"])
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "execution_group_key": concurrent_group_id,
        "command": command,
        "events": context["events"],
        "interval_ms": interval_ms,
        "cwd": platform["parsec"]["root"],
        "env": env,
        "uncore_events": [spec["event_name"] for spec in context["uncore_event_specs"]] if context["collect_uncore"] else [],
        "core_collection_scope": context["core_collection_scope"],
        "system_wide_cpus": affinity if context["core_collection_scope"] != "task_local" else [],
    }


def run_one_task(task: dict[str, object]) -> dict[str, object]:
    """Execute one perf capture and persist result metadata."""

    run_dir = ensure_dir(Path(str(task["run_dir"])))
    write_json(run_dir / "metadata.json", task["metadata"])
    results = run_workload_capture(
        run_dir=run_dir,
        workload_command=list(task["command"]),
        events=list(task["events"]),
        interval_ms=int(task["interval_ms"]),
        collect_interval=True,
        collect_aggregate=False,
        cwd=Path(str(task["cwd"])),
        env=dict(task["env"]),
        uncore_events=list(task["uncore_events"]),
        core_collection_scope=str(task["core_collection_scope"]),
        system_wide_cpus=list(task["system_wide_cpus"]) if task["system_wide_cpus"] else None,
    )
    write_json(run_dir / "collection_results.json", results)
    return {
        "run_id": task["metadata"]["run_id"],
        "set_key": task["metadata"].get("set_key", ""),
        "run_dir": str(run_dir),
        "returncode": results.get("interval_returncode"),
        "issues": results.get("issues", []),
    }


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def build_tasks(args: argparse.Namespace, context: dict[str, object]) -> list[dict[str, object]]:
    """Build set1/set2/set3 tasks with the expanded counter profile."""

    platform = context["platform"]
    workloads = selected_workloads(platform, args.workloads)
    if not workloads:
        raise SystemExit("No requested PARSEC workloads were detected.")
    available_cpus = list(platform["online_cpus"])
    sets = listify_csv_argument(args.sets)
    tasks: list[dict[str, object]] = []
    for rep in range(1, args.reps + 1):
        if "set1" in sets:
            output_dir = raw_output_dir_for_set(args, "set1")
            for workload in workloads:
                for threads in parse_int_list(args.set1_threads):
                    [affinity] = affinity_groups(available_cpus, [threads])
                    tasks.append(
                        make_task(
                            context=context,
                            output_dir=output_dir,
                            experiment_set=EXPERIMENT_SET_NAMES["set1"],
                            concurrent_group_id=f"set1_r{rep}_{workload}_t{threads}",
                            process_index=0,
                            process_count=1,
                            co_running_workloads=[workload],
                            workload=workload,
                            threads=threads,
                            affinity=affinity,
                            parsec_input=args.parsec_input,
                            interval_ms=args.interval_ms,
                            rep=rep,
                        )
                    )
        if "set2" in sets:
            output_dir = raw_output_dir_for_set(args, "set2")
            for group_index, group in enumerate(chunked(workloads, args.group_size), start=1):
                if len(group) < 2:
                    continue
                groups = affinity_groups(available_cpus, [1] * len(group))
                concurrent_group_id = f"set2_r{rep}_g{group_index}"
                for process_index, (workload, affinity) in enumerate(zip(group, groups)):
                    tasks.append(
                        make_task(
                            context=context,
                            output_dir=output_dir,
                            experiment_set=EXPERIMENT_SET_NAMES["set2"],
                            concurrent_group_id=concurrent_group_id,
                            process_index=process_index,
                            process_count=len(group),
                            co_running_workloads=group,
                            workload=workload,
                            threads=1,
                            affinity=affinity,
                            parsec_input=args.parsec_input,
                            interval_ms=args.interval_ms,
                            rep=rep,
                        )
                    )
        if "set3" in sets:
            output_dir = raw_output_dir_for_set(args, "set3")
            for group_index, group in enumerate(chunked(workloads, args.group_size), start=1):
                if len(group) < 2:
                    continue
                thread_counts = [args.hybrid_threads] * len(group)
                groups = affinity_groups(available_cpus, thread_counts)
                concurrent_group_id = f"set3_r{rep}_g{group_index}_t{args.hybrid_threads}"
                for process_index, (workload, affinity) in enumerate(zip(group, groups)):
                    tasks.append(
                        make_task(
                            context=context,
                            output_dir=output_dir,
                            experiment_set=EXPERIMENT_SET_NAMES["set3"],
                            concurrent_group_id=concurrent_group_id,
                            process_index=process_index,
                            process_count=len(group),
                            co_running_workloads=group,
                            workload=workload,
                            threads=args.hybrid_threads,
                            affinity=affinity,
                            parsec_input=args.parsec_input,
                            interval_ms=args.interval_ms,
                            rep=rep,
                        )
                    )
    return tasks


def use_finals_layout(args: argparse.Namespace) -> bool:
    return not bool(str(args.output_dir).strip())


def set_root(args: argparse.Namespace, set_key: str) -> Path:
    return ensure_dir(Path(args.dataset_root) / set_key)


def raw_output_dir_for_set(args: argparse.Namespace, set_key: str) -> Path:
    if use_finals_layout(args):
        return ensure_dir(set_root(args, set_key) / "raw")
    return ensure_dir(Path(args.output_dir))


def processed_dir_for_set(args: argparse.Namespace, set_key: str) -> Path:
    return ensure_dir(set_root(args, set_key) / "processed")


def results_dir_for_set(args: argparse.Namespace, set_key: str) -> Path:
    return ensure_dir(set_root(args, set_key) / "results" / "phase_family_ml")


def tasks_by_set(tasks: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        metadata = dict(task["metadata"])
        set_key = str(metadata.get("set_key", "")).strip()
        if set_key:
            grouped.setdefault(set_key, []).append(task)
    return grouped


def write_collection_manifests(args: argparse.Namespace, tasks: list[dict[str, object]]) -> dict[str, Path]:
    if not use_finals_layout(args):
        manifest_path = Path(args.output_dir) / "phase_family_ml_experiment_manifest.json"
        write_json(manifest_path, [task["metadata"] for task in tasks])
        return {"all": manifest_path}
    output: dict[str, Path] = {}
    for set_key, set_tasks in tasks_by_set(tasks).items():
        manifest_path = raw_output_dir_for_set(args, set_key) / "phase_family_ml_experiment_manifest.json"
        write_json(manifest_path, [task["metadata"] for task in set_tasks])
        output[set_key] = manifest_path
    return output


def write_collection_results(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    if not use_finals_layout(args):
        write_json(Path(args.output_dir) / "phase_family_ml_experiment_results.json", results)
        return
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in results:
        set_key = str(item.get("set_key", "")).strip()
        if set_key:
            grouped.setdefault(set_key, []).append(item)
    for set_key, rows in grouped.items():
        write_json(raw_output_dir_for_set(args, set_key) / "phase_family_ml_experiment_results.json", rows)


def run_checked(command: list[str], cwd: Path) -> None:
    print("[postprocess] " + " ".join(command), file=sys.stderr, flush=True)
    result = subprocess.run(command, cwd=str(cwd), text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Postprocess command failed with exit code {result.returncode}: {' '.join(command)}")


def run_set_postprocess(args: argparse.Namespace, set_keys: list[str]) -> None:
    if not use_finals_layout(args) or args.skip_postprocess:
        return
    project_root = PROJECT_ROOT
    for set_key in set_keys:
        raw_dir = raw_output_dir_for_set(args, set_key)
        processed_dir = processed_dir_for_set(args, set_key)
        results_dir = results_dir_for_set(args, set_key)
        manifest_path = raw_dir / "phase_family_ml_experiment_manifest.json"
        run_checked(
            [
                sys.executable,
                "scripts/merge_runs.py",
                "--input-dir",
                str(raw_dir),
                "--output-dir",
                str(processed_dir),
                "--manifest-path",
                str(manifest_path),
            ],
            project_root,
        )
        run_checked(
            [
                sys.executable,
                "-m",
                "phase_family_ml.run_pipeline",
                "--config",
                str(args.pipeline_config),
                "--input",
                str(processed_dir / "merged_interval_dataset.csv"),
                "--output-dir",
                str(results_dir),
                "--experiment-mode",
                str(args.experiment_mode),
            ],
            project_root,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="set1,set2,set3")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--parsec-input", default="test")
    parser.add_argument("--dataset-root", default=str(DEFAULT_FINALS_DATASET_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--set1-threads", default="2,4,8")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--hybrid-threads", type=int, default=2)
    parser.add_argument("--pipeline-config", default=str(PROJECT_ROOT / "config" / "phase_family_ml_defaults.json"))
    parser.add_argument("--experiment-mode", choices=["per_workload_holdout", "pooled_run_group", "config_group_holdout", "leave_one_workload_out", "all"], default="config_group_holdout")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--core-collection-scope", default="task_local", choices=["task_local", "system_wide_cpu", "system_wide_physical_core"])
    parser.add_argument("--collect-uncore", dest="collect_uncore", action="store_true")
    parser.add_argument("--no-collect-uncore", dest="collect_uncore", action="store_false")
    parser.set_defaults(collect_uncore=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    context = build_common_context(args)
    tasks = build_tasks(args, context)
    manifest_paths = write_collection_manifests(args, tasks)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "dataset_root": str(args.dataset_root) if use_finals_layout(args) else "",
                    "manifests": {key: str(path) for key, path in manifest_paths.items()},
                    "postprocess": bool(use_finals_layout(args) and not args.skip_postprocess),
                },
                indent=2,
            )
        )
        return

    results = []
    tasks_by_group: dict[str, list[dict[str, object]]] = {}
    for task in tasks:
        tasks_by_group.setdefault(str(task["execution_group_key"]), []).append(task)
    total_tasks = len(tasks)
    completed = 0
    failed = 0
    started = time.perf_counter()
    print(f"[collection] starting {total_tasks} family-lm tasks across {len(tasks_by_group)} groups", file=sys.stderr, flush=True)
    for group_index, group_tasks in enumerate(tasks_by_group.values(), start=1):
        group_id = str(group_tasks[0]["execution_group_key"]) if group_tasks else ""
        print(f"[collection] group {group_index}/{len(tasks_by_group)}: {group_id}", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=len(group_tasks)) as executor:
            futures = [executor.submit(run_one_task, task) for task in group_tasks]
            for future in as_completed(futures):
                item = future.result()
                results.append(item)
                completed += 1
                if item.get("returncode") not in (0, None):
                    failed += 1
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total_tasks - completed) / rate if rate > 0 else 0.0
                print(
                    f"[collection] {progress_bar(completed, total_tasks)} {completed}/{total_tasks} failed={failed} elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )
    write_collection_results(args, results)
    failed_runs = [item for item in results if item.get("returncode") not in (0, None)]
    if not failed_runs:
        run_set_postprocess(args, sorted(tasks_by_set(tasks).keys()))
    print(
        json.dumps(
            {
                "runs": len(results),
                "failed": len(failed_runs),
                "dataset_root": str(args.dataset_root) if use_finals_layout(args) else "",
                "manifests": {key: str(path) for key, path in manifest_paths.items()},
                "postprocess": bool(use_finals_layout(args) and not args.skip_postprocess and not failed_runs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
