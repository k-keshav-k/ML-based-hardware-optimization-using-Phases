"""Run live co-scheduling placement experiments for PARSEC workloads.

The driver measures real wall time for solo and co-run executions. It does not
invent a speedup number: analysis derives speedup/slowdown only from the solo
baselines collected by this same script.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from analysis.common import load_rows, phase_resource_classes, write_json, write_rows
from hpc_phase_analysis.detect import detect_platform
from hpc_phase_analysis.io_utils import listify_csv_argument
from hpc_phase_analysis.workloads import build_parsec_command


POLICIES = [
    "linux_default",
    "random_placement",
    "spread_physical",
    "static_workload",
    "phase_aware_oracle",
    "phase_aware_student",
    "phase_aware_transformer",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in listify_csv_argument(value)]


def physical_cpu_order(platform: dict[str, object]) -> list[int]:
    topology = dict(platform.get("cpu_topology", {}))
    by_core: dict[str, list[int]] = defaultdict(list)
    for cpu_text, item in topology.items():
        cpu = int(cpu_text)
        key = str(dict(item).get("physical_core_id", cpu_text))
        by_core[key].append(cpu)
    if not by_core:
        return [int(cpu) for cpu in platform.get("online_cpus", [])]
    return [min(cpus) for _, cpus in sorted(by_core.items())]


def cpu_order_by_package(platform: dict[str, object]) -> dict[str, list[int]]:
    topology = dict(platform.get("cpu_topology", {}))
    packages: dict[str, list[int]] = defaultdict(list)
    for cpu in physical_cpu_order(platform):
        item = dict(topology.get(str(cpu), {}))
        package = str(item.get("physical_package_id", "0"))
        packages[package].append(cpu)
    return dict(packages)


def allocate_disjoint(cpus: list[int], process_count: int, threads: int) -> list[list[int]]:
    need = process_count * threads
    if len(cpus) < need:
        raise SystemExit(f"Need {need} logical CPUs for this run, but only {len(cpus)} suitable CPUs were detected.")
    return [cpus[index * threads : (index + 1) * threads] for index in range(process_count)]


def load_workload_classes(phase_artifact: Path, prediction_name: str = "") -> dict[str, str]:
    dataset_dir = phase_artifact / "dataset"
    label_dir = phase_artifact / "labels"
    interval_rows = load_rows(dataset_dir / "interval_features.csv")
    window_rows = load_rows(label_dir / "window_labels.csv")
    class_by_phase = phase_resource_classes(interval_rows, window_rows)
    prediction_files = {
        "student": phase_artifact / "student_tree" / "student_decision_tree_predictions.csv",
        "transformer": phase_artifact / "transformer" / "transformer_predictions.csv",
    }
    pred_by_window: dict[int, int] = {}
    if prediction_name:
        path = prediction_files[prediction_name]
        if path.exists():
            pred_by_window = {int(row["window_id"]): int(row["y_pred_next_phase"]) for row in load_rows(path)}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in window_rows:
        phase = pred_by_window.get(int(row["window_id"]), int(row["phase_id"]))
        counts[row["workload"]][class_by_phase.get(phase, "balanced")] += 1
    return {workload: counter.most_common(1)[0][0] for workload, counter in counts.items() if counter}


def placement_for_policy(
    policy: str,
    workloads: list[str],
    threads: int,
    platform: dict[str, object],
    classes: dict[str, str],
    rng: random.Random,
) -> tuple[list[list[int]] | None, str]:
    if policy == "linux_default":
        return None, "linux_default_no_affinity"
    physical = physical_cpu_order(platform)
    if policy == "random_placement":
        shuffled = physical[:]
        rng.shuffle(shuffled)
        return allocate_disjoint(shuffled, len(workloads), threads), "random_disjoint_physical"
    packages = cpu_order_by_package(platform)
    if policy in {"static_workload", "phase_aware_oracle", "phase_aware_student", "phase_aware_transformer"} and len(packages) > 1:
        heavy = {"memory", "mixed"}
        ordered = sorted(workloads, key=lambda name: (classes.get(name, "balanced") not in heavy, name))
        placement: dict[str, list[int]] = {}
        package_ids = sorted(packages)
        offsets = {package: 0 for package in package_ids}
        for index, workload in enumerate(ordered):
            package = package_ids[index % len(package_ids)]
            start = offsets[package]
            end = start + threads
            if end > len(packages[package]):
                return allocate_disjoint(physical, len(workloads), threads), "phase_aware_fallback_single_pool"
            placement[workload] = packages[package][start:end]
            offsets[package] = end
        return [placement[name] for name in workloads], "phase_aware_package_spread"
    return allocate_disjoint(physical, len(workloads), threads), "spread_disjoint_physical"


def run_group(
    *,
    parsecmgmt: str,
    parsec_root: Path,
    output_dir: Path,
    run_kind: str,
    policy: str,
    rep: int,
    group_id: str,
    workloads: list[str],
    threads: int,
    parsec_input: str,
    affinities: list[list[int]] | None,
    classes: dict[str, str],
    placement_note: str,
    dry_run: bool,
) -> list[dict[str, object]]:
    env = dict(os.environ)
    env["PARSECDIR"] = str(parsec_root)
    run_dir = output_dir / "runs" / f"{run_kind}_{policy}_r{rep}_{group_id}_t{threads}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if dry_run:
        planned = []
        for index, workload in enumerate(workloads):
            base = build_parsec_command(parsecmgmt, workload, threads, parsec_input)
            affinity = affinities[index] if affinities is not None else []
            command = ["taskset", "-c", ",".join(str(cpu) for cpu in affinity)] + base if affinity else base
            planned.append(
                {
                    "run_id": run_dir.name,
                    "run_kind": run_kind,
                    "policy": policy,
                    "rep": rep,
                    "group_id": group_id,
                    "process_index": index,
                    "process_count": len(workloads),
                    "workload": workload,
                    "threads": threads,
                    "affinity": ",".join(str(cpu) for cpu in affinity),
                    "resource_class": classes.get(workload, "balanced"),
                    "co_running_workloads": ",".join(workloads),
                    "command": " ".join(command),
                    "cwd": str(parsec_root),
                    "start_time_s": "",
                    "end_time_s": "",
                    "duration_s": "",
                    "group_duration_s": "",
                    "returncode": "",
                    "stdout_path": str(run_dir / f"p{index}_{workload}.out"),
                    "stderr_path": str(run_dir / f"p{index}_{workload}.err"),
                    "placement_note": placement_note,
                }
            )
        return planned

    processes: list[tuple[str, subprocess.Popen[str], float, Path, Path, list[str], list[int], object, object]] = []
    start_group = time.perf_counter()
    for index, workload in enumerate(workloads):
        base = build_parsec_command(parsecmgmt, workload, threads, parsec_input)
        affinity = affinities[index] if affinities is not None else []
        command = ["taskset", "-c", ",".join(str(cpu) for cpu in affinity)] + base if affinity else base
        stdout_path = run_dir / f"p{index}_{workload}.out"
        stderr_path = run_dir / f"p{index}_{workload}.err"
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=str(parsec_root), env=env, stdout=stdout, stderr=stderr, text=True)
        processes.append((workload, process, started, stdout_path, stderr_path, command, affinity, stdout, stderr))
    for index, (workload, process, started, stdout_path, stderr_path, command, affinity, stdout, stderr) in enumerate(processes):
        returncode = process.wait()
        stdout.close()
        stderr.close()
        ended = time.perf_counter()
        rows.append(
            {
                "run_id": run_dir.name,
                "run_kind": run_kind,
                "policy": policy,
                "rep": rep,
                "group_id": group_id,
                "process_index": index,
                "process_count": len(workloads),
                "workload": workload,
                "threads": threads,
                "affinity": ",".join(str(cpu) for cpu in affinity),
                "resource_class": classes.get(workload, "balanced"),
                "co_running_workloads": ",".join(workloads),
                "command": " ".join(command),
                "cwd": str(parsec_root),
                "start_time_s": started,
                "end_time_s": ended,
                "duration_s": ended - started,
                "group_duration_s": ended - start_group,
                "returncode": returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "placement_note": placement_note,
            }
        )
    return rows


def workload_groups(workloads: list[str], group_size: int, mode: str) -> list[list[str]]:
    if mode == "all_pairs":
        return [list(group) for group in itertools.combinations(workloads, group_size)]
    return [workloads[index : index + group_size] for index in range(0, len(workloads), group_size) if len(workloads[index : index + group_size]) == group_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-artifact", default="results/phase_ml_core_uncore_large")
    parser.add_argument("--output-dir", default="results/online_coscheduling")
    parser.add_argument("--workloads", default="blackscholes,canneal,fluidanimate,freqmine,swaptions,streamcluster")
    parser.add_argument("--threads", default="1")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--group-mode", choices=["all_pairs", "chunked"], default="all_pairs")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--parsec-input", default="simsmall")
    parser.add_argument("--skip-solo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    platform = detect_platform()
    parsec = dict(platform["parsec"])
    if not parsec.get("available"):
        raise SystemExit("PARSEC was not detected. Set PARSEC_DIR to the PARSEC root before running this experiment.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = listify_csv_argument(args.workloads)
    threads_list = parse_int_list(args.threads)
    policies = listify_csv_argument(args.policies)
    unknown = [policy for policy in policies if policy not in POLICIES]
    if unknown:
        raise SystemExit("Unknown policies: " + ",".join(unknown))

    class_maps = {
        "oracle": load_workload_classes(Path(args.phase_artifact)),
        "student": load_workload_classes(Path(args.phase_artifact), "student"),
        "transformer": load_workload_classes(Path(args.phase_artifact), "transformer"),
    }
    rows: list[dict[str, object]] = []
    rng = random.Random(20260425)
    for threads in threads_list:
        if not args.skip_solo:
            for rep in range(1, args.reps + 1):
                for workload in workloads:
                    affinities, note = placement_for_policy("spread_physical", [workload], threads, platform, class_maps["oracle"], rng)
                    rows.extend(
                        run_group(
                            parsecmgmt=str(parsec["parsecmgmt"]),
                            parsec_root=Path(str(parsec["root"])),
                            output_dir=output_dir,
                            run_kind="solo",
                            policy="solo_spread",
                            rep=rep,
                            group_id=workload,
                            workloads=[workload],
                            threads=threads,
                            parsec_input=args.parsec_input,
                            affinities=affinities,
                            classes=class_maps["oracle"],
                            placement_note=note,
                            dry_run=args.dry_run,
                        )
                    )
        groups = workload_groups(workloads, args.group_size, args.group_mode)
        for rep in range(1, args.reps + 1):
            for group_index, group in enumerate(groups, start=1):
                for policy in policies:
                    class_key = "oracle"
                    if policy == "phase_aware_student":
                        class_key = "student"
                    elif policy == "phase_aware_transformer":
                        class_key = "transformer"
                    affinities, note = placement_for_policy(policy, group, threads, platform, class_maps[class_key], rng)
                    rows.extend(
                        run_group(
                            parsecmgmt=str(parsec["parsecmgmt"]),
                            parsec_root=Path(str(parsec["root"])),
                            output_dir=output_dir,
                            run_kind="corun",
                            policy=policy,
                            rep=rep,
                            group_id=f"g{group_index}_{'_'.join(group)}",
                            workloads=group,
                            threads=threads,
                            parsec_input=args.parsec_input,
                            affinities=affinities,
                            classes=class_maps[class_key],
                            placement_note=note,
                            dry_run=args.dry_run,
                        )
                    )
    fieldnames = [
        "run_id",
        "run_kind",
        "policy",
        "rep",
        "group_id",
        "process_index",
        "process_count",
        "workload",
        "threads",
        "affinity",
        "resource_class",
        "co_running_workloads",
        "command",
        "cwd",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "group_duration_s",
        "returncode",
        "stdout_path",
        "stderr_path",
        "placement_note",
    ]
    write_rows(output_dir / "online_runs.csv", rows, fieldnames)
    write_json(
        output_dir / "online_manifest.json",
        {
            "status": "dry_run" if args.dry_run else "completed",
            "rows": len(rows),
            "policies": policies,
            "workloads": workloads,
            "threads": threads_list,
            "parsec_input": args.parsec_input,
            "platform": {
                "cpu_model": platform.get("cpu_model", ""),
                "logical_cpus": platform.get("logical_cpus", ""),
                "socket_count": platform.get("socket_count", ""),
                "cores_per_socket": platform.get("cores_per_socket", ""),
            },
        },
    )
    row_kind = "planned" if args.dry_run else "measured"
    print(f"Wrote {len(rows)} {row_kind} rows to {output_dir / 'online_runs.csv'}")


if __name__ == "__main__":
    main()
