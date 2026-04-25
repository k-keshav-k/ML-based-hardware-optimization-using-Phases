"""Collect PARSEC phase traces at fixed CPU frequencies."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from analysis.common import write_json
from hpc_phase_analysis.io_utils import listify_csv_argument


def cpu_dirs() -> list[Path]:
    return sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"))


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def write_file(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def snapshot_state() -> dict[str, dict[str, str]]:
    state = {}
    for root in cpu_dirs():
        state[str(root)] = {
            "scaling_governor": read_file(root / "scaling_governor"),
            "scaling_min_freq": read_file(root / "scaling_min_freq"),
            "scaling_max_freq": read_file(root / "scaling_max_freq"),
            "scaling_setspeed": read_file(root / "scaling_setspeed"),
        }
    return state


def restore_state(state: dict[str, dict[str, str]]) -> None:
    for root_text, values in state.items():
        root = Path(root_text)
        for name in ["scaling_governor", "scaling_min_freq", "scaling_max_freq", "scaling_setspeed"]:
            path = root / name
            value = values.get(name, "")
            if path.exists() and value:
                try:
                    write_file(path, value)
                except OSError:
                    pass


def set_frequency(freq_khz: str) -> None:
    roots = cpu_dirs()
    if not roots:
        raise SystemExit("No cpufreq controls were found under /sys/devices/system/cpu.")
    for root in roots:
        governor = root / "scaling_governor"
        if governor.exists():
            write_file(governor, "userspace")
        for name in ["scaling_min_freq", "scaling_max_freq", "scaling_setspeed"]:
            path = root / name
            if path.exists():
                write_file(path, freq_khz)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequencies-khz", required=True)
    parser.add_argument("--output-root", default="results/dvfs_stress")
    parser.add_argument("--workloads", default="blackscholes,canneal,fluidanimate,freqmine,swaptions,streamcluster")
    parser.add_argument("--sets", default="set1,set2,set3")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--interval-ms", type=int, default=10)
    parser.add_argument("--parsec-input", default="simsmall")
    parser.add_argument("--set1-threads", default="2,4,8")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--hybrid-threads", type=int, default=2)
    parser.add_argument("--core-collection-scope", default="system_wide_physical_core")
    parser.add_argument("--collect-uncore", action="store_true")
    parser.add_argument("--phase-ml-config", default="config/phase_ml_indep_a100.json")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    original = snapshot_state()
    frequencies = listify_csv_argument(args.frequencies_khz)
    completed = []
    try:
        for freq in frequencies:
            set_frequency(freq)
            raw_dir = output_root / f"raw_{freq}khz"
            processed_dir = output_root / f"processed_{freq}khz"
            command = [
                "python",
                "-m",
                "phase_ml.collect_experiment_sets",
                "--sets",
                args.sets,
                "--workloads",
                args.workloads,
                "--reps",
                str(args.reps),
                "--interval-ms",
                str(args.interval_ms),
                "--parsec-input",
                args.parsec_input,
                "--set1-threads",
                args.set1_threads,
                "--group-size",
                str(args.group_size),
                "--hybrid-threads",
                str(args.hybrid_threads),
                "--core-collection-scope",
                args.core_collection_scope,
                "--output-dir",
                str(raw_dir),
            ]
            if args.collect_uncore:
                command.append("--collect-uncore")
            run_command(command)
            run_command(
                [
                    "python",
                    "scripts/merge_runs.py",
                    "--input-dir",
                    str(raw_dir),
                    "--output-dir",
                    str(processed_dir),
                    "--manifest-path",
                    str(raw_dir / "phase_ml_experiment_manifest.json"),
                ]
            )
            artifact_dir = output_root / f"phase_ml_{freq}khz"
            dataset_dir = artifact_dir / "dataset"
            label_dir = artifact_dir / "labels"
            run_command(
                [
                    "python",
                    "-m",
                    "phase_ml.build_dataset",
                    "--config",
                    args.phase_ml_config,
                    "--input",
                    str(processed_dir / "merged_interval_dataset.csv"),
                    "--output-dir",
                    str(dataset_dir),
                ]
            )
            run_command(
                [
                    "python",
                    "-m",
                    "phase_ml.label_phases",
                    "--config",
                    args.phase_ml_config,
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(label_dir),
                ]
            )
            completed.append(
                {
                    "frequency_khz": freq,
                    "raw_dir": str(raw_dir),
                    "processed_dir": str(processed_dir),
                    "artifact_dir": str(artifact_dir),
                    "dataset_dir": str(dataset_dir),
                    "label_dir": str(label_dir),
                }
            )
    finally:
        restore_state(original)
    write_json(output_root / "dvfs_collection_manifest.json", {"frequencies_khz": frequencies, "completed": completed})
    print(f"Collected {len(completed)} frequency settings under {output_root}")


if __name__ == "__main__":
    main()
