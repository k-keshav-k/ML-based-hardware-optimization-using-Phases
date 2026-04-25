"""Workload descriptors and command builders."""

from __future__ import annotations

from pathlib import Path

from .constants import SYNTHETIC_WORKLOADS


def pick_thread_counts(requested: list[int], available_cpus: list[int]) -> list[int]:
    max_threads = max(1, len(available_cpus))
    filtered = sorted({count for count in requested if 1 <= count <= max_threads})
    return filtered or [1]


def cpu_affinity_for_threads(thread_count: int, available_cpus: list[int]) -> list[int]:
    if not available_cpus:
        return list(range(thread_count))
    return available_cpus[:thread_count]


def synthetic_binary_path(project_root: Path) -> Path:
    return project_root / "synthetic_workloads" / "bin" / "phase_bench"


def synthetic_command(
    binary: Path,
    workload: str,
    threads: int,
    duration_ms: int,
    phase_ms: int,
    phase_log: Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "--workload",
        workload,
        "--threads",
        str(threads),
        "--duration-ms",
        str(duration_ms),
        "--phase-ms",
        str(phase_ms),
    ]
    if phase_log is not None:
        command.extend(["--phase-log", str(phase_log)])
    return command


def build_parsec_command(parsecmgmt: str, benchmark: str, threads: int, input_size: str = "native") -> list[str]:
    return [parsecmgmt, "-a", "run", "-p", benchmark, "-n", str(threads), "-i", input_size]


def build_spec_command(runcpu: str, benchmark: str, threads: int, config: str) -> list[str]:
    return [runcpu, "--action", "run", "--copies", "1", "--threads", str(threads), "--config", config, benchmark]


def default_synthetic_workloads() -> list[str]:
    return SYNTHETIC_WORKLOADS[:]
