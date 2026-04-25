"""Perf-based counter collection helpers."""

from __future__ import annotations

import csv
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from .events import normalize_event_name
from .io_utils import ensure_dir, safe_float


def clean_perf_event_name(event: str) -> str:
    value = event.strip()
    if value.endswith("]") and " [" in value:
        return value.rsplit(" [", 1)[0].strip()
    return value


def parse_perf_csv(text: str, interval_mode: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader = csv.reader(text.splitlines())
    for fields in reader:
        if not fields:
            continue
        if interval_mode and len(fields) >= 4:
            timestamp_s = safe_float(fields[0])
            if len(fields) >= 5 and fields[1].strip().startswith("CPU"):
                cpu_or_core_id = fields[1].strip().removeprefix("CPU")
                value = safe_float(fields[2])
                unit = fields[3].strip()
                event = clean_perf_event_name(fields[4].strip())
                metric_offset = 5
            else:
                cpu_or_core_id = ""
                value = safe_float(fields[1])
                unit = fields[2].strip()
                event = clean_perf_event_name(fields[3].strip())
                metric_offset = 4
            runtime_ns = safe_float(fields[metric_offset]) if len(fields) > metric_offset else float("nan")
            enabled_pct = safe_float(fields[metric_offset + 1]) if len(fields) > metric_offset + 1 else float("nan")
            metric_value = safe_float(fields[metric_offset + 2]) if len(fields) > metric_offset + 2 else float("nan")
            metric_unit = fields[metric_offset + 3].strip() if len(fields) > metric_offset + 3 else ""
            rows.append(
                {
                    "timestamp_ms": timestamp_s * 1000.0,
                    "cpu_or_core_id": cpu_or_core_id,
                    "value": value,
                    "unit": unit,
                    "event_name": event,
                    "event_name_normalized": normalize_event_name(event),
                    "runtime_ns": runtime_ns,
                    "enabled_pct": enabled_pct,
                    "metric_value": metric_value,
                    "metric_unit": metric_unit,
                    "status": "ok" if event else "unknown",
                }
            )
        elif not interval_mode and len(fields) >= 3:
            if len(fields) >= 4 and fields[0].strip().startswith("CPU"):
                cpu_or_core_id = fields[0].strip().removeprefix("CPU")
                value = safe_float(fields[1])
                unit = fields[2].strip()
                event = clean_perf_event_name(fields[3].strip())
                metric_offset = 4
            else:
                cpu_or_core_id = ""
                value = safe_float(fields[0])
                unit = fields[1].strip()
                event = clean_perf_event_name(fields[2].strip())
                metric_offset = 3
            runtime_ns = safe_float(fields[metric_offset]) if len(fields) > metric_offset else float("nan")
            enabled_pct = safe_float(fields[metric_offset + 1]) if len(fields) > metric_offset + 1 else float("nan")
            metric_value = safe_float(fields[metric_offset + 2]) if len(fields) > metric_offset + 2 else float("nan")
            metric_unit = fields[metric_offset + 3].strip() if len(fields) > metric_offset + 3 else ""
            rows.append(
                {
                    "timestamp_ms": "",
                    "cpu_or_core_id": cpu_or_core_id,
                    "value": value,
                    "unit": unit,
                    "event_name": event,
                    "event_name_normalized": normalize_event_name(event),
                    "runtime_ns": runtime_ns,
                    "enabled_pct": enabled_pct,
                    "metric_value": metric_value,
                    "metric_unit": metric_unit,
                    "status": "ok" if event else "unknown",
                }
            )
    return rows


def run_perf_stat(
    workload_command: list[str],
    events: list[str],
    interval_ms: int | None,
    cwd: Path,
    env: dict[str, str] | None = None,
    system_wide_cpus: list[int] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]], list[str], float]:
    perf_command = ["perf", "stat", "-x,"]
    if system_wide_cpus is not None:
        cpu_text = ",".join(str(cpu) for cpu in system_wide_cpus)
        perf_command.extend(["-a", "-A", "-C", cpu_text])
    if interval_ms is not None:
        perf_command.extend(["-I", str(interval_ms)])
    perf_command.extend(["-e", ",".join(events), "--"])
    perf_command.extend(workload_command)
    start = time.monotonic()
    completed = subprocess.run(
        perf_command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    elapsed_ms = (time.monotonic() - start) * 1000.0
    raw_text = (completed.stderr or "") + (completed.stdout or "")
    parsed = parse_perf_csv(raw_text, interval_mode=interval_ms is not None)
    issues = [
        line
        for line in raw_text.splitlines()
        if "not supported" in line.lower() or "permission denied" in line.lower() or "access to performance monitoring" in line.lower()
    ]
    return completed, parsed, issues, elapsed_ms


def start_perf_sidecar(events: list[str], interval_ms: int, cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str] | None:
    if not events:
        return None
    command = ["perf", "stat", "-x,", "-a", "-I", str(interval_ms), "-e", ",".join(events), "--", "sleep", "1000000"]
    try:
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError:
        return None


def start_pcm_monitor(pcm_template: str, output_path: Path, interval_ms: int, cwd: Path) -> subprocess.Popen[str] | None:
    command = pcm_template.format(output=output_path, interval_ms=interval_ms)
    try:
        return subprocess.Popen(
            shlex.split(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None


def stop_pcm_monitor(process: subprocess.Popen[str] | None) -> dict[str, object]:
    if process is None:
        return {"started": False, "returncode": None}
    process.send_signal(signal.SIGINT)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    return {
        "started": True,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def stop_perf_sidecar(process: subprocess.Popen[str] | None, interval_mode: bool = True) -> dict[str, object]:
    if process is None:
        return {"started": False, "returncode": None, "rows": [], "issues": [], "raw_text": ""}
    try:
        os.killpg(process.pid, signal.SIGINT)
    except OSError:
        process.send_signal(signal.SIGINT)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        stdout, stderr = process.communicate()
    raw_text = (stderr or "") + (stdout or "")
    return {
        "started": True,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "raw_text": raw_text,
        "rows": parse_perf_csv(raw_text, interval_mode=interval_mode),
        "issues": [
            line
            for line in raw_text.splitlines()
            if "not supported" in line.lower() or "permission denied" in line.lower() or "access to performance monitoring" in line.lower()
        ],
    }


def write_perf_artifacts(
    run_dir: Path,
    mode: str,
    completed: subprocess.CompletedProcess[str],
    parsed_rows: list[dict[str, object]],
) -> None:
    ensure_dir(run_dir)
    (run_dir / f"{mode}_perf_raw.txt").write_text((completed.stderr or "") + (completed.stdout or ""), encoding="utf-8")
    header = [
        "timestamp_ms",
        "cpu_or_core_id",
        "value",
        "unit",
        "event_name",
        "event_name_normalized",
        "runtime_ns",
        "enabled_pct",
        "metric_value",
        "metric_unit",
        "status",
    ]
    from .io_utils import write_csv_rows

    write_csv_rows(run_dir / f"{mode}_perf.csv", parsed_rows, header)


def write_sidecar_perf_artifacts(
    run_dir: Path,
    stem: str,
    raw_text: str,
    parsed_rows: list[dict[str, object]],
) -> None:
    ensure_dir(run_dir)
    (run_dir / f"{stem}_raw.txt").write_text(raw_text, encoding="utf-8")
    header = [
        "timestamp_ms",
        "cpu_or_core_id",
        "value",
        "unit",
        "event_name",
        "event_name_normalized",
        "runtime_ns",
        "enabled_pct",
        "metric_value",
        "metric_unit",
        "status",
    ]
    from .io_utils import write_csv_rows

    write_csv_rows(run_dir / f"{stem}.csv", parsed_rows, header)


def run_workload_capture(
    run_dir: Path,
    workload_command: list[str],
    events: list[str],
    interval_ms: int,
    collect_interval: bool,
    collect_aggregate: bool,
    cwd: Path,
    env: dict[str, str] | None = None,
    pcm_template: str = "",
    uncore_events: list[str] | None = None,
    core_collection_scope: str = "task_local",
    system_wide_cpus: list[int] | None = None,
) -> dict[str, object]:
    ensure_dir(run_dir)
    results: dict[str, object] = {"issues": []}
    if collect_interval:
        uncore_process = start_perf_sidecar(uncore_events or [], interval_ms, cwd, env) if uncore_events else None
        pcm_process = start_pcm_monitor(pcm_template, run_dir / "pcm_interval.csv", interval_ms, cwd) if pcm_template else None
        perf_cpus = system_wide_cpus if core_collection_scope != "task_local" else None
        completed, rows, issues, elapsed_ms = run_perf_stat(workload_command, events, interval_ms, cwd, env, perf_cpus)
        if uncore_process:
            uncore_results = stop_perf_sidecar(uncore_process)
            write_sidecar_perf_artifacts(
                run_dir,
                "uncore_interval_perf",
                str(uncore_results.get("raw_text", "")),
                list(uncore_results.get("rows", [])),
            )
            results["uncore_interval"] = {
                "started": uncore_results.get("started", False),
                "returncode": uncore_results.get("returncode"),
                "issues": uncore_results.get("issues", []),
            }
            results["uncore_interval_returncode"] = uncore_results.get("returncode")
            results["issues"] = list(results["issues"]) + list(uncore_results.get("issues", []))
        if pcm_process:
            results["pcm_interval"] = stop_pcm_monitor(pcm_process)
        write_perf_artifacts(run_dir, "interval", completed, rows)
        results["interval_returncode"] = completed.returncode
        results["interval_actual_duration_ms"] = elapsed_ms
        results["issues"] = list(results["issues"]) + issues
        time.sleep(0.05)
    if collect_aggregate:
        perf_cpus = system_wide_cpus if core_collection_scope != "task_local" else None
        completed, rows, issues, elapsed_ms = run_perf_stat(workload_command, events, None, cwd, env, perf_cpus)
        write_perf_artifacts(run_dir, "aggregate", completed, rows)
        results["aggregate_returncode"] = completed.returncode
        results["aggregate_actual_duration_ms"] = elapsed_ms
        results["issues"] = list(results["issues"]) + issues
    return results
