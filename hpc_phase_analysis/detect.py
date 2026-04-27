"""Platform and suite detection helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .constants import PROJECT_ROOT
from .io_utils import read_json, run_command, safe_int


def parse_key_value_lines(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def parse_cpu_range_list(value: str) -> list[int]:
    cpus: list[int] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(item))
    return cpus


def read_cache_info() -> list[dict[str, str]]:
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    entries: list[dict[str, str]] = []
    if not cache_root.exists():
        return entries
    for index_dir in sorted(cache_root.glob("index*")):
        entry: dict[str, str] = {"index": index_dir.name}
        for field in ["level", "type", "size", "coherency_line_size", "number_of_sets", "ways_of_associativity", "shared_cpu_list"]:
            path = index_dir / field
            entry[field] = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        entries.append(entry)
    return entries


def read_cpu_topology(online_cpus: list[int]) -> dict[str, dict[str, object]]:
    topology: dict[str, dict[str, object]] = {}
    for cpu in online_cpus:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package_path = root / "physical_package_id"
        core_path = root / "core_id"
        siblings_path = root / "thread_siblings_list"
        package_id = package_path.read_text(encoding="utf-8").strip() if package_path.exists() else "0"
        core_id = core_path.read_text(encoding="utf-8").strip() if core_path.exists() else str(cpu)
        topology[str(cpu)] = {
            "logical_cpu": cpu,
            "physical_package_id": package_id,
            "core_id": core_id,
            "physical_core_id": f"{package_id}:{core_id}",
            "thread_siblings_list": siblings_path.read_text(encoding="utf-8").strip() if siblings_path.exists() else str(cpu),
        }
    return topology


def discover_parsec() -> dict[str, object]:
    candidates = []
    if os.environ.get("PARSEC_DIR"):
        candidates.append(Path(os.environ["PARSEC_DIR"]))
    candidates.extend(
        [
            PROJECT_ROOT / "third_party" / "parsec-benchmark",
            Path("/opt/parsec"),
            Path.home() / "parsec-benchmark",
        ]
    )
    parsecmgmt_path = shutil.which("parsecmgmt")
    if parsecmgmt_path:
        candidates.append(Path(parsecmgmt_path).resolve().parents[1])
    for root in candidates:
        parsecmgmt = root / "bin" / "parsecmgmt"
        if parsecmgmt.exists():
            benchmarks: list[str] = []
            for pattern in ("pkgs/apps/*", "pkgs/kernels/*", "pkgs/netapps/*"):
                for item in sorted(root.glob(pattern)):
                    if item.is_dir():
                        benchmarks.append(item.name)
            return {
                "available": True,
                "root": str(root),
                "parsecmgmt": str(parsecmgmt),
                "benchmarks": benchmarks,
            }
    return {"available": False, "root": "", "parsecmgmt": "", "benchmarks": []}


def detect_platform() -> dict[str, object]:
    lscpu = run_command(["lscpu"])
    lscpu_data = parse_key_value_lines(lscpu.stdout)
    perf_path = shutil.which("perf") or ""
    perf_version = run_command(["perf", "--version"]).stdout.strip() if perf_path else ""
    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    perf_event_paranoid = paranoid_path.read_text(encoding="utf-8").strip() if paranoid_path.exists() else ""

    online_cpus = parse_cpu_range_list(lscpu_data.get("On-line CPU(s) list", "0"))
    cpu_topology = read_cpu_topology(online_cpus)
    uncore_devices = sorted(path.name for path in Path("/sys/bus/event_source/devices").glob("uncore_*"))
    uncore_imc_devices = [name for name in uncore_devices if name.startswith("uncore_imc_")]

    cpu_vendor = lscpu_data.get("Vendor ID", "")
    arch = lscpu_data.get("Architecture", "")
    vendor_key = "generic"
    lowered_vendor = cpu_vendor.lower()
    if "intel" in lowered_vendor:
        vendor_key = "intel"
    elif "amd" in lowered_vendor:
        vendor_key = "amd"
    elif "arm" in arch.lower() or "aarch64" in arch.lower():
        vendor_key = "arm"

    pcm_path = shutil.which("pcm-memory") or shutil.which("pcm.x") or ""

    return {
        "architecture": arch,
        "cpu_vendor": cpu_vendor,
        "vendor_key": vendor_key,
        "cpu_family": safe_int(lscpu_data.get("CPU family")),
        "cpu_model_id": safe_int(lscpu_data.get("Model")),
        "cpu_model": lscpu_data.get("Model name", ""),
        "logical_cpus": safe_int(lscpu_data.get("CPU(s)")),
        "threads_per_core": safe_int(lscpu_data.get("Thread(s) per core")),
        "cores_per_socket": safe_int(lscpu_data.get("Core(s) per socket")),
        "socket_count": safe_int(lscpu_data.get("Socket(s)")),
        "numa_nodes": safe_int(lscpu_data.get("NUMA node(s)")),
        "online_cpus": online_cpus,
        "cpu_topology": cpu_topology,
        "cache_summary": {
            "l1d": lscpu_data.get("L1d cache", ""),
            "l1i": lscpu_data.get("L1i cache", ""),
            "l2": lscpu_data.get("L2 cache", ""),
            "l3": lscpu_data.get("L3 cache", ""),
        },
        "cache_detail": read_cache_info(),
        "uncore_devices": uncore_devices,
        "uncore_imc_devices": uncore_imc_devices,
        "perf": {
            "path": perf_path,
            "version": perf_version,
            "perf_event_paranoid": perf_event_paranoid,
        },
        "intel_pcm": {
            "available": bool(pcm_path),
            "path": pcm_path,
        },
        "parsec": discover_parsec(),
        "lscpu_raw": lscpu.stdout,
    }


def load_platform_file(path: Path) -> dict[str, object]:
    return read_json(path)  # type: ignore[return-value]
