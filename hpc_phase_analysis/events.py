"""Event discovery and semantic alias mapping."""

from __future__ import annotations

import re
from pathlib import Path

from .constants import COUNTER_FAMILIES
from .io_utils import read_json, run_command, slugify, write_json

EVENT_ALIAS_RE = re.compile(r"^[A-Za-z0-9_./:=,-]+$")
REQUIRED_CORE_FAMILIES = [
    "instructions_retired",
    "cycles",
    "branch_mispredictions",
]
REQUIRED_MEMORY_OR_CACHE_FAMILIES = [
    "llc_references",
    "llc_misses",
    "offcore_demand_data_reads",
]
XEON_E5_2680_V2_FAMILY = 6
XEON_E5_2680_V2_MODEL = 62
SYSTEM_WIDE_UNCORE_FAMILIES = {
    "memory_read_bandwidth": "cas_count_read",
    "memory_write_bandwidth": "cas_count_write",
}


def normalize_event_name(name: str) -> str:
    value = name.strip().lower()
    value = value.replace(":u", "").replace(":k", "")
    value = re.sub(r"\s+", " ", value)
    return value


def extract_event_aliases(perf_list_output: str) -> dict[str, str]:
    discovered: dict[str, str] = {}
    for raw_line in perf_list_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Error:"):
            continue
        if line.endswith(":") and "[" not in line and "/" not in line:
            continue
        head = line.split("[", 1)[0].strip()
        if not head:
            continue
        head = re.split(r"\s{2,}", head, maxsplit=1)[0].strip()
        if not head:
            continue
        for alias in re.split(r"\s+OR\s+", head):
            alias = alias.strip().rstrip(",")
            if not alias or not EVENT_ALIAS_RE.fullmatch(alias):
                continue
            discovered[normalize_event_name(alias)] = alias
    return discovered


def discover_perf_list_output() -> dict[str, object]:
    result = run_command(["perf", "list"])
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def is_xeon_e5_2680_v2(platform_info: dict[str, object]) -> bool:
    return (
        str(platform_info.get("vendor_key", "")) == "intel"
        and int(platform_info.get("cpu_family", 0) or 0) == XEON_E5_2680_V2_FAMILY
        and int(platform_info.get("cpu_model_id", 0) or 0) == XEON_E5_2680_V2_MODEL
    )


def lookup_candidate(
    discovered_aliases: dict[str, str],
    candidates: list[str],
) -> tuple[str, str]:
    for candidate in candidates:
        normalized = normalize_event_name(candidate)
        if normalized in discovered_aliases:
            return discovered_aliases[normalized], candidate
    return "", ""


def resolve_uncore_family(platform_info: dict[str, object], family: str) -> dict[str, object] | None:
    if not is_xeon_e5_2680_v2(platform_info):
        return None
    if family not in SYSTEM_WIDE_UNCORE_FAMILIES:
        if family == "total_memory_bandwidth":
            return {
                "supported": True,
                "selected_event": "derived_total_memory_bandwidth",
                "selected_via": "uncore_imc/cas_count_read/ + uncore_imc/cas_count_write/",
                "selection_kind": "derived_uncore_total",
                "analysis_confident": True,
                "collection_scope": "derived_uncore",
                "event_alias": "",
                "requires_system_wide": True,
            }
        return None
    uncore_imc_devices = list(platform_info.get("uncore_imc_devices", []))
    if not uncore_imc_devices:
        return None
    event_alias = SYSTEM_WIDE_UNCORE_FAMILIES[family]
    return {
        "supported": True,
        "selected_event": f"uncore_imc/{event_alias}/",
        "selected_via": f"uncore_imc/{event_alias}/",
        "selection_kind": "vendor_preferred_uncore",
        "analysis_confident": True,
        "collection_scope": "system_wide_uncore",
        "event_alias": event_alias,
        "requires_system_wide": True,
    }


def build_alias_map(platform_info: dict[str, object], discovered_aliases: dict[str, str]) -> dict[str, dict[str, object]]:
    vendor_key = str(platform_info.get("vendor_key", "generic"))
    alias_map: dict[str, dict[str, object]] = {}
    for family, metadata in COUNTER_FAMILIES.items():
        preferred = metadata["preferred"]
        vendor_candidates = list(preferred.get(vendor_key, []))
        generic_candidates = list(preferred.get("generic", []))
        selected = ""
        source = ""
        selection_kind = "unsupported"
        collection_scope = "task_local"
        analysis_confident = False
        event_alias = ""
        requires_system_wide = False

        uncore_resolution = resolve_uncore_family(platform_info, family)
        if uncore_resolution:
            selected = str(uncore_resolution["selected_event"])
            source = str(uncore_resolution["selected_via"])
            selection_kind = str(uncore_resolution["selection_kind"])
            collection_scope = str(uncore_resolution["collection_scope"])
            analysis_confident = bool(uncore_resolution["analysis_confident"])
            event_alias = str(uncore_resolution["event_alias"])
            requires_system_wide = bool(uncore_resolution["requires_system_wide"])
        else:
            vendor_selected, vendor_source = lookup_candidate(discovered_aliases, vendor_candidates)
            generic_selected, generic_source = lookup_candidate(discovered_aliases, generic_candidates)
            if vendor_selected:
                selected = vendor_selected
                source = vendor_source
                selection_kind = "vendor_preferred"
                analysis_confident = True
            elif generic_selected:
                selected = generic_selected
                source = generic_source
                selection_kind = "generic_fallback"

        alias_map[family] = {
            "family": family,
            "label": metadata["label"],
            "supported": bool(selected),
            "selected_event": selected,
            "selected_via": source,
            "candidate_count": len(vendor_candidates) + len(generic_candidates),
            "candidates_checked": vendor_candidates + generic_candidates,
            "vendor_candidates": vendor_candidates,
            "generic_candidates": generic_candidates,
            "selection_kind": selection_kind,
            "analysis_confident": analysis_confident,
            "collection_scope": collection_scope,
            "requires_system_wide": requires_system_wide,
            "event_alias": event_alias,
            "portability": metadata["portability"],
            "interpretability": metadata["interpretability"],
            "collection_cost": metadata["collection_cost"],
        }
    return alias_map


def path_exists(path: str) -> bool:
    return Path(path).exists()


def detect_system_wide_policy_block(platform_info: dict[str, object]) -> tuple[bool, str]:
    paranoid = str(platform_info.get("perf", {}).get("perf_event_paranoid", "")).strip()
    try:
        paranoid_value = int(paranoid)
    except ValueError:
        paranoid_value = 0
    if paranoid_value > 0:
        result = run_command(["perf", "stat", "-x,", "-a", "-e", "cycles", "--", "sleep", "0.1"])
        text = f"{result.stderr}\n{result.stdout}".lower()
        if result.returncode != 0 or "access to performance monitoring" in text or "perf_event_paranoid" in text or "permission" in text:
            return True, (
                "System-wide perf is blocked on this host. "
                f"perf_event_paranoid={paranoid_value}; lower it or grant perf capability for uncore collection."
            )
    return False, ""


def supported_uncore_event_specs(
    platform_info: dict[str, object],
    alias_map: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    uncore_imc_devices = [str(item) for item in platform_info.get("uncore_imc_devices", [])]
    for family, metadata in alias_map.items():
        if metadata.get("collection_scope") != "system_wide_uncore":
            continue
        event_alias = str(metadata.get("event_alias", "")).strip()
        if not event_alias:
            continue
        for device in uncore_imc_devices:
            specs.append(
                {
                    "family": family,
                    "device": device,
                    "event_name": f"{device}/{event_alias}/",
                }
            )
    return specs


def probe_uncore_readiness(
    platform_info: dict[str, object],
    alias_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    uncore_devices = [str(item) for item in platform_info.get("uncore_devices", [])]
    uncore_imc_devices = [str(item) for item in platform_info.get("uncore_imc_devices", [])]
    policy_blocked, policy_reason = detect_system_wide_policy_block(platform_info)
    uncore_specs = supported_uncore_event_specs(platform_info, alias_map)
    readiness = {
        "uncore_devices_present": bool(uncore_devices),
        "uncore_imc_devices": uncore_imc_devices,
        "uncore_policy_blocked": policy_blocked,
        "uncore_reason": "",
        "uncore_probe_event": uncore_specs[0]["event_name"] if uncore_specs else "",
        "uncore_probe_returncode": 0,
        "uncore_probe_output": "",
        "study_ready_uncore": False,
    }
    if not uncore_imc_devices:
        readiness["uncore_reason"] = "No Intel uncore IMC PMUs were detected in /sys/bus/event_source/devices."
        return readiness
    if policy_blocked:
        readiness["uncore_reason"] = policy_reason
        return readiness
    if not uncore_specs:
        readiness["uncore_reason"] = "No confident uncore IMC events were selected for this host."
        return readiness
    probe = run_command(["perf", "stat", "-x,", "-a", "-e", uncore_specs[0]["event_name"], "--", "sleep", "0.1"])
    probe_output = f"{probe.stderr}\n{probe.stdout}".strip()
    readiness["uncore_probe_returncode"] = probe.returncode
    readiness["uncore_probe_output"] = probe_output
    blocked_text = probe_output.lower()
    if probe.returncode == 0 and "<not supported>" not in blocked_text:
        readiness["study_ready_uncore"] = True
        readiness["uncore_reason"] = "System-wide uncore IMC perf collection is available."
    elif "access to performance monitoring" in blocked_text or "perf_event_paranoid" in blocked_text or "permission" in blocked_text:
        readiness["uncore_policy_blocked"] = True
        readiness["uncore_reason"] = (
            "Intel uncore IMC PMUs exist, but system-wide perf collection is blocked by host policy. "
            "Lower perf_event_paranoid or grant perf capability."
        )
    else:
        readiness["uncore_reason"] = (
            "Intel uncore IMC PMUs were detected, but a direct perf probe did not return counts. "
            "Inspect kernel PMU support and the selected uncore aliases."
        )
    return readiness


def compute_study_readiness(
    platform_info: dict[str, object],
    alias_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    confident_task_local = {
        family
        for family, metadata in alias_map.items()
        if metadata.get("analysis_confident", False) and metadata.get("collection_scope") == "task_local"
    }
    missing_required = [family for family in REQUIRED_CORE_FAMILIES if family not in confident_task_local]
    cache_or_memory_ready = any(family in confident_task_local for family in REQUIRED_MEMORY_OR_CACHE_FAMILIES)
    if not cache_or_memory_ready:
        missing_required.append("one_of:llc_references|llc_misses|offcore_demand_data_reads")
    uncore_readiness = probe_uncore_readiness(platform_info, alias_map)
    study_ready_core = not missing_required
    strict_reason = ""
    if not study_ready_core:
        strict_reason = (
            "Strict hardware PMU study is not ready on this host because the required confident core families are missing: "
            + ", ".join(missing_required)
        )
    return {
        "study_ready_core": study_ready_core,
        "study_ready_uncore": bool(uncore_readiness["study_ready_uncore"]),
        "study_ready": study_ready_core,
        "missing_required_confident_families": missing_required,
        "strict_hardware_study_reason": strict_reason,
        "raw_confident_families": sorted(confident_task_local),
        "uncore_policy_blocked": bool(uncore_readiness["uncore_policy_blocked"]),
        "uncore_reason": str(uncore_readiness["uncore_reason"]),
        "uncore_probe_event": str(uncore_readiness["uncore_probe_event"]),
        "uncore_probe_returncode": int(uncore_readiness["uncore_probe_returncode"]),
        "uncore_probe_output": str(uncore_readiness["uncore_probe_output"]),
        "uncore_devices_present": bool(uncore_readiness["uncore_devices_present"]),
        "uncore_imc_devices": list(uncore_readiness["uncore_imc_devices"]),
    }


def write_discovery_artifacts(
    output_dir: Path,
    platform_info: dict[str, object],
    perf_list_output: dict[str, object],
    alias_map: dict[str, dict[str, object]],
    study_readiness: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perf_list.txt").write_text(str(perf_list_output["stderr"]) + str(perf_list_output["stdout"]), encoding="utf-8")
    platform_payload = dict(platform_info)
    if study_readiness is not None:
        platform_payload["pmu_study_readiness"] = study_readiness
        write_json(output_dir / "study_readiness.json", study_readiness)
    write_json(output_dir / "platform_info.json", platform_payload)
    write_json(output_dir / "event_alias_map.json", alias_map)


def load_alias_map(path: Path) -> dict[str, dict[str, object]]:
    return read_json(path)  # type: ignore[return-value]


def reverse_alias_map(alias_map: dict[str, dict[str, object]]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for family, metadata in alias_map.items():
        if metadata.get("collection_scope") == "system_wide_uncore":
            continue
        selected = str(metadata.get("selected_event", "")).strip()
        if selected:
            reverse[normalize_event_name(selected)] = family
            reverse[normalize_event_name(str(metadata.get("selected_via", "")))] = family
    return reverse


def supported_event_names(alias_map: dict[str, dict[str, object]], collection_scope: str = "task_local") -> list[str]:
    names: list[str] = []
    for metadata in alias_map.values():
        if metadata.get("collection_scope") != collection_scope:
            continue
        selected = str(metadata.get("selected_event", "")).strip()
        if selected:
            names.append(selected)
    return names


def confident_family_names(alias_map: dict[str, dict[str, object]]) -> list[str]:
    return [family for family, metadata in alias_map.items() if metadata.get("analysis_confident", False)]


def generic_fallback_family_names(alias_map: dict[str, dict[str, object]]) -> list[str]:
    return [family for family, metadata in alias_map.items() if metadata.get("selection_kind") == "generic_fallback"]


def unsupported_family_names(alias_map: dict[str, dict[str, object]]) -> list[str]:
    return [family for family, metadata in alias_map.items() if not metadata.get("supported", False)]


def alias_filename(family: str) -> str:
    return slugify(family)
