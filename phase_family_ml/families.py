"""Counter-family definitions and feature engineering helpers.

This module centralizes family membership, forbidden-feature checks, and the
small set of interpretable derived counter features used by family models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpc_phase_analysis.io_utils import safe_float

# Keep the policy aligned with the existing phase_ml constraints.
FORBIDDEN_PARTS = [
    "cycle",
    "ipc",
    "cpi",
    "elapsed",
    "stall",
    "per_ms",
    "interval_duration",
    "runtime",
]

# Raw counter columns grouped by semantic family.
FAMILY_COUNTERS: dict[str, list[str]] = {
    "L1": ["counter__l1d_loads", "counter__l1d_stores"],
    "L2": ["counter__l2_misses"],
    "LLC": ["counter__llc_references", "counter__llc_misses"],
    "memory_offcore": [
        "counter__offcore_demand_data_reads",
        "counter__memory_read_bandwidth",
        "counter__memory_write_bandwidth",
        "counter__total_memory_bandwidth",
    ],
    "branch_control": ["counter__branch_instructions", "counter__branch_mispredictions"],
    "core_fp": ["counter__instructions_retired", "counter__fp_arithmetic"],
}


@dataclass(frozen=True)
class FamilyFeature:
    """Definition of one derived feature used for family state estimation."""

    name: str
    family: str


DERIVED_FEATURES: list[FamilyFeature] = [
    FamilyFeature("l1d_loads_per_ki", "L1"),
    FamilyFeature("l1d_stores_per_ki", "L1"),
    FamilyFeature("l2_misses_per_ki", "L2"),
    FamilyFeature("llc_refs_per_ki", "LLC"),
    FamilyFeature("llc_misses_per_ki", "LLC"),
    FamilyFeature("llc_miss_rate", "LLC"),
    FamilyFeature("offcore_reads_per_ki", "memory_offcore"),
    FamilyFeature("memory_total_bandwidth", "memory_offcore"),
    FamilyFeature("memory_write_fraction", "memory_offcore"),
    FamilyFeature("branch_instructions_per_ki", "branch_control"),
    FamilyFeature("branch_miss_rate", "branch_control"),
    FamilyFeature("fp_arithmetic_per_ki", "core_fp"),
    FamilyFeature("instructions_per_interval", "core_fp"),
]


def _safe_div(numerator: float, denominator: float, scale: float = 1.0) -> float:
    """Return a scaled division result while handling missing/zero values."""

    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator * scale


def assert_no_forbidden_counter_columns(columns: list[str]) -> None:
    """Reject counter columns that violate the no-timing-derived policy."""

    bad = [column for column in columns if any(part in column.lower() for part in FORBIDDEN_PARTS)]
    if bad:
        raise ValueError("Forbidden counter columns detected: " + ", ".join(sorted(bad)))


def available_counters(rows: list[dict[str, str]]) -> list[str]:
    """Return present ``counter__*`` columns found in interval rows."""

    if not rows:
        return []
    columns = sorted({key for row in rows for key in row.keys() if key.startswith("counter__")})
    assert_no_forbidden_counter_columns(columns)
    return columns


def derive_feature_values(row: dict[str, str]) -> dict[str, float]:
    """Compute interpretable derived features for one interval row.

    The formulas intentionally avoid elapsed time, cycle counts, and per-ms
    features so they remain stable across frequency and runtime shifts.
    """

    inst = safe_float(row.get("counter__instructions_retired", ""))
    branch = safe_float(row.get("counter__branch_instructions", ""))
    branch_miss = safe_float(row.get("counter__branch_mispredictions", ""))
    l1_loads = safe_float(row.get("counter__l1d_loads", ""))
    l1_stores = safe_float(row.get("counter__l1d_stores", ""))
    l2_misses = safe_float(row.get("counter__l2_misses", ""))
    llc_refs = safe_float(row.get("counter__llc_references", ""))
    llc_misses = safe_float(row.get("counter__llc_misses", ""))
    offcore_reads = safe_float(row.get("counter__offcore_demand_data_reads", ""))
    mem_read = safe_float(row.get("counter__memory_read_bandwidth", ""))
    mem_write = safe_float(row.get("counter__memory_write_bandwidth", ""))
    mem_total = safe_float(row.get("counter__total_memory_bandwidth", ""))
    fp = safe_float(row.get("counter__fp_arithmetic", ""))
    if math.isnan(mem_total):
        if not math.isnan(mem_read) or not math.isnan(mem_write):
            mem_total = (0.0 if math.isnan(mem_read) else mem_read) + (0.0 if math.isnan(mem_write) else mem_write)
    return {
        "l1d_loads_per_ki": _safe_div(l1_loads, inst, 1000.0),
        "l1d_stores_per_ki": _safe_div(l1_stores, inst, 1000.0),
        "l2_misses_per_ki": _safe_div(l2_misses, inst, 1000.0),
        "llc_refs_per_ki": _safe_div(llc_refs, inst, 1000.0),
        "llc_misses_per_ki": _safe_div(llc_misses, inst, 1000.0),
        "llc_miss_rate": _safe_div(llc_misses, llc_refs),
        "offcore_reads_per_ki": _safe_div(offcore_reads, inst, 1000.0),
        "memory_total_bandwidth": mem_total,
        "memory_write_fraction": _safe_div(mem_write, mem_total),
        "branch_instructions_per_ki": _safe_div(branch, inst, 1000.0),
        "branch_miss_rate": _safe_div(branch_miss, branch),
        "fp_arithmetic_per_ki": _safe_div(fp, inst, 1000.0),
        "instructions_per_interval": inst,
    }


def derived_columns_for_family(family: str) -> list[str]:
    """Return derived feature names that feed one family state."""

    return [item.name for item in DERIVED_FEATURES if item.family == family]


def family_counter_availability(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Return available raw counters per family for the loaded dataset."""

    present = set(available_counters(rows))
    output: dict[str, list[str]] = {}
    for family, counters in FAMILY_COUNTERS.items():
        output[family] = [counter for counter in counters if counter in present]
    return output
