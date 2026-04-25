"""Hardware and software cost estimates for the detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_json, write_rows


def estimate_cost(output_dir: Path, feature_count: int = 8, centroid_entries: int = 16) -> dict[str, object]:
    rows = []
    for bits in [8, 10, 12, 16]:
        centroid_bits = centroid_entries * feature_count * bits
        feature_bits = feature_count * bits
        state_bits = 16 * 8 + 4 * 16
        total_bytes = (centroid_bits + feature_bits + state_bits + 7) // 8
        rows.append(
            {
                "fixed_point_bits": bits,
                "feature_count": feature_count,
                "centroid_entries": centroid_entries,
                "centroid_bytes_per_core": centroid_bits / 8,
                "feature_bytes_per_core": feature_bits / 8,
                "state_bytes_per_core": state_bits / 8,
                "total_bytes_per_core": total_bytes,
                "sub_abs_accumulate_ops_per_window": centroid_entries * feature_count,
            }
        )
    summary = {
        "implemented": "software trace replay and Python analysis",
        "not_implemented": "hardware/firmware detector, Intel CAT/resctrl control, live prefetch control",
        "pmu_event_families_required": [
            "instructions retired",
            "branch instructions",
            "branch mispredictions",
            "L1D loads/stores",
            "LLC references/misses",
            "offcore demand data reads",
            "optional uncore IMC read/write bandwidth",
        ],
        "counter_slot_risk": "The full feature set exceeds typical programmable core-counter slots; PMU-slot sensitivity must evaluate reduced feature groups.",
        "cost_rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "hardware_cost.csv",
        rows,
        [
            "fixed_point_bits",
            "feature_count",
            "centroid_entries",
            "centroid_bytes_per_core",
            "feature_bytes_per_core",
            "state_bytes_per_core",
            "total_bytes_per_core",
            "sub_abs_accumulate_ops_per_window",
        ],
    )
    write_json(output_dir / "hardware_cost_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--centroid-entries", type=int, default=16)
    args = parser.parse_args()
    estimate_cost(Path(args.output_dir), args.feature_count, args.centroid_entries)
    print(f"Wrote hardware cost estimates to {args.output_dir}")


if __name__ == "__main__":
    main()
