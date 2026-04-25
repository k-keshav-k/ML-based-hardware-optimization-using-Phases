"""Phase-sharing and equivalence-class analysis."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import entropy, load_rows, phase_resource_classes, write_json, write_rows


def analyze_uniqueness(dataset_dir: Path, label_dir: Path, output_dir: Path) -> dict[str, object]:
    interval_rows = load_rows(dataset_dir / "interval_features.csv")
    window_rows = load_rows(label_dir / "window_labels.csv")
    phase_classes = phase_resource_classes(interval_rows, window_rows)
    phase_workloads: dict[int, Counter[str]] = defaultdict(Counter)
    for row in window_rows:
        phase_workloads[int(row["phase_id"])][row["workload"]] += 1

    rows: list[dict[str, object]] = []
    for phase, counts in sorted(phase_workloads.items()):
        total = sum(counts.values())
        dominant, dominant_count = counts.most_common(1)[0]
        rows.append(
            {
                "phase": phase,
                "windows": total,
                "workload_count": sum(1 for value in counts.values() if value > 0),
                "dominant_workload": dominant,
                "dominant_share": dominant_count / total if total else 0.0,
                "workload_entropy_bits": entropy(counts),
                "resource_class": phase_classes.get(phase, "balanced"),
            }
        )
    shared = [row for row in rows if int(row["workload_count"]) >= 4]
    specific = [row for row in rows if float(row["dominant_share"]) >= 0.55]
    summary = {
        "phase_count": len(rows),
        "shared_phase_count": len(shared),
        "workload_specific_phase_count": len(specific),
        "mean_dominant_workload_share": sum(float(row["dominant_share"]) for row in rows) / max(1, len(rows)),
        "mean_workload_entropy_bits": sum(float(row["workload_entropy_bits"]) for row in rows) / max(1, len(rows)),
        "resource_class_counts": dict(Counter(str(row["resource_class"]) for row in rows)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "phase_uniqueness.csv",
        rows,
        ["phase", "windows", "workload_count", "dominant_workload", "dominant_share", "workload_entropy_bits", "resource_class"],
    )
    write_json(output_dir / "phase_uniqueness_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze_uniqueness(Path(args.dataset_dir), Path(args.label_dir), Path(args.output_dir))
    print(f"Wrote uniqueness analysis for {summary['phase_count']} phases.")


if __name__ == "__main__":
    main()
