"""Analyze live co-scheduling measurements."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from .common import load_rows, safe_float, write_json, write_rows


def median_solo(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["run_kind"] != "solo" or str(row.get("returncode", "")) != "0":
            continue
        duration = safe_float(row.get("duration_s"))
        if duration > 0:
            samples[(row["workload"], row["threads"])].append(duration)
    return {key: statistics.median(values) for key, values in samples.items() if values}


def analyze(input_csv: Path, output_dir: Path) -> dict[str, object]:
    rows = load_rows(input_csv)
    solo = median_solo(rows)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["run_kind"] == "corun":
            groups[(row["policy"], row["rep"], row["group_id"], row["threads"])].append(row)
    group_metrics = []
    for (policy, rep, group_id, threads), group_rows in groups.items():
        if any(str(row.get("returncode", "")) != "0" for row in group_rows):
            failed = 1
        else:
            failed = 0
        speedups = []
        slowdowns = []
        durations = []
        for row in group_rows:
            duration = safe_float(row.get("duration_s"))
            baseline = solo.get((row["workload"], row["threads"]))
            if baseline and duration > 0:
                speedups.append(baseline / duration)
                slowdowns.append(duration / baseline)
                durations.append(duration)
        if not speedups:
            continue
        weighted_speedup = sum(speedups)
        harmonic_speedup = len(slowdowns) / sum(slowdowns) if sum(slowdowns) > 0 else 0.0
        group_duration = max(safe_float(row.get("group_duration_s")) for row in group_rows)
        group_metrics.append(
            {
                "policy": policy,
                "rep": rep,
                "group_id": group_id,
                "threads": threads,
                "process_count": len(group_rows),
                "weighted_speedup": weighted_speedup,
                "harmonic_speedup": harmonic_speedup,
                "mean_slowdown": sum(slowdowns) / len(slowdowns),
                "max_slowdown": max(slowdowns),
                "fairness_min_speedup": min(speedups),
                "throughput_jobs_per_s": len(group_rows) / group_duration if group_duration > 0 else 0.0,
                "group_duration_s": group_duration,
                "failed": failed,
                "placement_note": ";".join(sorted({row.get("placement_note", "") for row in group_rows})),
            }
        )
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in group_metrics:
        by_policy[str(row["policy"])].append(row)
    summary_rows = []
    for policy, items in sorted(by_policy.items()):
        def mean(name: str) -> float:
            return sum(float(item[name]) for item in items) / len(items) if items else 0.0

        summary_rows.append(
            {
                "policy": policy,
                "groups": len(items),
                "failures": sum(int(item["failed"]) for item in items),
                "weighted_speedup_mean": mean("weighted_speedup"),
                "harmonic_speedup_mean": mean("harmonic_speedup"),
                "mean_slowdown": mean("mean_slowdown"),
                "max_slowdown_mean": mean("max_slowdown"),
                "fairness_min_speedup_mean": mean("fairness_min_speedup"),
                "throughput_jobs_per_s_mean": mean("throughput_jobs_per_s"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "online_group_metrics.csv", group_metrics)
    write_rows(
        output_dir / "online_policy_summary.csv",
        summary_rows,
        [
            "policy",
            "groups",
            "failures",
            "weighted_speedup_mean",
            "harmonic_speedup_mean",
            "mean_slowdown",
            "max_slowdown_mean",
            "fairness_min_speedup_mean",
            "throughput_jobs_per_s_mean",
        ],
    )
    best = max(summary_rows, key=lambda row: float(row["weighted_speedup_mean"])) if summary_rows else {}
    payload = {
        "analysis_type": "measured_online_coscheduling",
        "input_csv": str(input_csv),
        "solo_baseline_count": len(solo),
        "group_count": len(group_metrics),
        "best_weighted_speedup_policy": best,
    }
    write_json(output_dir / "online_coscheduling_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze(Path(args.input_csv), Path(args.output_dir))
    print(f"Analyzed {summary['group_count']} online co-scheduling groups.")


if __name__ == "__main__":
    main()

