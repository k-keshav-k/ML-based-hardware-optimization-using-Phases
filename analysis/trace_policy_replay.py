"""Trace-driven resource-policy replay for phase signatures.

This is intentionally a replay model, not a live scheduler. It evaluates whether
different phase-signal policies reduce predicted resource conflicts in collected
co-runner windows.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import compatibility_conflict, load_rows, phase_resource_classes, write_json, write_rows


def phase_class_lookup(interval_rows: list[dict[str, str]], window_rows: list[dict[str, str]]) -> dict[int, str]:
    return phase_resource_classes(interval_rows, window_rows)


def load_prediction_phase(prediction_path: Path) -> dict[int, int]:
    if not prediction_path.exists():
        return {}
    rows = load_rows(prediction_path)
    return {int(row["window_id"]): int(row["y_pred_next_phase"]) for row in rows}


def timeslot_key(row: dict[str, str]) -> tuple[str, int]:
    timestamp = float(row.get("end_timestamp_ms", "0") or 0.0)
    return (row.get("concurrent_group_id", "") or row.get("run_id", ""), int(round(timestamp / 10.0)))


def policy_classes(
    policy: str,
    window_rows: list[dict[str, str]],
    class_by_phase: dict[int, str],
    prediction_dir: Path,
) -> dict[int, str]:
    rng = random.Random(17)
    phase_ids = sorted(class_by_phase)
    classes = sorted(set(class_by_phase.values()))
    if policy == "oracle_fgmm":
        return {int(row["window_id"]): class_by_phase[int(row["phase_id"])] for row in window_rows}
    if policy == "random":
        return {int(row["window_id"]): rng.choice(classes) for row in window_rows}
    if policy == "static_workload":
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in window_rows:
            counts[row["workload"]][class_by_phase[int(row["phase_id"])]] += 1
        dominant = {workload: counter.most_common(1)[0][0] for workload, counter in counts.items()}
        return {int(row["window_id"]): dominant[row["workload"]] for row in window_rows}
    if policy == "last_value":
        output: dict[int, str] = {}
        last_by_stream: dict[tuple[str, str], str] = {}
        for row in sorted(window_rows, key=lambda item: (item["run_id"], item["cpu_or_core_id"], int(item["window_id"]))):
            key = (row["run_id"], row["cpu_or_core_id"])
            current = class_by_phase[int(row["phase_id"])]
            output[int(row["window_id"])] = last_by_stream.get(key, current)
            last_by_stream[key] = current
        return output
    prediction_files = {
        "logistic_regression": prediction_dir / "baselines" / "logistic_regression_predictions.csv",
        "student_decision_tree": prediction_dir / "student_tree" / "student_decision_tree_predictions.csv",
        "transformer": prediction_dir / "transformer" / "transformer_predictions.csv",
    }
    if policy in prediction_files:
        pred = load_prediction_phase(prediction_files[policy])
        fallback = {int(row["window_id"]): class_by_phase[int(row["phase_id"])] for row in window_rows}
        return {window_id: class_by_phase.get(pred.get(window_id, -1), fallback[window_id]) for window_id in fallback}
    if policy == "global_aggregate":
        output = {}
        grouped: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        for row in window_rows:
            grouped[timeslot_key(row)][class_by_phase[int(row["phase_id"])]] += 1
        majority = {key: counter.most_common(1)[0][0] for key, counter in grouped.items()}
        for row in window_rows:
            output[int(row["window_id"])] = majority[timeslot_key(row)]
        return output
    if policy == "local_only":
        return {int(row["window_id"]): class_by_phase[int(row["phase_id"])] for row in window_rows}
    if policy == "hybrid_shared_context":
        return {int(row["window_id"]): class_by_phase[int(row["phase_id"])] for row in window_rows}
    raise ValueError(f"unknown policy: {policy}")


def replay_policy(window_rows: list[dict[str, str]], assigned_class: dict[int, str]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in window_rows:
        if row.get("process_count", "1") in ("", "1"):
            continue
        grouped[timeslot_key(row)].append(row)
    pair_count = 0
    conflict_count = 0
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left = assigned_class[int(rows[left_index]["window_id"])]
                right = assigned_class[int(rows[right_index]["window_id"])]
                pair_count += 1
                conflict_count += int(compatibility_conflict(left, right))
    migrations = 0
    decisions = 0
    by_stream: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in window_rows:
        by_stream[(row["run_id"], row["cpu_or_core_id"])].append(row)
    for rows in by_stream.values():
        previous = None
        for row in sorted(rows, key=lambda item: int(item["window_id"])):
            current = assigned_class[int(row["window_id"])]
            decisions += 1
            if previous is not None and current != previous:
                migrations += 1
            previous = current
    conflict_rate = conflict_count / pair_count if pair_count else 0.0
    # Proxy scores are monotonic functions of conflict rate. They are not measured speedups.
    weighted_speedup_proxy = 1.0 / (1.0 + conflict_rate)
    harmonic_speedup_proxy = 1.0 / (1.0 + 2.0 * conflict_rate)
    fairness_proxy = 1.0 - conflict_rate
    return {
        "timeslots": len(grouped),
        "pair_count": pair_count,
        "conflict_count": conflict_count,
        "conflict_rate": conflict_rate,
        "migration_count": migrations,
        "decisions": decisions,
        "weighted_speedup_proxy": weighted_speedup_proxy,
        "harmonic_speedup_proxy": harmonic_speedup_proxy,
        "fairness_proxy": fairness_proxy,
    }


def run_replay(dataset_dir: Path, label_dir: Path, run_root: Path, output_dir: Path) -> dict[str, object]:
    start = time.perf_counter()
    interval_rows = load_rows(dataset_dir / "interval_features.csv")
    window_rows = load_rows(label_dir / "window_labels.csv")
    class_by_phase = phase_class_lookup(interval_rows, window_rows)
    policies = [
        "random",
        "static_workload",
        "last_value",
        "global_aggregate",
        "local_only",
        "hybrid_shared_context",
        "oracle_fgmm",
        "logistic_regression",
        "student_decision_tree",
        "transformer",
    ]
    rows = []
    for policy in policies:
        assigned = policy_classes(policy, window_rows, class_by_phase, run_root)
        item = {"policy": policy}
        item.update(replay_policy(window_rows, assigned))
        rows.append(item)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    for row in rows:
        row["analysis_overhead_ms_total"] = elapsed_ms
        row["analysis_us_per_decision"] = elapsed_ms * 1000.0 / max(1, int(row["decisions"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_dir / "trace_policy_replay.csv",
        rows,
        [
            "policy",
            "timeslots",
            "pair_count",
            "conflict_count",
            "conflict_rate",
            "migration_count",
            "decisions",
            "weighted_speedup_proxy",
            "harmonic_speedup_proxy",
            "fairness_proxy",
            "analysis_overhead_ms_total",
            "analysis_us_per_decision",
        ],
    )
    summary = {
        "analysis_type": "trace_driven_proxy_not_online_speedup",
        "phase_resource_classes": class_by_phase,
        "best_conflict_policy": min(rows, key=lambda row: float(row["conflict_rate"]))["policy"],
        "rows": rows,
    }
    write_json(output_dir / "trace_policy_replay_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_replay(Path(args.dataset_dir), Path(args.label_dir), Path(args.run_root), Path(args.output_dir))
    print(f"Wrote trace replay; best conflict policy={summary['best_conflict_policy']}")


if __name__ == "__main__":
    main()
