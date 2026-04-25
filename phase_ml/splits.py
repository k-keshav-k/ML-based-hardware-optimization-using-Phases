"""Leakage-aware split helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def run_grouped_split(
    rows: list[dict[str, object]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    by_workload: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        workload = str(row.get("workload", ""))
        run_id = str(row.get("run_id", ""))
        if run_id and run_id not in by_workload[workload]:
            by_workload[workload].append(run_id)
    rng = np.random.default_rng(seed)
    split_by_run: dict[str, str] = {}
    for workload, run_ids in by_workload.items():
        shuffled = list(run_ids)
        rng.shuffle(shuffled)
        count = len(shuffled)
        train_count = max(1, int(round(count * train_fraction))) if count else 0
        val_count = int(round(count * val_fraction)) if count - train_count > 1 else 0
        if train_count + val_count >= count and count > 1:
            train_count = count - 1
            val_count = 0
        for index, run_id in enumerate(shuffled):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            split_by_run[run_id] = split
    return split_by_run


def workload_holdout_split(rows: list[dict[str, object]], holdout_workload: str | None = None) -> dict[str, str]:
    workloads = sorted({str(row.get("workload", "")) for row in rows if row.get("workload", "")})
    if not workloads:
        return {}
    holdout = holdout_workload or workloads[-1]
    split_by_run: dict[str, str] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        workload = str(row.get("workload", ""))
        split_by_run[run_id] = "test" if workload == holdout else "train"
    return split_by_run
