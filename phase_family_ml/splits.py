"""Experiment split builders for family-wise phase modeling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentSplit:
    """One concrete split assignment used by a training/evaluation run."""

    mode: str
    name: str
    split_by_run: dict[str, str]


def _run_grouped_split(rows: list[dict[str, str]], train_fraction: float, val_fraction: float, seed: int) -> dict[str, str]:
    """Split run IDs into train/val/test while keeping intervals grouped by run."""

    run_ids = sorted({str(row.get("run_id", "")) for row in rows if row.get("run_id", "")})
    rng = np.random.default_rng(seed)
    shuffled = list(run_ids)
    rng.shuffle(shuffled)
    count = len(shuffled)
    if count == 0:
        return {}
    train_count = max(1, int(round(count * train_fraction)))
    val_count = int(round(count * val_fraction)) if count - train_count > 1 else 0
    if train_count + val_count >= count and count > 1:
        train_count = count - 1
        val_count = 0
    output: dict[str, str] = {}
    for index, run_id in enumerate(shuffled):
        if index < train_count:
            output[run_id] = "train"
        elif index < train_count + val_count:
            output[run_id] = "val"
        else:
            output[run_id] = "test"
    return output


def _per_workload_holdout(rows: list[dict[str, str]], train_fraction: float, val_fraction: float, seed: int) -> dict[str, str]:
    """Split runs independently inside each workload group."""

    by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row.get("workload", ""))].append(row)
    output: dict[str, str] = {}
    for offset, workload in enumerate(sorted(by_workload)):
        payload = _run_grouped_split(by_workload[workload], train_fraction, val_fraction, seed + offset)
        output.update(payload)
    return output


def _leave_one_workload_out(rows: list[dict[str, str]], holdout_workload: str) -> dict[str, str]:
    """Use one workload as test and split remaining runs into train/val."""

    split_by_run: dict[str, str] = {}
    non_holdout_rows: list[dict[str, str]] = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        if str(row.get("workload", "")) == holdout_workload:
            split_by_run[run_id] = "test"
        else:
            non_holdout_rows.append(row)
    train_val = _run_grouped_split(non_holdout_rows, train_fraction=0.85, val_fraction=0.15, seed=31)
    for run_id, split in train_val.items():
        if run_id not in split_by_run:
            split_by_run[run_id] = split
    return split_by_run


def build_experiment_splits(
    rows: list[dict[str, str]],
    mode: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> list[ExperimentSplit]:
    """Return one or more concrete splits for the requested experiment mode."""

    if mode not in {"per_workload_holdout", "pooled_run_group", "leave_one_workload_out", "all"}:
        raise ValueError(f"Unknown experiment mode: {mode}")
    output: list[ExperimentSplit] = []
    if mode in {"per_workload_holdout", "all"}:
        output.append(
            ExperimentSplit(
                mode="per_workload_holdout",
                name="per_workload_holdout",
                split_by_run=_per_workload_holdout(rows, train_fraction, val_fraction, seed),
            )
        )
    if mode in {"pooled_run_group", "all"}:
        output.append(
            ExperimentSplit(
                mode="pooled_run_group",
                name="pooled_run_group",
                split_by_run=_run_grouped_split(rows, train_fraction, val_fraction, seed + 101),
            )
        )
    if mode in {"leave_one_workload_out", "all"}:
        workloads = sorted({str(row.get("workload", "")) for row in rows if row.get("workload", "")})
        for workload in workloads:
            output.append(
                ExperimentSplit(
                    mode="leave_one_workload_out",
                    name=f"leave_one_workload_out__{workload}",
                    split_by_run=_leave_one_workload_out(rows, workload),
                )
            )
    return output
