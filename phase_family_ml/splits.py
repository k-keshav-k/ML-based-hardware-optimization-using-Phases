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
    """Split execution groups while keeping co-running processes together."""

    runs_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        group_id = str(row.get("concurrent_group_id", "")).strip() or run_id
        runs_by_group[group_id].add(run_id)
    split_by_group = _split_items(sorted(runs_by_group), train_fraction, val_fraction, seed)
    output: dict[str, str] = {}
    for group_id, run_ids in runs_by_group.items():
        split = split_by_group.get(group_id, "train")
        for run_id in run_ids:
            output[run_id] = split
    return output


def _split_items(items: list[object], train_fraction: float, val_fraction: float, seed: int) -> dict[object, str]:
    rng = np.random.default_rng(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    count = len(shuffled)
    if count == 0:
        return {}
    train_count = max(1, int(round(count * train_fraction)))
    val_count = int(round(count * val_fraction)) if count - train_count > 1 else 0
    if train_count + val_count >= count and count > 1:
        train_count = count - 1
        val_count = 0
    output: dict[object, str] = {}
    for index, item in enumerate(shuffled):
        if index < train_count:
            output[item] = "train"
        elif index < train_count + val_count:
            output[item] = "val"
        else:
            output[item] = "test"
    return output


def _co_running_workloads(row: dict[str, str]) -> str:
    workloads = [item.strip() for item in str(row.get("co_running_workloads", "")).split(",") if item.strip()]
    if not workloads:
        workload = str(row.get("workload", "")).strip()
        workloads = [workload] if workload else []
    return ",".join(sorted(workloads))


def _config_key(row: dict[str, str]) -> tuple[str, ...]:
    """Group repetitions and all processes from one collection scenario."""

    return (
        str(row.get("experiment_set", "")),
        str(row.get("requested_input_size", "")),
        _co_running_workloads(row),
        str(row.get("threads", "")),
        str(row.get("process_count", "")),
        str(row.get("core_collection_scope", row.get("collection_scope", ""))),
        str(row.get("pairing_mode", "")),
    )


def _config_grouped_split(rows: list[dict[str, str]], train_fraction: float, val_fraction: float, seed: int) -> dict[str, str]:
    """Split config groups while keeping all reps for a config together."""

    rows_by_execution_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        group_id = str(row.get("concurrent_group_id", "")).strip() or run_id
        rows_by_execution_group[group_id].append(row)

    runs_by_config: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for group_rows in rows_by_execution_group.values():
        representative = dict(group_rows[0])
        representative["threads"] = ",".join(
            sorted({str(row.get("threads", "")).strip() for row in group_rows if str(row.get("threads", "")).strip()})
        )
        config = _config_key(representative)
        runs_by_config[config].update(
            str(row.get("run_id", "")).strip() for row in group_rows if str(row.get("run_id", "")).strip()
        )
    split_by_config = _split_items(sorted(runs_by_config), train_fraction, val_fraction, seed)
    output: dict[str, str] = {}
    for config, run_ids in runs_by_config.items():
        split = split_by_config.get(config, "train")
        for run_id in run_ids:
            output[run_id] = split
    return output


def _per_workload_holdout(rows: list[dict[str, str]], train_fraction: float, val_fraction: float, seed: int) -> dict[str, str]:
    """Stratify by complete workload scenario without splitting co-runners."""

    by_workload_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_workload_scenario[_co_running_workloads(row)].append(row)
    output: dict[str, str] = {}
    for offset, scenario in enumerate(sorted(by_workload_scenario)):
        payload = _run_grouped_split(by_workload_scenario[scenario], train_fraction, val_fraction, seed + offset)
        output.update(payload)
    return output


def _leave_one_workload_out(rows: list[dict[str, str]], holdout_workload: str) -> dict[str, str]:
    """Hold out every co-running scenario that contains the named workload."""

    split_by_run: dict[str, str] = {}
    non_holdout_rows: list[dict[str, str]] = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        scenario_workloads = set(_co_running_workloads(row).split(","))
        if holdout_workload in scenario_workloads:
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

    if mode not in {"per_workload_holdout", "pooled_run_group", "config_group_holdout", "leave_one_workload_out", "all"}:
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
    if mode in {"config_group_holdout", "all"}:
        output.append(
            ExperimentSplit(
                mode="config_group_holdout",
                name="config_group_holdout",
                split_by_run=_config_grouped_split(rows, train_fraction, val_fraction, seed + 211),
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
