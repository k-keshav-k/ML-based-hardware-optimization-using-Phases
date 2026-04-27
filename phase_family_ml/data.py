"""Shared data-loading helpers for family label artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows

from .families import FAMILY_COUNTERS


@dataclass
class FamilyLabelData:
    """In-memory view of one family label CSV."""

    rows: list[dict[str, str]]
    family_state: np.ndarray
    future_states: np.ndarray
    split: np.ndarray


def load_family_labels(path: Path, horizon: int) -> FamilyLabelData:
    """Load one family label artifact and materialize numeric arrays."""

    rows = load_csv_rows(path)
    family_state = np.asarray([int(row.get("family_state", "-1") or -1) for row in rows], dtype=int)
    future_states = np.asarray(
        [[int(row.get(f"future_state_{step}", "-1") or -1) for step in range(1, horizon + 1)] for row in rows],
        dtype=int,
    )
    split = np.asarray([row.get("split", "train") for row in rows])
    return FamilyLabelData(rows=rows, family_state=family_state, future_states=future_states, split=split)


def load_scope_family_data(experiment_dir: Path, scope: str, horizon: int) -> dict[str, FamilyLabelData]:
    """Load all family label files for one experiment/scope pair."""

    scope_dir = experiment_dir / f"threshold_{scope}"
    output: dict[str, FamilyLabelData] = {}
    for family in FAMILY_COUNTERS:
        path = scope_dir / f"family_labels_{family}.csv"
        if path.exists():
            output[family] = load_family_labels(path, horizon)
    return output


def shared_row_count(data_by_family: dict[str, FamilyLabelData]) -> int:
    """Return the row count that all loaded families share."""

    if not data_by_family:
        return 0
    return min(len(data.rows) for data in data_by_family.values())


def states_matrix(data_by_family: dict[str, FamilyLabelData], horizon: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Assemble aligned state/future tensors across families.

    Returns
    - family names in matrix order
    - current states shape [N, F]
    - future states shape [N, F, H]
    - split labels shape [N]
    - metadata rows (from the first family)
    """

    families = sorted(data_by_family.keys())
    if not families:
        return [], np.empty((0, 0), dtype=int), np.empty((0, 0, horizon), dtype=int), np.empty(0, dtype=object), []
    n = shared_row_count(data_by_family)
    current = np.full((n, len(families)), -1, dtype=int)
    future = np.full((n, len(families), horizon), -1, dtype=int)
    split = np.asarray(["train"] * n)
    for family_index, family in enumerate(families):
        payload = data_by_family[family]
        current[:, family_index] = payload.family_state[:n]
        future[:, family_index, :] = payload.future_states[:n, :]
        split = payload.split[:n]
    metadata_rows = data_by_family[families[0]].rows[:n]
    return families, current, future, split, metadata_rows
