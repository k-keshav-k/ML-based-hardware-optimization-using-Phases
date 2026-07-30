"""Robustness analyses for train-only phase clustering."""

from __future__ import annotations

import math

import numpy as np

from .labels import _cluster_one_scope, safe_counter_columns
from .splits import ExperimentSplit


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the adjusted Rand index without a scikit-learn dependency."""

    first = np.asarray(first, dtype=int)
    second = np.asarray(second, dtype=int)
    valid = (first >= 0) & (second >= 0)
    first = first[valid]
    second = second[valid]
    if first.size < 2:
        return 1.0

    first_values = sorted(set(first.tolist()))
    second_values = sorted(set(second.tolist()))
    table = np.zeros((len(first_values), len(second_values)), dtype=np.int64)
    first_index = {value: index for index, value in enumerate(first_values)}
    second_index = {value: index for index, value in enumerate(second_values)}
    for left, right in zip(first, second):
        table[first_index[int(left)], second_index[int(right)]] += 1

    def choose_two(values: np.ndarray) -> float:
        values = values.astype(float)
        return float(np.sum(values * (values - 1.0) / 2.0))

    joint = choose_two(table)
    row_pairs = choose_two(np.sum(table, axis=1))
    column_pairs = choose_two(np.sum(table, axis=0))
    total_pairs = float(first.size * (first.size - 1) / 2)
    expected = row_pairs * column_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    return (joint - expected) / denominator if denominator else 1.0


def cluster_stability_rows(
    rows: list[dict[str, str]],
    split: ExperimentSplit,
    *,
    k_values: list[int],
    seeds: list[int],
) -> list[dict[str, object]]:
    """Measure label stability across initializations for each candidate k."""

    counters = safe_counter_columns(rows)
    row_mask = np.ones(len(rows), dtype=bool)
    output: list[dict[str, object]] = []
    for k in k_values:
        assignments: list[np.ndarray] = []
        summaries: list[dict[str, object]] = []
        minimum_shares: list[float] = []
        for seed in seeds:
            states, summary = _cluster_one_scope(
                rows,
                counters,
                split.split_by_run,
                row_mask,
                int(k),
                int(seed),
            )
            assignments.append(states)
            summaries.append(summary)
            counts = np.bincount(states[states >= 0], minlength=max(1, int(k)))
            minimum_shares.append(float(np.min(counts) / np.sum(counts)) if np.sum(counts) else 0.0)

        pairwise = [
            adjusted_rand_index(assignments[left], assignments[right])
            for left in range(len(assignments))
            for right in range(left + 1, len(assignments))
        ]
        output.append(
            {
                "experiment": split.name,
                "split_mode": split.mode,
                "k": int(k),
                "seed_count": len(seeds),
                "seed_pair_count": len(pairwise),
                "pairwise_ari_mean": float(np.mean(pairwise)) if pairwise else 1.0,
                "pairwise_ari_min": min(pairwise) if pairwise else 1.0,
                "mean_train_distance": float(np.mean([float(item["mean_train_distance"]) for item in summaries])),
                "min_centroid_distance_mean": float(np.mean([float(item["min_centroid_distance"]) for item in summaries])),
                "minimum_cluster_share_mean": float(np.mean(minimum_shares)) if minimum_shares else math.nan,
                "safe_counter_count": len(counters),
            }
        )
    return output
