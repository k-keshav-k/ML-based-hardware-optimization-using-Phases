#!/usr/bin/env python3
"""Generate paper figures from the completed phase-family final run."""

from __future__ import annotations

import csv
import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from phase_family_ml.metrics import classification_metrics
from phase_family_ml.tree import DecisionTree

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SET_INFO = {
    "Set 1": {
        "summary": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "predictions": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_predictions.csv"),
        "hardware": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_hardware_budget.csv"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/ablation/family_ablation_results.csv"),
        "cluster_summary": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/counter_sequences/config_group_holdout/phase_cluster_summary.csv"),
        "cluster_manifest": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/counter_sequences/config_group_holdout/counter_sequence_manifest.json"),
        "raw_manifest": Path("/scratch/kk6081/finals_dataset/set1/raw/phase_family_ml_experiment_manifest.json"),
        "thread_accuracy": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_thread_accuracy.csv"),
        "thread_behavior": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_phase_behavior_by_thread.csv"),
        "workload_accuracy": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_workload_accuracy.csv"),
    },
    "Set 2": {
        "summary": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "predictions": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_predictions.csv"),
        "hardware": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_hardware_budget.csv"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/ablation/family_ablation_results.csv"),
        "cluster_summary": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/counter_sequences/config_group_holdout/phase_cluster_summary.csv"),
        "cluster_manifest": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/counter_sequences/config_group_holdout/counter_sequence_manifest.json"),
        "raw_manifest": Path("/scratch/kk6081/finals_dataset/set2/raw/phase_family_ml_experiment_manifest.json"),
        "thread_accuracy": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_thread_accuracy.csv"),
        "thread_behavior": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_phase_behavior_by_thread.csv"),
        "workload_accuracy": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_workload_accuracy.csv"),
    },
    "Set 3": {
        "summary": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "predictions": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_predictions.csv"),
        "hardware": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_hardware_budget.csv"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/ablation/family_ablation_results.csv"),
        "cluster_summary": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/counter_sequences/config_group_holdout/phase_cluster_summary.csv"),
        "cluster_manifest": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/counter_sequences/config_group_holdout/counter_sequence_manifest.json"),
        "raw_manifest": Path("/scratch/kk6081/finals_dataset/set3/raw/phase_family_ml_experiment_manifest.json"),
        "thread_accuracy": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_thread_accuracy.csv"),
        "thread_behavior": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_phase_behavior_by_thread.csv"),
        "workload_accuracy": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_workload_accuracy.csv"),
    },
}

FAMILIES = ["L1", "LLC", "branch_control", "core_fp", "memory_offcore"]
FAMILY_LABELS = {
    "L1": "L1",
    "LLC": "LLC",
    "branch_control": "Branch",
    "core_fp": "Core/FP",
    "memory_offcore": "Memory",
}
STATE_LABELS = ["Low", "Moderate", "High"]
COUNTER_LABELS = {
    "counter__branch_instructions": "Branch inst.",
    "counter__branch_mispredictions": "Branch mispred.",
    "counter__instructions_retired": "Inst. retired",
    "counter__l1d_loads": "L1D loads",
    "counter__l1d_stores": "L1D stores",
    "counter__llc_misses": "LLC misses",
    "counter__llc_references": "LLC refs.",
    "counter__memory_read_bandwidth": "Mem read BW",
    "counter__memory_write_bandwidth": "Mem write BW",
    "counter__offcore_demand_data_reads": "Offcore reads",
    "counter__total_memory_bandwidth": "Total mem BW",
}
PRESSURE_COUNTERS = {
    "counter__branch_mispredictions",
    "counter__llc_misses",
    "counter__memory_read_bandwidth",
    "counter__memory_write_bandwidth",
    "counter__total_memory_bandwidth",
}

MODEL_SPECS = [
    ("baseline_last_state", "Last state", "#4c78a8"),
    ("hsmm_duration_phase_predictor", "HSMM", "#54a24b"),
    ("online_phase_history_tree", "Tree (local)", "#e45756"),
    ("online_phase_history_tree_all_families", "Tree (all fam.)", "#f58518"),
    ("rocket_phase_classifier_all_families", "ROCKET", "#72b7b2"),
    ("tcn_phase_classifier_all_families", "TCN", "#b279a2"),
    ("tiny_transformer_phase_classifier_all_families", "Tiny Transformer", "#9d755d"),
]
MODEL_LABELS = {key: label for key, label, _ in MODEL_SPECS}
MODEL_COLORS = {key: color for key, _, color in MODEL_SPECS}
SET_MARKERS = {"Set 1": "o", "Set 2": "s", "Set 3": "^"}
CONTROL_MODEL = "online_phase_history_tree_all_families"
CONTROL_FAMILY = "L1"
L1_POLICY_POWER = {
    0: 0.82,
    1: 0.94,
    2: 1.00,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text())


def to_float(value: str | float | int | None, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> float:
    return 100.0 * value


def summary_rows() -> dict[str, list[dict[str, str]]]:
    return {
        set_name: [
            row
            for row in read_csv(info["summary"])
            if row.get("experiment") == "config_group_holdout" and row.get("scope") == "global"
        ]
        for set_name, info in SET_INFO.items()
    }


def ablation_rows() -> dict[str, list[dict[str, str]]]:
    return {set_name: read_csv(info["ablation"]) for set_name, info in SET_INFO.items()}


def hardware_rows() -> dict[str, list[dict[str, str]]]:
    return {set_name: read_csv(info["hardware"]) for set_name, info in SET_INFO.items()}


def cluster_models() -> dict[str, dict[str, object]]:
    return {set_name: read_json(info["cluster_manifest"]) for set_name, info in SET_INFO.items()}


def cluster_summaries() -> dict[str, dict[str, str]]:
    return {set_name: read_csv(info["cluster_summary"])[0] for set_name, info in SET_INFO.items()}


def raw_manifests() -> dict[str, list[dict[str, object]]]:
    return {set_name: read_json(info["raw_manifest"]) for set_name, info in SET_INFO.items()}


def selected_counters(rows_by_set: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for set_name, rows in rows_by_set.items():
        selected: dict[str, str] = {}
        for row in rows:
            if row.get("selected", "").lower() in {"1", "true", "yes"}:
                selected[row["family"]] = row["counter_set"]
        output[set_name] = selected
    return output


def metric_matrix(
    rows_by_set: dict[str, list[dict[str, str]]],
    metric: str,
    models: list[str],
) -> np.ndarray:
    values = np.zeros((len(models), len(SET_INFO)))
    for col, set_name in enumerate(SET_INFO):
        rows = rows_by_set[set_name]
        for row_index, model in enumerate(models):
            entries = [to_float(row.get(metric, "")) for row in rows if row.get("model") == model]
            entries = [value for value in entries if math.isfinite(value)]
            values[row_index, col] = statistics.mean(entries) if entries else 0.0
    return values


def parse_keyed_pairs(serialized: str) -> dict[int, float]:
    output: dict[int, float] = {}
    for item in serialized.split(";"):
        if not item:
            continue
        key, value = item.split(":", 1)
        output[int(key)] = float(value)
    return output


def fill_train_medians(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col in range(matrix.shape[1]):
        train_values = matrix[train_mask, col]
        clean = train_values[np.isfinite(train_values)]
        median = float(np.median(clean)) if clean.size else 0.0
        filled[~np.isfinite(filled[:, col]), col] = median
    return filled


def score_counter_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    split = np.asarray([row["split"] for row in rows], dtype=object)
    values = np.asarray([to_float(row.get("counter_value", "")) for row in rows], dtype=float).reshape(-1, 1)
    targets = np.asarray([int(row.get("future_state_1", "-1")) for row in rows], dtype=int)
    current = np.asarray([int(row.get("family_state", "-1")) for row in rows], dtype=int)
    values = fill_train_medians(values, split == "train")

    train = split == "train"
    if not np.any(train):
        train = np.ones(split.shape[0], dtype=bool)
    tree = DecisionTree(max_depth=5, min_samples_leaf=3)
    tree.fit(values[train], targets[train])
    pred = tree.predict(values)

    valid = targets >= 0
    selection = split == "val"
    if not np.any(selection):
        selection = split == "test"
    if not np.any(selection):
        selection = split != "train"
    usable = selection & valid
    if not np.any(usable):
        usable = valid
    if not np.any(usable):
        return {"accuracy": 0.0, "macro_f1": 0.0, "high_usage_recall": 0.0, "validation_score": 0.0}

    metrics = classification_metrics(targets[usable], pred[usable], current_state=current[usable])
    score = 0.7 * float(metrics["accuracy"]) + 0.3 * float(metrics["high_usage_recall"])
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "high_usage_recall": float(metrics["high_usage_recall"]),
        "validation_score": float(score),
    }


def set1_thread1_ablation_rows(global_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    base = (
        Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/")
        / "counter_sequences/config_group_holdout/threshold_global/counter_value_sequences"
    )
    output: list[dict[str, str]] = []
    best_by_family: dict[str, tuple[float, int]] = {}
    for row in global_rows:
        if row.get("candidate_type") != "singleton" or row.get("family") not in FAMILIES:
            continue
        counter = row["counter_set"]
        path = base / f"counter_sequence_{counter}.csv"
        if not path.exists():
            continue
        thread_rows = [item for item in read_csv(path) if item.get("threads") == "1"]
        if not thread_rows:
            continue
        metrics = score_counter_rows(thread_rows)
        plot_row = dict(row)
        plot_row.update(
            {
                "selection_accuracy": str(metrics["accuracy"]),
                "selection_macro_f1": str(metrics["macro_f1"]),
                "selection_high_usage_recall": str(metrics["high_usage_recall"]),
                "validation_score": str(metrics["validation_score"]),
                "selected": "0",
                "note": "set1_thread_1_only",
            }
        )
        output.append(plot_row)
        key = plot_row["family"]
        current_best = best_by_family.get(key)
        if current_best is None or metrics["validation_score"] > current_best[0]:
            best_by_family[key] = (metrics["validation_score"], len(output) - 1)

    for _, index in best_by_family.values():
        output[index]["selected"] = "1"
    return output


def plot_counter_ablation(rows_by_set: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    for axis, (set_name, rows) in zip(axes, rows_by_set.items()):
        if set_name == "Set 1":
            rows = set1_thread1_ablation_rows(rows)
        singleton_rows = [
            row
            for row in rows
            if row.get("candidate_type") == "singleton" and row.get("counter_set") and row.get("family") in FAMILIES
        ]
        plot_rows = []
        for family in FAMILIES:
            plot_rows.extend([row for row in singleton_rows if row["family"] == family])
        xs = np.arange(len(plot_rows))
        ys = [pct(to_float(row.get("validation_score", ""))) for row in plot_rows]
        colors = ["#54a24b" if row.get("selected", "").lower() in {"1", "true", "yes"} else "#c7c7c7" for row in plot_rows]
        axis.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.2)
        labels = []
        for row in plot_rows:
            counter = row["counter_set"].replace("counter__", "")
            labels.append(counter.replace("_", "\n"))
        axis.set_title("Set 1 (1 thread)" if set_name == "Set 1" else set_name)
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation score (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_vs_removed_counters.pdf")
    plt.close(fig)


def plot_phase_labels(
    manifests: dict[str, dict[str, object]],
    summaries: dict[str, dict[str, str]],
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.2, 6.4),
        gridspec_kw={"height_ratios": [4.0, 1.6]},
    )
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.09, top=0.92, wspace=0.42, hspace=0.48)
    for column, set_name in enumerate(SET_INFO):
        manifest = manifests[set_name]
        model = manifest["phase_cluster_models"][0]
        centroids = np.asarray(model["centroids"], dtype=float).T
        safe_counters = manifest["safe_counter_columns"]
        heat_ax = axes[0, column]
        scale = np.max(np.abs(centroids)) or 1.0
        image = heat_ax.imshow(centroids, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
        heat_ax.set_title(set_name)
        heat_ax.set_xticks(range(3))
        heat_ax.set_xticklabels(STATE_LABELS)
        heat_ax.set_yticks(range(len(safe_counters)))
        heat_ax.set_yticklabels([COUNTER_LABELS.get(counter, counter) for counter in safe_counters], fontsize=8)
        for label, counter in zip(heat_ax.get_yticklabels(), safe_counters):
            if counter in PRESSURE_COUNTERS:
                label.set_color("#b22222")
        for x_index in range(3):
            for y_index in range(len(safe_counters)):
                heat_ax.text(
                    x_index,
                    y_index,
                    f"{centroids[y_index, x_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="black",
                )

        pressure_ax = axes[1, column]
        pressure_scores = parse_keyed_pairs(summaries[set_name]["state_order"])
        shares = parse_keyed_pairs(summaries[set_name]["cluster_sizes"])
        total = float(sum(shares.values())) or 1.0
        xs = np.arange(3)
        ys = [pressure_scores[index] for index in xs]
        bars = pressure_ax.bar(xs, ys, color=["#4c78a8", "#72b7b2", "#e45756"], edgecolor="black", linewidth=0.3)
        pressure_ax.axhline(0.0, color="black", linewidth=0.5)
        pressure_ax.set_xticks(xs)
        pressure_ax.set_xticklabels(STATE_LABELS, fontsize=8)
        pressure_ax.set_ylabel("Pressure score", fontsize=8)
        pressure_ax.grid(axis="y", alpha=0.25)
        for bar, index in zip(bars, xs):
            share = 100.0 * shares[index] / total
            y_value = bar.get_height()
            pressure_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_value + (0.15 if y_value >= 0 else -0.35),
                f"{share:.1f}%",
                ha="center",
                va="bottom" if y_value >= 0 else "top",
                fontsize=8,
            )

    colorbar_axis = fig.add_axes([0.90, 0.43, 0.018, 0.42])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Normalized centroid value", fontsize=9)
    fig.savefig(FIG_DIR / "phase_label_clusters.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(rows_by_set: dict[str, list[dict[str, str]]]) -> None:
    models = [model for model, _, _ in MODEL_SPECS]
    accuracy_values = pct(metric_matrix(rows_by_set, "accuracy", models))
    transition_values = pct(metric_matrix(rows_by_set, "transition_case_accuracy", models))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    for axis, values, title in zip(
        axes,
        [accuracy_values, transition_values],
        ["Mean test accuracy (%)", "Mean transition accuracy (%)"],
    ):
        image = axis.imshow(values, cmap="YlGnBu", aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(SET_INFO)))
        axis.set_xticklabels(list(SET_INFO.keys()))
        axis.set_yticks(range(len(models)))
        axis.set_yticklabels([MODEL_LABELS[model] for model in models], fontsize=8)
        for y_index in range(values.shape[0]):
            for x_index in range(values.shape[1]):
                axis.text(
                    x_index,
                    y_index,
                    f"{values[y_index, x_index]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.pdf")
    plt.close(fig)


def plot_hardware_tradeoffs(rows_by_set: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    metric_specs = [
        ("accuracy", "Mean test accuracy (%)"),
        ("transition_case_accuracy", "Mean transition accuracy (%)"),
    ]
    for axis, (metric_key, ylabel) in zip(axes, metric_specs):
        for set_name in SET_INFO:
            rows = rows_by_set[set_name]
            marker = SET_MARKERS[set_name]
            for model, _, color in MODEL_SPECS:
                model_rows = [row for row in rows if row.get("model") == model]
                if not model_rows:
                    continue
                memory = statistics.mean(to_float(row.get("estimated_memory_bytes", "")) for row in model_rows)
                metric_value = statistics.mean(to_float(row.get(metric_key, "")) for row in model_rows)
                axis.scatter(
                    memory,
                    pct(metric_value),
                    s=64,
                    color=color,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.35,
                    alpha=0.9,
                )
        axis.set_xscale("log")
        axis.set_xlabel("Estimated memory (bytes, log scale)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black", markersize=7, label=label)
        for _, label, color in MODEL_SPECS
    ]
    set_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="none", markersize=7, label=set_name)
        for set_name, marker in SET_MARKERS.items()
    ]
    axes[0].legend(handles=model_handles, frameon=False, fontsize=7, loc="lower right")
    axes[1].legend(handles=set_handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardware_cost_vs_accuracy.pdf")
    plt.close(fig)


def plot_set1_thread_scaling() -> None:
    thread_rows = read_csv(SET_INFO["Set 1"]["thread_accuracy"])
    behavior_rows = read_csv(SET_INFO["Set 1"]["thread_behavior"])

    accuracy_by_model: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in thread_rows:
        model = row["model"]
        if model not in {
            "baseline_last_state",
            "online_phase_history_tree",
            "online_phase_history_tree_all_families",
            "tiny_transformer_phase_classifier_all_families",
        }:
            continue
        accuracy_by_model[model][int(row["threads"])].append(float(row["accuracy"]))

    behavior_by_thread: dict[int, dict[str, float]] = {}
    for row in behavior_rows:
        if row["family"] != "L1":
            continue
        threads = int(row["threads"])
        behavior_by_thread[threads] = {
            "transition_rate": float(row["transition_rate"]),
            "avg_run_length": float(row["avg_run_length"]),
        }

    thread_values = sorted(behavior_by_thread)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

    left = axes[0]
    transition = [pct(behavior_by_thread[thread]["transition_rate"]) for thread in thread_values]
    run_length = [behavior_by_thread[thread]["avg_run_length"] for thread in thread_values]
    left.bar([str(thread) for thread in thread_values], transition, color="#4c78a8", alpha=0.85)
    left.set_xlabel("Thread count")
    left.set_ylabel("Transition rate (%)", color="#4c78a8")
    left.tick_params(axis="y", labelcolor="#4c78a8")
    left.grid(axis="y", alpha=0.25)
    right = left.twinx()
    right.plot([str(thread) for thread in thread_values], run_length, color="#e45756", marker="o", linewidth=2.0)
    right.set_ylabel("Average run length", color="#e45756")
    right.tick_params(axis="y", labelcolor="#e45756")
    left.set_title("Set 1 phase behavior")

    axis = axes[1]
    for model in [
        "baseline_last_state",
        "online_phase_history_tree",
        "online_phase_history_tree_all_families",
        "tiny_transformer_phase_classifier_all_families",
    ]:
        values = [
            pct(statistics.mean(accuracy_by_model[model][thread]))
            for thread in thread_values
        ]
        axis.plot(
            thread_values,
            values,
            marker="o",
            linewidth=2.0,
            color=MODEL_COLORS.get(model, "#4c78a8"),
            label=MODEL_LABELS.get(model, model),
        )
    axis.set_xlabel("Thread count")
    axis.set_ylabel("Mean test accuracy (%)")
    axis.set_xticks(thread_values)
    axis.grid(alpha=0.25)
    axis.set_title("Set 1 accuracy by thread count")
    axis.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "set1_thread_scaling.pdf")
    plt.close(fig)


def _policy_streams(set_name: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    streams: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    path = SET_INFO[set_name]["predictions"]
    prefix = f"{CONTROL_MODEL},{CONTROL_FAMILY},"
    with path.open(newline="") as handle:
        header = handle.readline().rstrip("\n").split(",")
        for line in handle:
            if not line.startswith(prefix):
                continue
            values = next(csv.reader([line]))
            row = dict(zip(header, values))
            if row.get("split") == "test":
                streams[(row["run_id"], row["core_id"])].append(row)
    return streams


def _safe_phase(value: str) -> int:
    phase = int(value)
    if phase < 0:
        return 0
    if phase > 2:
        return 2
    return phase


def l1_control_policy_summary() -> list[dict[str, float | str]]:
    """Estimate a tiny L1 control policy from predicted one-step phases.

    This is an opportunity analysis, not a measured silicon power result. The
    policy maps predicted L1 state to a relative L1 power mode and uses a
    two-sample confirmation filter before changing modes.
    """
    output: list[dict[str, float | str]] = []
    for set_name in SET_INFO:
        streams = _policy_streams(set_name)
        total = 0
        prediction_matches = 0
        guarded_power = 0.0
        raw_power = 0.0
        oracle_power = 0.0
        low_power_actions = 0
        underprovisioned = 0
        severe_underprovisioned = 0
        action_changes = 0

        for rows in streams.values():
            rows.sort(key=lambda item: int(item["row_index"]))
            current_action = 1
            pending_action: int | None = None
            pending_count = 0
            previous_action = current_action

            for row in rows:
                actual = _safe_phase(row["y_true_phase_t_plus_1"])
                desired = _safe_phase(row["y_pred_phase_t_plus_1"])
                if desired == current_action:
                    pending_action = None
                    pending_count = 0
                elif desired == pending_action:
                    pending_count += 1
                else:
                    pending_action = desired
                    pending_count = 1
                if pending_count >= 2:
                    current_action = desired
                    pending_action = None
                    pending_count = 0

                if total > 0 and current_action != previous_action:
                    action_changes += 1
                previous_action = current_action
                total += 1
                prediction_matches += int(desired == actual)
                raw_power += L1_POLICY_POWER[desired]
                guarded_power += L1_POLICY_POWER[current_action]
                oracle_power += L1_POLICY_POWER[actual]
                low_power_actions += int(current_action == 0)
                underprovisioned += int(current_action < actual)
                severe_underprovisioned += int(current_action == 0 and actual == 2)

        full_power = float(total) or 1.0
        guarded_saving = 100.0 * (1.0 - guarded_power / full_power)
        oracle_saving = 100.0 * (1.0 - oracle_power / full_power)
        output.append(
            {
                "set": set_name,
                "rows": float(total),
                "prediction_accuracy_pct": 100.0 * prediction_matches / full_power,
                "raw_policy_saving_pct": 100.0 * (1.0 - raw_power / full_power),
                "guarded_policy_saving_pct": guarded_saving,
                "oracle_policy_saving_pct": oracle_saving,
                "oracle_saving_captured_pct": 100.0 * guarded_saving / oracle_saving if oracle_saving > 0 else 0.0,
                "low_power_action_pct": 100.0 * low_power_actions / full_power,
                "underprovision_pct": 100.0 * underprovisioned / full_power,
                "severe_underprovision_pct": 100.0 * severe_underprovisioned / full_power,
                "action_changes_per_1k": 1000.0 * action_changes / full_power,
            }
        )
    return output


def plot_l1_control_policy(rows: list[dict[str, float | str]]) -> None:
    sets = [str(row["set"]) for row in rows]
    x = np.arange(len(sets))
    width = 0.26
    guarded = [float(row["guarded_policy_saving_pct"]) for row in rows]
    oracle = [float(row["oracle_policy_saving_pct"]) for row in rows]
    risk = [float(row["underprovision_pct"]) for row in rows]

    fig, axis = plt.subplots(figsize=(8.0, 3.4))
    axis.bar(x - width, guarded, width, color="#4c78a8", label="Guarded policy saving")
    axis.bar(x, oracle, width, color="#72b7b2", label="Oracle saving")
    axis.bar(x + width, risk, width, color="#e45756", label="Underprovision risk")
    axis.set_xticks(x)
    axis.set_xticklabels(sets)
    axis.set_ylabel("Intervals or L1 power proxy (%)")
    axis.set_ylim(0, max(oracle + risk + guarded) * 1.22)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.set_title("Small predicted-phase L1 control-policy simulation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "l1_control_policy.pdf")
    plt.close(fig)

    csv_path = FIG_DIR / "l1_control_policy_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_raw_counter_phase_boundaries() -> None:
    """Show raw selected-counter traces with resource-family phase boundaries."""
    base = Path(
        "/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/"
        "counter_sequences/config_group_holdout/threshold_global/counter_value_sequences"
    )
    run_id = (
        "parsec_family_set1_single_process_multithread_r5_set1_r5_"
        "streamcluster_t16_streamcluster_p0_t16_20260429T215149Z_30ce9bb46a"
    )
    core_id = "1:2"
    start = 0
    window = 85
    family_specs = [
        ("counter__l1d_loads", "L1 data loads", "#4c78a8"),
        ("counter__llc_references", "LLC references", "#54a24b"),
        ("counter__branch_instructions", "Branch instructions", "#f58518"),
        ("counter__instructions_retired", "Instructions retired", "#b279a2"),
        ("counter__offcore_demand_data_reads", "Offcore demand reads", "#e45756"),
    ]
    state_colors = {
        0: "#e8f1fb",
        1: "#fff4cf",
        2: "#fde0dd",
    }
    phase_labels = ["Phase 1", "Phase 2", "Phase 3"]

    traces: dict[str, list[dict[str, str]]] = {}
    for counter, _, _ in family_specs:
        path = base / f"counter_sequence_{counter}.csv"
        rows: list[dict[str, str]] = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["run_id"] == run_id and row["core_id"] == core_id:
                    rows.append(row)
                    if len(rows) >= start + window:
                        break
        if len(rows) < start + window:
            raise RuntimeError(f"not enough rows for {counter}: found {len(rows)}")
        traces[counter] = rows[start : start + window]

    reference_times = np.asarray(
        [to_float(row["timestamp_ms"]) for row in traces[family_specs[0][0]]],
        dtype=float,
    )
    times = reference_times - reference_times[0]

    fig, axes = plt.subplots(
        len(family_specs),
        1,
        figsize=(10.8, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    for axis, (counter, ylabel, color) in zip(axes, family_specs):
        rows = traces[counter]
        values = np.asarray([to_float(row["counter_value"]) for row in rows], dtype=float)
        states = [int(row["family_state"]) for row in rows]
        positive_values = values[np.isfinite(values) & (values > 0)]
        if positive_values.size == 0:
            raise RuntimeError(f"no positive values to plot for {counter}")
        values[~np.isfinite(values) | (values <= 0)] = np.nan

        span_start = 0
        for index in range(1, len(states) + 1):
            if index == len(states) or states[index] != states[span_start]:
                axis.axvspan(
                    times[span_start],
                    times[index - 1] if index < len(states) else times[-1],
                    color=state_colors.get(states[span_start], "#eeeeee"),
                    alpha=0.55,
                    linewidth=0,
                )
                if index < len(states):
                    axis.axvline(times[index], color="black", linestyle="--", linewidth=0.6, alpha=0.45)
                span_start = index

        axis.plot(times, values, color=color, linewidth=1.35)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="both", labelsize=8)
        axis.text(
            0.995,
            0.78,
            COUNTER_LABELS.get(rows[0]["counter_name"], rows[0]["counter_name"]),
            ha="right",
            va="center",
            transform=axis.transAxes,
            fontsize=8,
            color="#333333",
        )

    axes[-1].set_xlabel("Time within selected window (ms)")
    axes[0].set_title(
        "Raw selected-counter traces reveal repeated Phase 1/2/3 regions "
        "(Set 1, streamcluster, 16 threads, one physical core)",
        fontsize=10,
    )
    handles = [
        Line2D([0], [0], color=state_colors[index], linewidth=8, label=phase_labels[index])
        for index in range(3)
    ] + [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Phase boundary")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(FIG_DIR / "raw_counter_phase_boundaries.pdf")
    plt.close(fig)


def write_summary_json(
    rows_by_set: dict[str, list[dict[str, str]]],
    ablations: dict[str, list[dict[str, str]]],
    manifests: dict[str, dict[str, object]],
    summaries: dict[str, dict[str, str]],
    raw_runs: dict[str, list[dict[str, object]]],
    control_rows: list[dict[str, float | str]],
) -> None:
    output: dict[str, object] = {"sets": {}, "models": {}, "l1_control_policy": control_rows}
    for set_name in SET_INFO:
        rows = rows_by_set[set_name]
        selected = selected_counters(ablations)[set_name]
        cluster_row = summaries[set_name]
        output["sets"][set_name] = {
            "run_count": len(raw_runs[set_name]),
            "rows": int(cluster_row["rows"]),
            "train_rows": int(cluster_row["train_rows"]),
            "thread_values": sorted({item["threads"] for item in raw_runs[set_name]}),
            "process_counts": sorted({item["process_count"] for item in raw_runs[set_name]}),
            "input_sizes": sorted({item["requested_input_size"] for item in raw_runs[set_name]}),
            "selected_counters": selected,
            "cluster_state_order": cluster_row["state_order"],
            "cluster_sizes": cluster_row["cluster_sizes"],
        }
        for model, _, _ in MODEL_SPECS:
            model_rows = [row for row in rows if row["model"] == model]
            if not model_rows:
                continue
            output["sets"][set_name][model] = {
                "accuracy": statistics.mean(float(row["accuracy"]) for row in model_rows),
                "macro_f1": statistics.mean(float(row["macro_f1"]) for row in model_rows),
                "balanced_accuracy": statistics.mean(float(row["balanced_accuracy"]) for row in model_rows),
                "transition_accuracy": statistics.mean(float(row["transition_case_accuracy"]) for row in model_rows),
                "estimated_memory_bytes": statistics.mean(float(row["estimated_memory_bytes"]) for row in model_rows),
                "inference_latency_us": statistics.mean(float(row["inference_latency_us"]) for row in model_rows),
            }

    for model, label, _ in MODEL_SPECS:
        per_set_entries = [output["sets"][set_name][model] for set_name in SET_INFO if model in output["sets"][set_name]]
        output["models"][model] = {
            "label": label,
            "mean_accuracy": statistics.mean(item["accuracy"] for item in per_set_entries),
            "mean_macro_f1": statistics.mean(item["macro_f1"] for item in per_set_entries),
            "mean_balanced_accuracy": statistics.mean(item["balanced_accuracy"] for item in per_set_entries),
            "mean_transition_accuracy": statistics.mean(item["transition_accuracy"] for item in per_set_entries),
            "mean_estimated_memory_bytes": statistics.mean(item["estimated_memory_bytes"] for item in per_set_entries),
            "mean_inference_latency_us": statistics.mean(item["inference_latency_us"] for item in per_set_entries),
        }

    (FIG_DIR / "paper_results_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-l1-policy",
        action="store_true",
        help="regenerate only the pilot L1 control-policy figure and CSV",
    )
    args = parser.parse_args()

    if args.only_l1_policy:
        control_rows = l1_control_policy_summary()
        plot_l1_control_policy(control_rows)
        print(f"[figures] wrote L1 control-policy outputs to {FIG_DIR}")
        return

    rows_by_set = summary_rows()
    ablations = ablation_rows()
    manifests = cluster_models()
    summaries = cluster_summaries()
    raw_runs = raw_manifests()

    plot_counter_ablation(ablations)
    plot_phase_labels(manifests, summaries)
    plot_model_comparison(rows_by_set)
    plot_hardware_tradeoffs(rows_by_set)
    plot_set1_thread_scaling()
    control_rows = l1_control_policy_summary()
    plot_l1_control_policy(control_rows)
    plot_raw_counter_phase_boundaries()
    write_summary_json(rows_by_set, ablations, manifests, summaries, raw_runs, control_rows)
    print(f"[figures] wrote refreshed figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
