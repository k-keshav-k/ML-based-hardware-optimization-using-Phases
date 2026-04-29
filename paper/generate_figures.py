#!/usr/bin/env python3
"""Generate paper figures from local phase-family artifacts.

The script intentionally uses only the standard library plus numpy/matplotlib so
it can run in the current project environment without pandas.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

FAMILIES = ["L1", "LLC", "branch_control", "core_fp", "memory_offcore"]
FAMILY_LABELS = {
    "L1": "L1",
    "LLC": "LLC",
    "branch_control": "Branch",
    "core_fp": "Core/FP",
    "memory_offcore": "Memory",
}
STATE_LABELS = ["Low", "Moderate", "High"]
SET_INFO = {
    "Set 1": {
        "summary": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "depth_sweep": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/phase_detector_depth_sweep_summary_all.csv"),
        "manifest": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_manifest.json"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/ablation/family_ablation_results.csv"),
    },
    "Set 2": {
        "summary": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "depth_sweep": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/phase_detector_depth_sweep_summary_all.csv"),
        "manifest": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_manifest.json"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/ablation/family_ablation_results.csv"),
    },
    "Set 3": {
        "summary": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv"),
        "depth_sweep": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/phase_detector_depth_sweep_summary_all.csv"),
        "manifest": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_manifest.json"),
        "ablation": Path("/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/ablation/family_ablation_results.csv"),
    },
}

TREE_MODEL = "online_phase_history_tree"
BASELINE_MODELS = [
    ("baseline_last_state", "Last state", "#f58518"),
    ("baseline_majority", "Majority", "#bab0ac"),
    ("baseline_state_conditioned_majority", "Cond. majority", "#54a24b"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float) -> float:
    return 100.0 * value


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0


def high_recall(y_true: list[int], y_pred: list[int]) -> float:
    positives = [i for i, value in enumerate(y_true) if value == 2]
    if not positives:
        return 0.0
    return sum(1 for i in positives if y_pred[i] == 2) / len(positives)


def detector_summary_rows() -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for set_name, info in SET_INFO.items():
        rows = read_csv(info["summary"])
        output[set_name] = [
            row
            for row in rows
            if row.get("experiment") == "config_group_holdout" and row.get("scope") == "global"
        ]
    return output


def detector_manifests() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for set_name, info in SET_INFO.items():
        output[set_name] = json.loads(info["manifest"].read_text())
    return output


def detector_depth_sweep_rows() -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for set_name, info in SET_INFO.items():
        path = info["depth_sweep"]
        rows = read_csv(path) if path.exists() else []
        output[set_name] = [
            row
            for row in rows
            if row.get("experiment") == "config_group_holdout" and row.get("scope") == "global"
        ]
    return output


def _mean_metric(rows: list[dict[str, str]], model: str, key: str) -> float:
    values = [to_float(row.get(key, "")) for row in rows if row.get("model") == model]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else 0.0


def ablation_rows() -> dict[str, list[dict[str, str]]]:
    return {name: read_csv(info["ablation"]) for name, info in SET_INFO.items()}


def selected_counters(rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for set_name, set_rows in rows.items():
        selected[set_name] = {}
        for row in set_rows:
            if row.get("candidate_type") == "singleton" and row.get("selected") == "1":
                selected[set_name][row["family"]] = row["counter_set"].replace("counter__", "")
    return selected


def plot_family_metrics(metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.36
    for ax, set_name in zip(axes, SET_INFO):
        acc = [pct(metrics[set_name][family]["accuracy"]) for family in FAMILIES]
        f1 = [pct(metrics[set_name][family]["macro_f1"]) for family in FAMILIES]
        ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4c78a8")
        ax.bar(x + width / 2, f1, width, label="Macro F1", color="#f58518")
        ax.set_title(set_name)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation score (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "family_accuracy_f1.pdf")
    plt.close(fig)


def plot_counter_ablation(rows: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    for ax, (set_name, set_rows) in zip(axes, rows.items()):
        points = []
        labels = []
        colors = []
        for family in FAMILIES:
            family_rows = [
                r
                for r in set_rows
                if r.get("family") == family and r.get("candidate_type") == "singleton" and r.get("counter_set")
            ]
            for row in family_rows:
                points.append(pct(to_float(row.get("validation_score", ""))))
                label = row["counter_set"].replace("counter__", "").replace("_", "\n")
                labels.append(label)
                colors.append("#54a24b" if row.get("selected") == "1" else "#bab0ac")
        ax.bar(np.arange(len(points)), points, color=colors)
        ax.set_title(set_name)
        ax.set_xticks(np.arange(len(points)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Ablation validation score (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selected_vs_removed_counters.pdf")
    plt.close(fig)


def plot_counter_count(rows: dict[str, list[dict[str, str]]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    markers = {"Set 1": "o", "Set 2": "s", "Set 3": "^"}
    for set_index, (set_name, set_rows) in enumerate(rows.items()):
        xs = []
        ys = []
        colors = []
        for row in set_rows:
            if row.get("candidate_type") == "singleton" and row.get("counter_set"):
                xs.append(1 + (set_index - 1) * 0.05)
                ys.append(pct(to_float(row.get("validation_score", ""))))
                colors.append("#4c78a8" if row.get("selected") == "1" else "#9ecae9")
            elif row.get("candidate_type") == "one_per_family_exhaustive" and row.get("counter_set"):
                count = len([item for item in row["counter_set"].split(",") if item])
                xs.append(count + (set_index - 1) * 0.05)
                ys.append(pct(to_float(row.get("validation_score", ""))))
                colors.append("#e45756" if row.get("selected") == "1" else "#f2a19d")
        ax.scatter(xs, ys, s=34, marker=markers[set_name], label=set_name, c=colors, edgecolors="black", linewidths=0.25)
    ax.set_xlabel("Counters in ablation candidate")
    ax.set_ylabel("Validation score (%)")
    ax.set_xticks([1, 5])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_vs_counter_count.pdf")
    plt.close(fig)


def plot_baselines(summary_rows: dict[str, list[dict[str, str]]]) -> None:
    metric_specs = [
        ("accuracy", "Mean accuracy (%)"),
        ("macro_f1", "Mean macro F1 (%)"),
        ("transition_case_accuracy", "Mean transition accuracy (%)"),
    ]
    model_specs = [
        (TREE_MODEL, "Tree", "#4c78a8"),
        *BASELINE_MODELS,
    ]
    x = np.arange(len(SET_INFO))
    width = 0.18
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for index, (model_key, label, color) in enumerate(model_specs):
            values = [
                pct(_mean_metric(summary_rows[set_name], model_key, metric_key))
                for set_name in SET_INFO
            ]
            ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(list(SET_INFO.keys()))
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_comparison.pdf")
    plt.close(fig)


def plot_depth_and_cost(
    summary_rows: dict[str, list[dict[str, str]]],
    manifests: dict[str, dict[str, object]],
    depth_sweep_rows: dict[str, list[dict[str, str]]],
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = {"Set 1": "#4c78a8", "Set 2": "#f58518", "Set 3": "#54a24b"}
    used_sweep = any(depth_sweep_rows[set_name] for set_name in SET_INFO)
    for set_name, manifest in manifests.items():
        sweep = sorted(depth_sweep_rows[set_name], key=lambda row: to_float(row.get("tree_max_depth", "")))
        if sweep:
            xs = [to_float(row.get("tree_max_depth", "")) for row in sweep]
            ys = [pct(to_float(row.get("mean_accuracy", ""))) for row in sweep]
            ax.plot(xs, ys, marker="o", label=set_name, color=colors[set_name])
            best_index = int(np.argmax(ys))
            ax.scatter([xs[best_index]], [ys[best_index]], s=72, color=colors[set_name], edgecolors="black", linewidths=0.4)
        else:
            depth = float(manifest["tree_depth"])
            mean_acc = pct(_mean_metric(summary_rows[set_name], TREE_MODEL, "accuracy"))
            ax.scatter([depth], [mean_acc], label=set_name, s=58, color=colors[set_name])
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Mean accuracy (%)")
    if used_sweep:
        all_depths = [
            to_float(row.get("tree_max_depth", ""))
            for rows in depth_sweep_rows.values()
            for row in rows
        ]
        all_depths = [depth for depth in all_depths if math.isfinite(depth)]
        if all_depths:
            ax.set_xticks(sorted({int(depth) for depth in all_depths}))
            ax.set_xlim(min(all_depths) - 0.3, max(all_depths) + 0.3)
    else:
        ax.set_xticks([6])
        ax.set_xlim(5.7, 6.3)
    ax.set_ylim(55, 90)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tree_depth_vs_accuracy.pdf")
    plt.close(fig)

    cost_summary: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for set_name, manifest in manifests.items():
        sweep = sorted(depth_sweep_rows[set_name], key=lambda row: to_float(row.get("estimated_storage_bytes", "")))
        if sweep:
            xs = [to_float(row.get("estimated_storage_bytes", "")) for row in sweep]
            ys = [pct(to_float(row.get("mean_accuracy", ""))) for row in sweep]
            cost_summary[set_name] = xs[-1]
            ax.plot(xs, ys, marker="o", label=set_name, color=colors[set_name])
            best_index = int(np.argmax(ys))
            ax.scatter([xs[best_index]], [ys[best_index]], s=72, color=colors[set_name], edgecolors="black", linewidths=0.4)
            ax.annotate(
                f"{set_name} best d={int(to_float(sweep[best_index].get('tree_max_depth', '0')))}",
                (xs[best_index], ys[best_index]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )
        else:
            storage_bytes = float(manifest["estimated_storage_bytes"])
            cost_summary[set_name] = storage_bytes
            mean_acc = pct(_mean_metric(summary_rows[set_name], TREE_MODEL, "accuracy"))
            ax.scatter([storage_bytes], [mean_acc], label=f"{set_name} ({storage_bytes:.0f} B)", s=58, color=colors[set_name])
            ax.annotate(set_name, (storage_bytes, mean_acc), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Estimated horizon-detector storage (bytes)")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_ylim(55, 90)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardware_cost_vs_accuracy.pdf")
    plt.close(fig)
    return cost_summary


def write_summary(
    summary_rows: dict[str, list[dict[str, str]]],
    ablations: dict[str, list[dict[str, str]]],
    selected: dict[str, dict[str, str]],
    manifests: dict[str, dict[str, object]],
    cost_summary: dict[str, float],
) -> None:
    output: dict[str, object] = {"sets": {}, "selected_counters": selected, "storage_bytes": cost_summary}
    for set_name in SET_INFO:
        rows = summary_rows[set_name]
        output["sets"][set_name] = {
            "history_length": int(manifests[set_name]["history_length"]),
            "prediction_horizon": int(manifests[set_name]["prediction_horizon"]),
            "tree_depth": int(manifests[set_name]["tree_depth"]),
            "mean_accuracy": _mean_metric(rows, TREE_MODEL, "accuracy"),
            "mean_macro_f1": _mean_metric(rows, TREE_MODEL, "macro_f1"),
            "mean_high_recall": _mean_metric(rows, TREE_MODEL, "high_usage_recall"),
            "mean_stable_accuracy": _mean_metric(rows, TREE_MODEL, "stable_case_accuracy"),
            "mean_transition_accuracy": _mean_metric(rows, TREE_MODEL, "transition_case_accuracy"),
            "tree_baseline_accuracy": _mean_metric(rows, TREE_MODEL, "accuracy"),
            "last_baseline_accuracy": _mean_metric(rows, "baseline_last_state", "accuracy"),
            "majority_baseline_accuracy": _mean_metric(rows, "baseline_majority", "accuracy"),
            "state_conditioned_majority_accuracy": _mean_metric(rows, "baseline_state_conditioned_majority", "accuracy"),
            "tree_baseline_macro_f1": _mean_metric(rows, TREE_MODEL, "macro_f1"),
            "last_baseline_macro_f1": _mean_metric(rows, "baseline_last_state", "macro_f1"),
            "majority_baseline_macro_f1": _mean_metric(rows, "baseline_majority", "macro_f1"),
            "state_conditioned_majority_macro_f1": _mean_metric(rows, "baseline_state_conditioned_majority", "macro_f1"),
        }
        selected_global = [r for r in ablations[set_name] if r.get("candidate_type") == "one_per_family_exhaustive" and r.get("selected") == "1"]
        if selected_global:
            output["sets"][set_name]["selected_global_validation_score"] = to_float(selected_global[0].get("validation_score", ""))
    (FIG_DIR / "paper_results_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    summary_rows = detector_summary_rows()
    manifests = detector_manifests()
    depth_sweep_rows = detector_depth_sweep_rows()
    ablations = ablation_rows()
    selected = selected_counters(ablations)
    plot_counter_ablation(ablations)
    plot_counter_count(ablations)
    plot_baselines(summary_rows)
    cost_summary = plot_depth_and_cost(summary_rows, manifests, depth_sweep_rows)
    write_summary(summary_rows, ablations, selected, manifests, cost_summary)
    print(f"[figures] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
