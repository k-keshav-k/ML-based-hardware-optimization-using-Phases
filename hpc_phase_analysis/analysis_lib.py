"""Correlation analysis, plotting, and recommendations."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, rankdata

from .constants import COUNTER_FAMILIES, DEFAULT_CORRELATION_THRESHOLDS, METADATA_COLUMNS
from .io_utils import ensure_dir, load_csv_rows, safe_float, write_csv_rows, write_json


def load_numeric_view(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    rows = load_csv_rows(path)
    columns = [key for key in rows[0].keys() if key not in METADATA_COLUMNS] if rows else []
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            matrix[i, j] = safe_float(row.get(column, ""))
    return rows, columns, matrix


def drop_nan_columns(columns: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    keep = [index for index in range(matrix.shape[1]) if not np.all(np.isnan(matrix[:, index]))]
    return [columns[index] for index in keep], matrix[:, keep]


def correlation_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0))
    if method == "pearson":
        return np.corrcoef(matrix, rowvar=False)
    if method == "spearman":
        ranked = np.apply_along_axis(rankdata, 0, matrix)
        return np.corrcoef(ranked, rowvar=False)
    if method == "kendall":
        size = matrix.shape[1]
        output = np.eye(size)
        for i in range(size):
            for j in range(i + 1, size):
                value = kendalltau(matrix[:, i], matrix[:, j], nan_policy="omit").correlation
                output[i, j] = value
                output[j, i] = value
        return output
    raise ValueError(f"Unsupported method: {method}")


def vif_scores(matrix: np.ndarray, columns: list[str]) -> list[dict[str, object]]:
    scores: list[dict[str, object]] = []
    if matrix.shape[1] <= 1:
        return scores
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    corr = corr + np.eye(corr.shape[0]) * 1e-8
    try:
        inverse = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        inverse = np.eye(corr.shape[0]) * 1e6
    for index, column in enumerate(columns):
        vif = float(inverse[index, index])
        if vif < 0:
            vif = math.inf
        scores.append({"indicator": column, "vif": vif})
    return sorted(scores, key=lambda item: item["vif"], reverse=True)


def correlation_pairs(columns: list[str], matrix: np.ndarray, method: str) -> list[dict[str, object]]:
    corr = correlation_matrix(matrix, method)
    pairs: list[dict[str, object]] = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        value = float(corr[i, j])
        pairs.append(
            {
                "indicator_a": columns[i],
                "indicator_b": columns[j],
                "correlation": value,
                "abs_correlation": abs(value),
                "method": method,
            }
        )
    return sorted(pairs, key=lambda item: item["abs_correlation"], reverse=True)


def connected_components(columns: list[str], corr: np.ndarray, threshold: float) -> list[list[str]]:
    adjacency = {column: set() for column in columns}
    for i, j in itertools.combinations(range(len(columns)), 2):
        if abs(corr[i, j]) >= threshold:
            adjacency[columns[i]].add(columns[j])
            adjacency[columns[j]].add(columns[i])
    seen = set()
    groups = []
    for column in columns:
        if column in seen:
            continue
        stack = [column]
        component = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)
        if len(component) > 1:
            groups.append(sorted(component))
    return groups


def figure_size(columns: list[str]) -> tuple[float, float]:
    edge = max(6.0, min(18.0, 0.55 * len(columns)))
    return edge, edge


def display_indicator(indicator: str) -> str:
    family_name = indicator
    suffix = ""
    if indicator.startswith("counter__"):
        family_name = indicator[len("counter__") :]
    elif indicator.startswith("norm__"):
        family_name = indicator[len("norm__") :]
    elif indicator.startswith("metric__"):
        return indicator[len("metric__") :].replace("_", " ").title()
    if family_name.endswith("_per_ms"):
        suffix = " / ms"
        family_name = family_name[: -len("_per_ms")]
    elif family_name.endswith("_per_ki"):
        suffix = " / KI"
        family_name = family_name[: -len("_per_ki")]
    label = COUNTER_FAMILIES.get(family_name, {}).get("label", family_name.replace("_", " ").title())
    return f"{label}{suffix}"


def plot_heatmap(corr: np.ndarray, columns: list[str], title: str, base_path: Path) -> None:
    if corr.size == 0:
        return
    fig, ax = plt.subplots(figsize=figure_size(columns))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    display_columns = [display_indicator(column) for column in columns]
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(display_columns, rotation=90, fontsize=10)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(display_columns, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_clustered_heatmap(corr: np.ndarray, columns: list[str], title: str, heatmap_path: Path, dendrogram_path: Path) -> list[str]:
    if corr.size == 0 or len(columns) < 2:
        return columns
    distance = 1.0 - np.clip(np.abs(corr), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    order = leaves_list(linkage_matrix)
    ordered_columns = [columns[index] for index in order]
    ordered_corr = corr[np.ix_(order, order)]
    plot_heatmap(ordered_corr, ordered_columns, title, heatmap_path)

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(columns)), 5))
    dendrogram(linkage_matrix, labels=columns, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram")
    fig.tight_layout()
    fig.savefig(dendrogram_path.with_suffix(".png"), dpi=220)
    fig.savefig(dendrogram_path.with_suffix(".pdf"))
    plt.close(fig)
    return ordered_columns


def plot_vif(vif_rows: list[dict[str, object]], base_path: Path) -> None:
    if not vif_rows:
        return
    labels = [display_indicator(str(row["indicator"])) for row in vif_rows[:20]]
    raw_values = [float(row["vif"]) for row in vif_rows[:20]]
    finite_values = [value for value in raw_values if math.isfinite(value)]
    ceiling = min(max(finite_values) * 1.1 if finite_values else 100.0, 100.0)
    if ceiling <= 0:
        ceiling = 10.0
    values = [min(value, ceiling) if math.isfinite(value) else ceiling for value in raw_values]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 6))
    ax.bar(range(len(labels)), values, color="#2e86ab")
    ax.set_title("Variance inflation factors")
    ax.set_ylabel("VIF")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_top_correlations(pairs: list[dict[str, object]], base_path: Path) -> None:
    if not pairs:
        return
    selected = pairs[:10] + list(reversed(sorted(pairs, key=lambda item: item["correlation"])[:10]))
    labels = [f"{display_indicator(str(row['indicator_a']))} vs {display_indicator(str(row['indicator_b']))}" for row in selected]
    values = [row["correlation"] for row in selected]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(labels))))
    colors = ["#3a7d44" if value >= 0 else "#b23a48" for value in values]
    ax.barh(labels, values, color=colors)
    ax.set_xlim(-1, 1)
    ax.set_title("Top positive and negative correlations")
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_distributions(matrix: np.ndarray, columns: list[str], base_path: Path) -> None:
    if matrix.size == 0:
        return
    top_columns = columns[: min(6, len(columns))]
    fig, axes = plt.subplots(len(top_columns), 1, figsize=(10, 2.4 * len(top_columns)))
    if len(top_columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, top_columns):
        index = columns.index(column)
        values = matrix[:, index]
        axis.hist(values[~np.isnan(values)], bins=30, color="#4c78a8", alpha=0.85)
        axis.set_title(display_indicator(column))
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_scatter_matrix(matrix: np.ndarray, columns: list[str], base_path: Path) -> None:
    if len(columns) < 2:
        return
    chosen = columns[: min(4, len(columns))]
    indices = [columns.index(name) for name in chosen]
    fig, axes = plt.subplots(len(chosen), len(chosen), figsize=(3 * len(chosen), 3 * len(chosen)))
    for i, row_name in enumerate(chosen):
        for j, col_name in enumerate(chosen):
            ax = axes[i, j]
            x = matrix[:, indices[j]]
            y = matrix[:, indices[i]]
            if i == j:
                ax.hist(x[~np.isnan(x)], bins=20, color="#4c78a8", alpha=0.8)
            else:
                ax.scatter(x, y, s=8, alpha=0.45, color="#f58518")
            if i == len(chosen) - 1:
                ax.set_xlabel(display_indicator(col_name), rotation=45)
            if j == 0:
                ax.set_ylabel(display_indicator(row_name))
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def feature_score(indicator: str, missing_fraction: float) -> float:
    family = indicator.replace("counter__", "").replace("norm__", "")
    family = family.replace("_per_ki", "").replace("_per_ms", "")
    metadata = COUNTER_FAMILIES.get(family, {"portability": 5, "interpretability": 5, "collection_cost": 3})
    return (
        2.5 * float(metadata["portability"])
        + 2.0 * float(metadata["interpretability"])
        - 1.5 * float(metadata["collection_cost"])
        - 25.0 * missing_fraction
    )


def recommend_representatives(
    groups: list[list[str]],
    missingness: dict[str, float],
    corr_lookup: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    for group_id, group in enumerate(groups, start=1):
        scored = sorted(group, key=lambda item: feature_score(item, missingness.get(item, 0.0)), reverse=True)
        keep = scored[0]
        drop = scored[1:]
        reason = (
            f"Selected for higher portability/interpretability and lower missingness; "
            f"max within-group |rho|={max(abs(corr_lookup.get(tuple(sorted((keep, other))), 0.0)) for other in drop):.3f}"
        )
        recommendations.append(
            {
                "group_id": f"G{group_id:02d}",
                "candidate_indicators": ", ".join(group),
                "representative_to_keep": keep,
                "indicators_to_drop": ", ".join(drop),
                "reason": reason,
            }
        )
    return recommendations


def grouped_subset(rows: list[dict[str, str]], matrix: np.ndarray, columns: list[str], key: str) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(key, "")
        if value:
            buckets[value].append(index)
    return {bucket: matrix[indices, :] for bucket, indices in buckets.items() if len(indices) >= 5 and len(columns) >= 2}


def plot_stability(grouped_corrs: dict[str, list[dict[str, object]]], base_path: Path, pair_count: int = 5) -> None:
    if not grouped_corrs:
        return
    pair_names = []
    for rows in grouped_corrs.values():
        for row in rows[:pair_count]:
            name = f"{row['indicator_a']} vs {row['indicator_b']}"
            if name not in pair_names:
                pair_names.append(name)
            if len(pair_names) >= pair_count:
                break
        if len(pair_names) >= pair_count:
            break
    if not pair_names:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = list(grouped_corrs.keys())
    for pair_name in pair_names:
        values = []
        for group in groups:
            value = math.nan
            for row in grouped_corrs[group]:
                name = f"{row['indicator_a']} vs {row['indicator_b']}"
                if name == pair_name:
                    value = row["correlation"]
                    break
            values.append(value)
        ax.plot(groups, values, marker="o", label=pair_name)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Correlation stability across groups")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=220)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def analyze_view(
    view_name: str,
    csv_path: Path,
    output_root: Path,
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    thresholds = thresholds or DEFAULT_CORRELATION_THRESHOLDS
    rows, columns, matrix = load_numeric_view(csv_path)
    columns, matrix = drop_nan_columns(columns, matrix)
    output_dir = ensure_dir(output_root / view_name)
    summary: dict[str, object] = {
        "view_name": view_name,
        "row_count": len(rows),
        "indicator_count": len(columns),
        "csv_path": str(csv_path),
        "plots": {},
    }
    if matrix.shape[0] < 5 or matrix.shape[1] < 2:
        write_json(output_dir / "analysis_summary.json", summary)
        return summary

    pearson = correlation_matrix(matrix, "pearson")
    spearman = correlation_matrix(matrix, "spearman")
    kendall = correlation_matrix(matrix, "kendall")

    np.savetxt(output_dir / "pearson_matrix.csv", pearson, delimiter=",")
    np.savetxt(output_dir / "spearman_matrix.csv", spearman, delimiter=",")
    np.savetxt(output_dir / "kendall_matrix.csv", kendall, delimiter=",")
    write_csv_rows(output_dir / "indicator_columns.csv", [{"indicator": column} for column in columns], ["indicator"])

    plot_heatmap(pearson, columns, f"{view_name} Pearson heatmap", output_dir / "pearson_heatmap")
    plot_heatmap(spearman, columns, f"{view_name} Spearman heatmap", output_dir / "spearman_heatmap")
    ordered_columns = plot_clustered_heatmap(spearman, columns, f"{view_name} clustered Spearman heatmap", output_dir / "clustered_heatmap", output_dir / "dendrogram")
    pairs = correlation_pairs(columns, spearman, "spearman")
    write_csv_rows(output_dir / "top_correlations.csv", pairs)
    plot_top_correlations(pairs, output_dir / "top_correlations")
    plot_distributions(matrix, columns, output_dir / "indicator_distributions")
    plot_scatter_matrix(matrix, columns, output_dir / "scatter_matrix")

    vif_rows = vif_scores(matrix, columns)
    write_csv_rows(output_dir / "vif.csv", vif_rows, ["indicator", "vif"])
    plot_vif(vif_rows, output_dir / "vif")

    groups_lookup: dict[str, list[list[str]]] = {}
    corr_lookup: dict[tuple[str, str], float] = {}
    for pair in pairs:
        corr_lookup[tuple(sorted((str(pair["indicator_a"]), str(pair["indicator_b"]))))] = float(pair["correlation"])
    for threshold in thresholds:
        groups = connected_components(columns, spearman, threshold)
        groups_lookup[f"{threshold:.2f}"] = groups
        write_csv_rows(
            output_dir / f"correlation_groups_{int(threshold * 100):02d}.csv",
            [
                {"group_id": f"G{index:02d}", "members": ", ".join(group), "size": len(group)}
                for index, group in enumerate(groups, start=1)
            ],
            ["group_id", "members", "size"],
        )

    grouped_by_workload = {
        key: correlation_pairs(columns, subset, "spearman")
        for key, subset in grouped_subset(rows, matrix, columns, "workload").items()
    }
    grouped_by_phase = {
        key: correlation_pairs(columns, subset, "spearman")
        for key, subset in grouped_subset(rows, matrix, columns, "phase_label").items()
    }
    grouped_by_threads = {
        key: correlation_pairs(columns, subset, "spearman")
        for key, subset in grouped_subset(rows, matrix, columns, "threads").items()
    }

    plot_stability(grouped_by_workload, output_dir / "stability_by_workload")
    plot_stability(grouped_by_phase, output_dir / "stability_by_phase")

    for bucket, subset in grouped_subset(rows, matrix, columns, "workload").items():
        plot_heatmap(correlation_matrix(subset, "spearman"), columns, f"{view_name} Spearman by workload: {bucket}", output_dir / f"heatmap_workload_{bucket}")
    for bucket, subset in grouped_subset(rows, matrix, columns, "phase_label").items():
        plot_heatmap(correlation_matrix(subset, "spearman"), columns, f"{view_name} Spearman by phase: {bucket}", output_dir / f"heatmap_phase_{bucket}")

    summary["plots"] = {
        "pearson": str((output_dir / "pearson_heatmap.png").relative_to(output_root.parent)),
        "spearman": str((output_dir / "spearman_heatmap.png").relative_to(output_root.parent)),
        "clustered": str((output_dir / "clustered_heatmap.png").relative_to(output_root.parent)),
        "dendrogram": str((output_dir / "dendrogram.png").relative_to(output_root.parent)),
        "vif": str((output_dir / "vif.png").relative_to(output_root.parent)),
        "top_correlations": str((output_dir / "top_correlations.png").relative_to(output_root.parent)),
    }
    summary["groups"] = groups_lookup
    summary["top_pairs"] = pairs[:20]
    summary["ordered_columns"] = ordered_columns
    summary["columns"] = columns
    summary["workload_groups"] = list(grouped_by_workload.keys())
    summary["phase_groups"] = list(grouped_by_phase.keys())
    summary["thread_groups"] = list(grouped_by_threads.keys())
    write_json(output_dir / "analysis_summary.json", summary)
    return summary


def analyze_all_views(preprocess_dir: Path, output_root: Path, preprocess_summary: dict[str, object]) -> dict[str, object]:
    views = {
        "raw": preprocess_dir / "raw_winsorized.csv",
        "normalized": preprocess_dir / "normalized_winsorized.csv",
        "derived": preprocess_dir / "derived_winsorized.csv",
    }
    summaries = {name: analyze_view(name, path, output_root) for name, path in views.items()}

    missingness = preprocess_summary.get("missingness", {})
    normalized_columns = summaries["normalized"].get("ordered_columns", [])
    normalized_path = output_root / "normalized" / "top_correlations.csv"
    corr_pairs = load_csv_rows(normalized_path) if normalized_path.exists() else []
    corr_lookup = {
        tuple(sorted((row["indicator_a"], row["indicator_b"]))): safe_float(row["correlation"])
        for row in corr_pairs
    }
    groups = summaries["normalized"].get("groups", {}).get("0.80", [])
    if not groups:
        groups = summaries["raw"].get("groups", {}).get("0.80", [])
    recommendations = recommend_representatives(groups, missingness, corr_lookup)
    write_csv_rows(output_root / "recommendations.csv", recommendations)

    for base_name in ("global_heatmap_grid", "stratified_heatmap_grid"):
        for suffix in (".png", ".pdf"):
            path = (output_root / base_name).with_suffix(suffix)
            if path.exists():
                path.unlink()

    summary = {
        "views": summaries,
        "recommendations_path": str(output_root / "recommendations.csv"),
    }
    write_json(output_root / "analysis_summary.json", summary)
    return summary
