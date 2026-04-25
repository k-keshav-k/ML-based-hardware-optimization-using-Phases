"""Analyze cross-frequency feature stability for timing-independent signatures."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .common import FEATURE_GROUPS, load_rows, safe_float, standardize, write_json, write_rows


def js_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values()) or 1
    right_total = sum(right.values()) or 1
    p = np.asarray([left[key] / left_total for key in keys], dtype=float)
    q = np.asarray([right[key] / right_total for key in keys], dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return math.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m))


def artifact_rows(name: str, artifact_dir: Path) -> tuple[list[dict[str, object]], dict[str, Counter[str]]]:
    interval_path = artifact_dir / "dataset" / "interval_features.csv"
    label_path = artifact_dir / "labels" / "window_labels.csv"
    rows = load_rows(interval_path)
    medians: dict[str, list[list[float]]] = defaultdict(list)
    features = [feature for feature in FEATURE_GROUPS["all"] if feature in rows[0]]
    for row in rows:
        medians[row["workload"]].append([safe_float(row.get(feature, "")) for feature in features])
    output = []
    for workload, vectors in medians.items():
        matrix = np.asarray(vectors, dtype=float)
        values = []
        for col in range(matrix.shape[1]):
            clean = matrix[:, col][np.isfinite(matrix[:, col])]
            values.append(float(np.median(clean)) if clean.size else 0.0)
        output.append({"artifact": name, "workload": workload, "features": features, "vector": values})
    phase_dist: dict[str, Counter[str]] = defaultdict(Counter)
    if label_path.exists():
        for row in load_rows(label_path):
            phase_dist[row["workload"]][row["phase_id"]] += 1
    return output, phase_dist


def analyze(artifacts: list[tuple[str, Path]], output_dir: Path) -> dict[str, object]:
    vectors = []
    phase_dists: dict[tuple[str, str], Counter[str]] = {}
    for name, path in artifacts:
        items, dists = artifact_rows(name, path)
        vectors.extend(items)
        for workload, counter in dists.items():
            phase_dists[(name, workload)] = counter
    matrix = standardize(np.asarray([item["vector"] for item in vectors], dtype=float))
    for item, vector in zip(vectors, matrix):
        item["standardized"] = vector
    drift_rows = []
    by_workload: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in vectors:
        by_workload[str(item["workload"])].append(item)
    for workload, items in by_workload.items():
        for left, right in combinations(items, 2):
            drift = float(np.linalg.norm(np.asarray(left["standardized"]) - np.asarray(right["standardized"])))
            dist = js_distance(phase_dists.get((str(left["artifact"]), workload), Counter()), phase_dists.get((str(right["artifact"]), workload), Counter()))
            drift_rows.append(
                {
                    "workload": workload,
                    "left_artifact": left["artifact"],
                    "right_artifact": right["artifact"],
                    "centroid_drift_l2": drift,
                    "phase_distribution_js_distance": dist,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "dvfs_stress.csv", drift_rows)
    summary = {
        "analysis_type": "cross_frequency_timing_independent_stability",
        "artifact_count": len(artifacts),
        "comparison_count": len(drift_rows),
        "mean_centroid_drift_l2": sum(float(row["centroid_drift_l2"]) for row in drift_rows) / len(drift_rows) if drift_rows else 0.0,
        "mean_phase_distribution_js_distance": sum(float(row["phase_distribution_js_distance"]) for row in drift_rows) / len(drift_rows) if drift_rows else 0.0,
    }
    write_json(output_dir / "dvfs_stress_summary.json", summary)
    return summary


def parse_artifact(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split(":", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, help="name:/path/to/phase_ml_artifact")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = analyze([parse_artifact(item) for item in args.artifact], Path(args.output_dir))
    print(f"Analyzed {summary['comparison_count']} cross-frequency comparisons.")


if __name__ == "__main__":
    main()

