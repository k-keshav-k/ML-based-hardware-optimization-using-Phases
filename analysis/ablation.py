"""Detector ablations for feature groups, metrics, quantization, and persistence."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from phase_ml.metrics import classification_metrics

from .common import FEATURE_GROUPS, load_rows, standardize, write_json, write_rows
from .detector import CentroidDetector, apply_persistence, quantize_fixed_point


def window_features(dataset_dir: Path, feature_subset: list[str]) -> tuple[np.ndarray, list[str]]:
    arrays = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    x = np.asarray(arrays["X"], dtype=float)
    columns = [str(item) for item in arrays["feature_columns"]]
    indices = [columns.index(column) for column in feature_subset if column in columns]
    if not indices:
        raise ValueError(f"feature subset not present: {feature_subset}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        features = np.nanmean(x[:, :, indices], axis=1)
    for col in range(features.shape[1]):
        values = features[:, col]
        clean = values[np.isfinite(values)]
        fill = float(np.median(clean)) if clean.size else 0.0
        values[~np.isfinite(values)] = fill
        features[:, col] = values
    return standardize(features), [columns[index] for index in indices]


def correlation_pruned_features(dataset_dir: Path, threshold: float = 0.90) -> list[str]:
    arrays = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    columns = [str(item) for item in arrays["feature_columns"]]
    candidates = [column for column in FEATURE_GROUPS["all"] if column in columns]
    x, used = window_features(dataset_dir, candidates)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    kept: list[int] = []
    for index in range(len(used)):
        if all(abs(float(corr[index, prior])) < threshold for prior in kept):
            kept.append(index)
    return [used[index] for index in kept]


def evaluate_detector(
    dataset_dir: Path,
    label_dir: Path,
    feature_group: str,
    metric: str,
    persistence: int,
    bits: int | None,
    max_centroids: int,
    feature_subset: list[str] | None = None,
) -> dict[str, object]:
    rows = load_rows(label_dir / "window_labels.csv")
    x, used_columns = window_features(dataset_dir, feature_subset or FEATURE_GROUPS[feature_group])
    if bits is not None:
        x = quantize_fixed_point(x, bits)
    y = np.asarray([int(row["next_phase_id"]) for row in rows], dtype=int)
    current = np.asarray([int(row["phase_id"]) for row in rows], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in rows], dtype=int)
    split = np.asarray([row["split"] for row in rows])
    train = split == "train"
    eval_mask = split != "train"
    detector = CentroidDetector.fit(x[train], y[train], metric=metric, max_centroids=max_centroids)
    pred = detector.predict(x)
    pred = apply_persistence(pred, persistence)
    pred_change = (pred != current).astype(int)
    metrics = classification_metrics(y[eval_mask], pred[eval_mask], change[eval_mask], pred_change[eval_mask])
    return {
        "feature_group": feature_group,
        "metric": metric,
        "persistence": persistence,
        "fixed_point_bits": bits if bits is not None else "float",
        "max_centroids": max_centroids,
        "feature_count": len(used_columns),
        **metrics,
    }


def run_ablations(dataset_dir: Path, label_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in ["branch", "l1", "llc_offcore", "shared", "local", "all"]:
        rows.append(evaluate_detector(dataset_dir, label_dir, group, "manhattan", 1, None, 16))
    rows.append(evaluate_detector(dataset_dir, label_dir, "correlation_pruned", "manhattan", 1, None, 16, correlation_pruned_features(dataset_dir)))
    for metric in ["manhattan", "euclidean", "cosine"]:
        rows.append(evaluate_detector(dataset_dir, label_dir, "all", metric, 1, None, 16))
    for persistence in [1, 2, 3, 5]:
        rows.append(evaluate_detector(dataset_dir, label_dir, "all", "manhattan", persistence, None, 16))
    for bits in [8, 10, 12, 16]:
        rows.append(evaluate_detector(dataset_dir, label_dir, "all", "manhattan", 1, bits, 16))
    for k in [4, 8, 12, 16]:
        rows.append(evaluate_detector(dataset_dir, label_dir, "all", "manhattan", 1, None, k))
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_group",
        "metric",
        "persistence",
        "fixed_point_bits",
        "max_centroids",
        "feature_count",
        "accuracy",
        "macro_f1",
        "phase_change_precision",
        "phase_change_recall",
        "phase_change_f1",
    ]
    write_rows(output_dir / "detector_ablation.csv", rows, fieldnames)
    summary = {
        "analysis_type": "offline_detector_ablation",
        "best_accuracy": max(rows, key=lambda row: float(row["accuracy"])),
        "best_phase_change_f1": max(rows, key=lambda row: float(row["phase_change_f1"])),
        "rows": rows,
    }
    write_json(output_dir / "detector_ablation_summary.json", summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = run_ablations(Path(args.dataset_dir), Path(args.label_dir), Path(args.output_dir))
    print(f"Wrote {len(rows)} detector ablation rows.")


if __name__ == "__main__":
    main()
