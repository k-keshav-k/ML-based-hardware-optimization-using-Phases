"""Classical next-phase prediction baselines."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, write_csv_rows, write_json

from .features import fill_nan_with_medians
from .metrics import classification_metrics


def load_window_dataset(dataset_dir: Path, label_dir: Path) -> tuple[np.ndarray, list[dict[str, str]], dict[str, object]]:
    arrays = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    x = arrays["X"].astype(float)
    labels = load_csv_rows(label_dir / "window_labels.csv")
    model = read_json(label_dir / "label_model.json")
    if not isinstance(model, dict):
        raise ValueError("label_model.json must contain an object")
    return x, labels, model


def aligned_arrays(x: np.ndarray, rows: list[dict[str, str]], medians: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # window_id ties labels to tensor rows; this keeps downstream split logic
    # consistent with window_labels.csv.
    used_indices = np.asarray([int(row["window_id"]) for row in rows], dtype=int)
    x_used = x[used_indices]
    flat = fill_nan_with_medians(x_used.reshape(x_used.shape[0], -1), np.tile(medians, x_used.shape[1]))
    y = np.asarray([int(row["next_phase_id"]) for row in rows], dtype=int)
    current = np.asarray([int(row["phase_id"]) for row in rows], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in rows], dtype=int)
    run_length = np.asarray([int(row.get("current_run_length", "1") or 1) for row in rows], dtype=int)
    return flat, y, current, change, run_length


@dataclass
class NearestCentroid:
    centroids: dict[int, np.ndarray]
    fallback: int

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        centroids = {int(label): x[y == label].mean(axis=0) for label in sorted(set(y.tolist()))}
        fallback = Counter(y.tolist()).most_common(1)[0][0]
        return cls(centroids=centroids, fallback=int(fallback))

    def predict(self, x: np.ndarray) -> np.ndarray:
        labels = sorted(self.centroids)
        if not labels:
            return np.full(x.shape[0], self.fallback, dtype=int)
        centers = np.vstack([self.centroids[label] for label in labels])
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.asarray([labels[index] for index in distances.argmin(axis=1)], dtype=int)


@dataclass
class TreeNode:
    prediction: int
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict_one(self, row: np.ndarray) -> int:
        if self.feature < 0 or self.left is None or self.right is None:
            return self.prediction
        if row[self.feature] <= self.threshold:
            return self.left.predict_one(row)
        return self.right.predict_one(row)


class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 3) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root: TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._build(x, y, 0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree has not been fitted")
        return np.asarray([self.root.predict_one(row) for row in x], dtype=int)

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        prediction = Counter(y.tolist()).most_common(1)[0][0]
        node = TreeNode(prediction=int(prediction))
        if depth >= self.max_depth or len(set(y.tolist())) <= 1 or x.shape[0] < 2 * self.min_samples_leaf:
            return node
        best = self._best_split(x, y)
        if best is None:
            return node
        feature, threshold = best
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build(x[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        best_score = float("inf")
        best: tuple[int, float] | None = None
        for feature in range(x.shape[1]):
            values = np.unique(np.quantile(x[:, feature], np.linspace(0.1, 0.9, 9)))
            for threshold in values:
                left = x[:, feature] <= threshold
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                score = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / y.size
                if score < best_score:
                    best_score = score
                    best = (feature, float(threshold))
        return best


def gini(y: np.ndarray) -> float:
    counts = np.asarray(list(Counter(y.tolist()).values()), dtype=float)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs**2))


class SoftmaxRegression:
    def __init__(self, epochs: int = 180, lr: float = 0.08, l2: float = 0.0005, hinge: bool = False) -> None:
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.hinge = hinge
        self.classes: np.ndarray = np.empty(0, dtype=int)
        self.weights: np.ndarray = np.empty((0, 0), dtype=float)
        self.mean: np.ndarray = np.empty(0, dtype=float)
        self.scale: np.ndarray = np.empty(0, dtype=float)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        self.classes = np.asarray(sorted(set(y.tolist())), dtype=int)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        xs = (x - self.mean) / self.scale
        xs = np.c_[np.ones(xs.shape[0]), xs]
        self.weights = np.zeros((xs.shape[1], self.classes.size), dtype=float)
        y_index = np.asarray([int(np.where(self.classes == label)[0][0]) for label in y], dtype=int)
        for _ in range(self.epochs):
            scores = xs @ self.weights
            if self.hinge:
                margins = scores - scores[np.arange(xs.shape[0]), y_index][:, None] + 1.0
                margins[np.arange(xs.shape[0]), y_index] = 0.0
                active = margins > 0.0
                active_counts = active.sum(axis=1)
                indicator = active.astype(float)
                indicator[np.arange(xs.shape[0]), y_index] = -active_counts
                grad = xs.T @ indicator / xs.shape[0] + self.l2 * self.weights
            else:
                scores -= scores.max(axis=1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=1, keepdims=True)
                probs[np.arange(xs.shape[0]), y_index] -= 1.0
                grad = xs.T @ probs / xs.shape[0] + self.l2 * self.weights
            self.weights -= self.lr * grad
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.mean) / self.scale
        xs = np.c_[np.ones(xs.shape[0]), xs]
        return self.classes[(xs @ self.weights).argmax(axis=1)]


class RLEMarkov:
    def __init__(self) -> None:
        self.table: dict[tuple[int, int], int] = {}
        self.phase_backoff: dict[int, int] = {}
        self.fallback = 0

    @staticmethod
    def bucket(length: int) -> int:
        if length <= 1:
            return 1
        if length <= 3:
            return 3
        if length <= 7:
            return 7
        return 15

    def fit(self, current: np.ndarray, run_length: np.ndarray, y: np.ndarray) -> "RLEMarkov":
        buckets: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        by_phase: dict[int, Counter[int]] = defaultdict(Counter)
        for phase, length, target in zip(current, run_length, y):
            buckets[(int(phase), self.bucket(int(length)))][int(target)] += 1
            by_phase[int(phase)][int(target)] += 1
        self.table = {key: counter.most_common(1)[0][0] for key, counter in buckets.items()}
        self.phase_backoff = {key: counter.most_common(1)[0][0] for key, counter in by_phase.items()}
        self.fallback = Counter(y.tolist()).most_common(1)[0][0]
        return self

    def predict(self, current: np.ndarray, run_length: np.ndarray) -> np.ndarray:
        preds = []
        for phase, length in zip(current, run_length):
            phase = int(phase)
            key = (phase, self.bucket(int(length)))
            preds.append(self.table.get(key, self.phase_backoff.get(phase, self.fallback)))
        return np.asarray(preds, dtype=int)


def train_baselines(dataset_dir: Path, label_dir: Path, output_dir: Path, config: dict[str, object]) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    x, rows, label_model = load_window_dataset(dataset_dir, label_dir)
    medians = np.asarray(label_model["model"]["standardizer"]["medians"], dtype=float)
    flat, y, current, change, run_length = aligned_arrays(x, rows, medians)
    split = np.asarray([row.get("split", "train") for row in rows])
    train_mask = split == "train"
    eval_mask = split != "train"
    if not np.any(eval_mask):
        eval_mask = np.ones(split.shape[0], dtype=bool)
    results: dict[str, object] = {}
    models = {
        # Predict "no transition" by copying the current phase id.
        "last_value": None,
        "nearest_centroid": NearestCentroid.fit(flat[train_mask], y[train_mask]),
        "decision_tree": DecisionTree(
            int(config.get("decision_tree_max_depth", 5)),
            int(config.get("decision_tree_min_samples_leaf", 3)),
        ).fit(flat[train_mask], y[train_mask]),
        "logistic_regression": SoftmaxRegression(
            int(config.get("logistic_epochs", 180)),
            float(config.get("logistic_lr", 0.08)),
            float(config.get("logistic_l2", 0.0005)),
        ).fit(flat[train_mask], y[train_mask]),
        "rle_markov": RLEMarkov().fit(current[train_mask], run_length[train_mask], y[train_mask]),
    }
    if bool(config.get("include_linear_svm", False)):
        models["linear_svm"] = SoftmaxRegression(
            int(config.get("logistic_epochs", 180)),
            float(config.get("logistic_lr", 0.08)),
            float(config.get("logistic_l2", 0.0005)),
            hinge=True,
        ).fit(flat[train_mask], y[train_mask])
    for name, model in models.items():
        start = time.perf_counter()
        if name == "last_value":
            pred = current.copy()
        elif name == "rle_markov":
            pred = model.predict(current, run_length)  # type: ignore[union-attr]
        else:
            pred = model.predict(flat)  # type: ignore[union-attr]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        pred_change = (pred != current).astype(int)
        metrics = classification_metrics(y[eval_mask], pred[eval_mask], change[eval_mask], pred_change[eval_mask])
        metrics["inference_ms_total"] = elapsed_ms
        metrics["inference_us_per_sample"] = elapsed_ms * 1000.0 / max(1, pred.size)
        results[name] = metrics
        prediction_rows = []
        for row, true_phase, predicted, true_change, predicted_change in zip(rows, y, pred, change, pred_change):
            prediction_rows.append(
                {
                    "model": name,
                    "window_id": row.get("window_id", ""),
                    "split": row.get("split", ""),
                    "workload": row.get("workload", ""),
                    "run_id": row.get("run_id", ""),
                    "phase_id": row.get("phase_id", ""),
                    "y_true_next_phase": int(true_phase),
                    "y_pred_next_phase": int(predicted),
                    "y_true_phase_change": int(true_change),
                    "y_pred_phase_change": int(predicted_change),
                }
            )
        write_csv_rows(output_dir / f"{name}_predictions.csv", prediction_rows)
    write_json(output_dir / "baseline_summary.json", results)
    return results
