"""Small detector primitives used by the paper artifact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ewma_update(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * current + (1.0 - alpha) * previous


def quantize_fixed_point(values: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 1:
        raise ValueError("bits must be greater than one")
    clipped = np.clip(values, -4.0, 4.0)
    levels = (1 << bits) - 1
    scaled = np.round((clipped + 4.0) / 8.0 * levels)
    return scaled / levels * 8.0 - 4.0


def distance_matrix(x: np.ndarray, centroids: np.ndarray, metric: str) -> np.ndarray:
    if metric == "manhattan":
        return np.abs(x[:, None, :] - centroids[None, :, :]).sum(axis=2)
    if metric == "euclidean":
        return np.sqrt(((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    if metric == "cosine":
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        c_norm = np.linalg.norm(centroids, axis=1, keepdims=True).T
        denom = np.maximum(x_norm * c_norm, 1e-12)
        return 1.0 - (x @ centroids.T) / denom
    raise ValueError(f"unknown metric: {metric}")


@dataclass
class CentroidDetector:
    centroids: np.ndarray
    labels: np.ndarray
    metric: str = "manhattan"

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, metric: str = "manhattan", max_centroids: int = 16) -> "CentroidDetector":
        labels = np.asarray(sorted(set(int(item) for item in y.tolist())), dtype=int)
        if labels.size > max_centroids:
            counts = [(int(label), int(np.sum(y == label))) for label in labels]
            labels = np.asarray([label for label, _ in sorted(counts, key=lambda item: item[1], reverse=True)[:max_centroids]], dtype=int)
        centroids = np.vstack([x[y == label].mean(axis=0) for label in labels])
        return cls(centroids=centroids, labels=labels, metric=metric)

    def predict(self, x: np.ndarray) -> np.ndarray:
        distances = distance_matrix(x, self.centroids, self.metric)
        return self.labels[distances.argmin(axis=1)]


def apply_persistence(labels: np.ndarray, threshold: int) -> np.ndarray:
    if threshold <= 1 or labels.size == 0:
        return labels.copy()
    output = labels.copy()
    current = int(labels[0])
    candidate = current
    count = 0
    for index, label in enumerate(labels):
        label = int(label)
        if label == current:
            candidate = label
            count = 0
            output[index] = current
            continue
        if label == candidate:
            count += 1
        else:
            candidate = label
            count = 1
        if count >= threshold:
            current = candidate
        output[index] = current
    return output
