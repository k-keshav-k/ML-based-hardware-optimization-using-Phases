"""Small numpy clustering/preprocessing utilities for phase labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardizer":
        medians = np.zeros(matrix.shape[1], dtype=float)
        filled = matrix.copy()
        for index in range(matrix.shape[1]):
            values = filled[:, index]
            clean = values[np.isfinite(values)]
            medians[index] = float(np.median(clean)) if clean.size else 0.0
            values[~np.isfinite(values)] = medians[index]
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[scales < 1e-12] = 1.0
        return cls(medians=medians, means=means, scales=scales)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        filled = matrix.copy()
        for index in range(filled.shape[1]):
            filled[~np.isfinite(filled[:, index]), index] = self.medians[index]
        return (filled - self.means) / self.scales


@dataclass
class PCATransformer:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, variance: float) -> "PCATransformer":
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        if matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
            return cls(mean=mean, components=np.eye(matrix.shape[1]), explained_variance_ratio=np.ones(matrix.shape[1]))
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        eigen_values = (singular_values**2) / max(1, matrix.shape[0] - 1)
        total = float(eigen_values.sum())
        ratios = eigen_values / total if total > 0 else np.ones_like(eigen_values) / len(eigen_values)
        keep = int(np.searchsorted(np.cumsum(ratios), variance) + 1)
        keep = max(1, min(keep, vt.shape[0]))
        return cls(mean=mean, components=vt[:keep], explained_variance_ratio=ratios[:keep])

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) @ self.components.T


@dataclass
class KMeansModel:
    centers: np.ndarray

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        distances = ((matrix[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(int)


def fit_kmeans(matrix: np.ndarray, k: int, seed: int, max_iter: int = 80) -> KMeansModel:
    if matrix.shape[0] == 0:
        raise ValueError("Cannot fit k-means on an empty matrix")
    k = max(1, min(k, matrix.shape[0]))
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(max_iter):
        new_labels = KMeansModel(centers).predict(matrix)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            mask = labels == cluster
            if np.any(mask):
                centers[cluster] = matrix[mask].mean(axis=0)
            else:
                centers[cluster] = matrix[rng.integers(0, matrix.shape[0])]
    return KMeansModel(centers=centers)


def choose_kmeans(matrix: np.ndarray, k_min: int, k_max: int, seed: int, max_iter: int) -> KMeansModel:
    best_score = float("inf")
    best_model: KMeansModel | None = None
    upper = max(k_min, min(k_max, matrix.shape[0]))
    for k in range(max(1, k_min), upper + 1):
        model = fit_kmeans(matrix, k, seed + k, max_iter)
        labels = model.predict(matrix)
        inertia = float(((matrix - model.centers[labels]) ** 2).sum())
        score = matrix.shape[0] * np.log(max(inertia / max(1, matrix.shape[0]), 1e-12)) + k * matrix.shape[1] * np.log(max(2, matrix.shape[0]))
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise ValueError("Unable to fit k-means")
    return best_model


@dataclass
class DiagonalGMM:
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def _log_prob(self, matrix: np.ndarray) -> np.ndarray:
        variances = np.maximum(self.variances, 1e-6)
        log_det = np.log(variances).sum(axis=1)
        diff = matrix[:, None, :] - self.means[None, :, :]
        mahal = ((diff**2) / variances[None, :, :]).sum(axis=2)
        return -0.5 * (matrix.shape[1] * np.log(2 * np.pi) + log_det[None, :] + mahal) + np.log(np.maximum(self.weights, 1e-12))[None, :]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self._log_prob(matrix).argmax(axis=1).astype(int)


def fit_gmm(matrix: np.ndarray, k: int, seed: int, max_iter: int = 80) -> tuple[DiagonalGMM, float]:
    base = fit_kmeans(matrix, k, seed, max_iter=30)
    labels = base.predict(matrix)
    means = base.centers.copy()
    variances = np.tile(np.var(matrix, axis=0) + 1e-4, (means.shape[0], 1))
    weights = np.ones(means.shape[0]) / means.shape[0]
    previous_ll = -float("inf")
    for _ in range(max_iter):
        model = DiagonalGMM(weights=weights, means=means, variances=variances)
        log_prob = model._log_prob(matrix)
        max_log = log_prob.max(axis=1, keepdims=True)
        resp = np.exp(log_prob - max_log)
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-12)
        nk = resp.sum(axis=0) + 1e-12
        weights = nk / matrix.shape[0]
        means = (resp.T @ matrix) / nk[:, None]
        for cluster in range(means.shape[0]):
            diff = matrix - means[cluster]
            variances[cluster] = (resp[:, cluster][:, None] * diff**2).sum(axis=0) / nk[cluster] + 1e-6
        ll = float(np.sum(max_log[:, 0] + np.log(np.maximum(np.exp(log_prob - max_log).sum(axis=1), 1e-12))))
        if abs(ll - previous_ll) < 1e-4:
            break
        previous_ll = ll
    model = DiagonalGMM(weights=weights, means=means, variances=variances)
    params = (k - 1) + 2 * k * matrix.shape[1]
    bic = -2.0 * previous_ll + params * np.log(max(2, matrix.shape[0]))
    return model, float(bic)


def choose_gmm(matrix: np.ndarray, k_min: int, k_max: int, seed: int, max_iter: int) -> DiagonalGMM:
    best_bic = float("inf")
    best_model: DiagonalGMM | None = None
    upper = max(k_min, min(k_max, matrix.shape[0]))
    for k in range(max(1, k_min), upper + 1):
        model, bic = fit_gmm(matrix, k, seed + k, max_iter=max_iter)
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise ValueError("Unable to fit GMM")
    return best_model


def smooth_labels(labels: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or labels.size == 0:
        return labels.copy(), np.zeros(labels.shape[0], dtype=bool)
    radius = window // 2
    output = labels.copy()
    for index in range(labels.size):
        lo = max(0, index - radius)
        hi = min(labels.size, index + radius + 1)
        values, counts = np.unique(labels[lo:hi], return_counts=True)
        output[index] = values[counts.argmax()]
    return output, output != labels


def enforce_min_duration(labels: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    if minimum <= 1 or labels.size == 0:
        return labels.copy(), np.zeros(labels.shape[0], dtype=bool)
    output = labels.copy()
    changed = np.zeros(labels.shape[0], dtype=bool)
    index = 0
    while index < labels.size:
        start = index
        value = output[index]
        while index < labels.size and output[index] == value:
            index += 1
        end = index
        if end - start >= minimum:
            continue
        replacement = output[start - 1] if start > 0 else (output[end] if end < labels.size else value)
        if replacement != value:
            output[start:end] = replacement
            changed[start:end] = True
    return output, changed
