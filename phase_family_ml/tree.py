"""Small decision tree used by ablation and distilled students."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class TreeNode:
    prediction: int
    counts: dict[int, int]
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

    def leaf_for(self, row: np.ndarray) -> "TreeNode":
        if self.feature < 0 or self.left is None or self.right is None:
            return self
        if row[self.feature] <= self.threshold:
            return self.left.leaf_for(row)
        return self.right.leaf_for(row)


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

    def predict_top_k(self, x: np.ndarray, k: int, num_classes: int = 3) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree has not been fitted")
        output = np.full((x.shape[0], k), -1, dtype=int)
        for index, row in enumerate(x):
            leaf = self.root.leaf_for(row)
            ordered = [label for label, _count in sorted(leaf.counts.items(), key=lambda item: (-item[1], item[0]))]
            for label in range(num_classes):
                if label not in leaf.counts:
                    ordered.append(label)
            output[index, : min(k, len(ordered))] = ordered[:k]
        return output

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        counts = Counter(y.tolist())
        prediction = counts.most_common(1)[0][0]
        node = TreeNode(prediction=int(prediction), counts={int(label): int(count) for label, count in counts.items()})
        if depth >= self.max_depth or len(set(y.tolist())) <= 1 or x.shape[0] < 2 * self.min_samples_leaf:
            return node
        split = self._best_split(x, y)
        if split is None:
            return node
        feature, threshold = split
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
