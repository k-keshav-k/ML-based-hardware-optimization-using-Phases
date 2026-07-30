"""Model runners for next-phase prediction baselines and lightweight sequence models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .transformer_model import build_family_transformer, require_torch
from .tree import DecisionTree, TreeNode


@dataclass
class ModelRun:
    predictions: np.ndarray
    training_seconds: float
    inference_seconds: float
    parameter_count: int = 0
    table_entries: int = 0
    details: dict[str, object] = field(default_factory=dict)
    warning: str = ""


def _node_counts(node: TreeNode | None, depth: int = 0) -> tuple[int, int, int]:
    if node is None:
        return 0, 0, depth
    if node.feature < 0 or node.left is None or node.right is None:
        return 0, 1, depth
    left_internal, left_leaves, left_depth = _node_counts(node.left, depth + 1)
    right_internal, right_leaves, right_depth = _node_counts(node.right, depth + 1)
    return 1 + left_internal + right_internal, left_leaves + right_leaves, max(left_depth, right_depth)


def tree_storage_bytes(internal_nodes: int, leaves: int, feature_count: int) -> float:
    if feature_count <= 0:
        return 0.0
    total_nodes = internal_nodes + leaves
    feature_bits = max(1, math.ceil(math.log2(feature_count)))
    child_bits = 2 * max(1, math.ceil(math.log2(max(1, total_nodes))))
    threshold_bits = 16
    phase_bits = 2
    bits = internal_nodes * (feature_bits + threshold_bits + child_bits) + leaves * phase_bits
    return bits / 8.0


def _one_hot(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    output = np.zeros((y.shape[0], num_classes), dtype=float)
    valid = (y >= 0) & (y < num_classes)
    output[np.where(valid)[0], y[valid].astype(int)] = 1.0
    return output


def run_decision_tree(
    hist_features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    max_depth: int,
    min_samples_leaf: int,
) -> ModelRun:
    fit_start = time.perf_counter()
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(hist_features[train_mask], target[train_mask])
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = tree.predict(hist_features)
    inference_seconds = time.perf_counter() - infer_start

    internal_nodes, leaves, actual_depth = _node_counts(tree.root)
    return ModelRun(
        predictions=pred.astype(int),
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        table_entries=internal_nodes + leaves,
        details={
            "tree_internal_nodes": internal_nodes,
            "tree_leaves": leaves,
            "tree_depth": actual_depth,
            "tree_max_depth": max_depth,
            "feature_count": hist_features.shape[1],
            "estimated_storage_bytes": tree_storage_bytes(internal_nodes, leaves, hist_features.shape[1]),
        },
    )


def run_last_state_baseline(current_phase: np.ndarray) -> ModelRun:
    infer_start = time.perf_counter()
    pred = current_phase.astype(int).copy()
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(predictions=pred, training_seconds=0.0, inference_seconds=inference_seconds, table_entries=1)


def run_majority_baseline(target: np.ndarray, train_mask: np.ndarray) -> ModelRun:
    fit_start = time.perf_counter()
    train_target = target[train_mask & (target >= 0)]
    if train_target.size == 0:
        train_target = target[target >= 0]
    majority = int(np.bincount(train_target, minlength=3).argmax()) if train_target.size else 0
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = np.full(target.shape[0], majority, dtype=int)
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(predictions=pred, training_seconds=training_seconds, inference_seconds=inference_seconds, table_entries=3)


def run_state_conditioned_majority(current_phase: np.ndarray, target: np.ndarray, train_mask: np.ndarray) -> ModelRun:
    fit_start = time.perf_counter()
    mapping: dict[int, int] = {}
    fallback = int(np.bincount(target[train_mask & (target >= 0)], minlength=3).argmax()) if np.any(train_mask & (target >= 0)) else 0
    for state in range(3):
        mask = train_mask & (current_phase == state) & (target >= 0)
        if np.any(mask):
            mapping[state] = int(np.bincount(target[mask], minlength=3).argmax())
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = np.asarray([mapping.get(int(state), fallback) for state in current_phase.astype(int)], dtype=int)
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        table_entries=max(1, len(mapping)) * 3,
    )


def run_markov(current_phase: np.ndarray, target: np.ndarray, train_mask: np.ndarray) -> ModelRun:
    fit_start = time.perf_counter()
    counts = np.ones((3, 3), dtype=float)
    for state in range(3):
        mask = train_mask & (current_phase == state) & (target >= 0)
        if np.any(mask):
            bincount = np.bincount(target[mask], minlength=3).astype(float)
            counts[state, :] += bincount
    table = counts / counts.sum(axis=1, keepdims=True)
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = np.asarray([int(np.argmax(table[int(state)])) for state in current_phase.astype(int)], dtype=int)
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        table_entries=9,
        details={"transition_table": table.tolist()},
    )


def _bucketize_run_lengths(run_lengths: np.ndarray, bucket_edges: list[int]) -> np.ndarray:
    buckets = np.zeros(run_lengths.shape[0], dtype=int)
    for index, value in enumerate(run_lengths.astype(int)):
        bucket = 0
        while bucket < len(bucket_edges) and value > int(bucket_edges[bucket]):
            bucket += 1
        buckets[index] = bucket
    return buckets


def run_rle_markov(
    current_phase: np.ndarray,
    current_run_length: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    bucket_edges: list[int],
) -> ModelRun:
    bucket_ids = _bucketize_run_lengths(current_run_length, bucket_edges)
    bucket_count = len(bucket_edges) + 1
    fit_start = time.perf_counter()
    counts = np.ones((3, bucket_count, 3), dtype=float)
    for state in range(3):
        for bucket in range(bucket_count):
            mask = train_mask & (current_phase == state) & (bucket_ids == bucket) & (target >= 0)
            if np.any(mask):
                counts[state, bucket, :] += np.bincount(target[mask], minlength=3).astype(float)
    table = counts / counts.sum(axis=2, keepdims=True)
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = np.asarray([int(np.argmax(table[int(state), int(bucket_ids[idx])])) for idx, state in enumerate(current_phase.astype(int))], dtype=int)
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        table_entries=int(counts.size),
        details={"bucket_count": bucket_count},
    )


def run_hsmm_approx(
    current_phase: np.ndarray,
    current_run_length: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    bucket_edges: list[int],
) -> ModelRun:
    bucket_ids = _bucketize_run_lengths(current_run_length, bucket_edges)
    bucket_count = len(bucket_edges) + 1
    fit_start = time.perf_counter()
    stay_counts = np.ones((3, bucket_count, 2), dtype=float)
    transition_counts = np.ones((3, 3), dtype=float)
    for index in np.where(train_mask & (target >= 0))[0]:
        state = int(current_phase[index])
        bucket = int(bucket_ids[index])
        next_state = int(target[index])
        stayed = 1 if next_state == state else 0
        stay_counts[state, bucket, stayed] += 1.0
        if next_state != state:
            transition_counts[state, next_state] += 1.0
    stay_probs = stay_counts / stay_counts.sum(axis=2, keepdims=True)
    transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = np.full(target.shape[0], 0, dtype=int)
    for index, state in enumerate(current_phase.astype(int)):
        bucket = int(bucket_ids[index])
        continue_prob = float(stay_probs[state, bucket, 1])
        if continue_prob >= 0.5:
            pred[index] = state
        else:
            next_probs = transition_probs[state].copy()
            next_probs[state] = 0.0
            pred[index] = int(np.argmax(next_probs))
    inference_seconds = time.perf_counter() - infer_start
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        table_entries=int(stay_counts.size + transition_counts.size),
        details={"bucket_count": bucket_count},
    )


def _fit_linear_head(features: np.ndarray, target: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    x = np.concatenate([features, np.ones((features.shape[0], 1), dtype=float)], axis=1)
    y = _one_hot(target, num_classes=3)
    gram = x.T @ x + reg * np.eye(x.shape[1], dtype=float)
    rhs = x.T @ y
    return np.linalg.solve(gram, rhs)


def _predict_linear_head(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.concatenate([features, np.ones((features.shape[0], 1), dtype=float)], axis=1)
    logits = x @ weights
    return np.argmax(logits, axis=1).astype(int)


def _random_kernels(feature_dim: int, history_length: int, num_kernels: int, seed: int) -> list[dict[str, np.ndarray | int | float]]:
    rng = np.random.default_rng(seed)
    kernels: list[dict[str, np.ndarray | int | float]] = []
    lengths = [3, 5, 7]
    for _ in range(num_kernels):
        kernel_len = min(history_length, int(rng.choice(lengths)))
        dilation = int(rng.integers(1, max(2, history_length // max(1, kernel_len))))
        effective_len = 1 + (kernel_len - 1) * dilation
        if effective_len > history_length:
            dilation = 1
            effective_len = kernel_len
        kernels.append(
            {
                "weights": rng.normal(0.0, 1.0, size=(kernel_len, feature_dim)).astype(float),
                "bias": float(rng.normal(0.0, 1.0)),
                "dilation": dilation,
            }
        )
    return kernels


def _rocket_features(sequences: np.ndarray, kernels: list[dict[str, np.ndarray | int | float]]) -> np.ndarray:
    output = np.zeros((sequences.shape[0], len(kernels) * 2), dtype=float)
    for sample_index, sequence in enumerate(sequences):
        for kernel_index, kernel in enumerate(kernels):
            weights = np.asarray(kernel["weights"], dtype=float)
            bias = float(kernel["bias"])
            dilation = int(kernel["dilation"])
            kernel_len = weights.shape[0]
            effective_len = 1 + (kernel_len - 1) * dilation
            activation = []
            for start in range(sequence.shape[0] - effective_len + 1):
                window = sequence[start : start + effective_len : dilation]
                activation.append(float(np.sum(window * weights) + bias))
            if activation:
                act = np.asarray(activation, dtype=float)
                output[sample_index, kernel_index * 2] = float(np.max(act))
                output[sample_index, kernel_index * 2 + 1] = float(np.mean(act > 0.0))
    return output


def run_rocket_fallback(
    sequences: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    num_kernels: int,
    seed: int = 17,
) -> ModelRun:
    fit_start = time.perf_counter()
    kernels = _random_kernels(sequences.shape[2], sequences.shape[1], num_kernels, seed)
    features = _rocket_features(sequences, kernels)
    weights = _fit_linear_head(features[train_mask], target[train_mask])
    training_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    pred = _predict_linear_head(features, weights)
    inference_seconds = time.perf_counter() - infer_start
    parameter_count = int(sum(np.asarray(kernel["weights"]).size + 1 for kernel in kernels) + weights.size)
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        details={"kernel_count": num_kernels, "feature_count": features.shape[1]},
        warning="sktime not available; used internal ROCKET-style fallback",
    )


def _torch_train_classifier(
    model,
    sequences: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, float, int]:
    torch, _nn = require_torch()
    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(target.astype(int), dtype=torch.long)
    indices = np.where(train_mask)[0]
    if indices.size == 0:
        indices = np.arange(sequences.shape[0], dtype=int)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    fit_start = time.perf_counter()
    model.train()
    for _ in range(max(1, epochs)):
        np.random.shuffle(indices)
        for start in range(0, indices.size, max(1, batch_size)):
            batch_idx = indices[start : start + max(1, batch_size)]
            optimizer.zero_grad()
            logits = model(x[batch_idx])
            if logits.ndim == 3:
                logits = logits[:, 0, :]
            loss = loss_fn(logits, y[batch_idx])
            loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - fit_start
    model.eval()
    with torch.no_grad():
        infer_start = time.perf_counter()
        logits = model(x)
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        inference_seconds = time.perf_counter() - infer_start
    parameter_count = int(sum(param.numel() for param in model.parameters()))
    return pred, training_seconds, inference_seconds, parameter_count


def run_tcn(
    sequences: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    channels: int = 16,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> ModelRun:
    try:
        torch, nn = require_torch()
    except SystemExit as exc:
        return ModelRun(
            predictions=np.full(target.shape[0], -1, dtype=int),
            training_seconds=0.0,
            inference_seconds=0.0,
            details={"skipped": True},
            warning=str(exc),
        )

    class CausalConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
            self.relu = nn.ReLU()
            self.dilation = dilation

        def forward(self, x):
            pad = (self.conv.kernel_size[0] - 1) * self.dilation
            x_pad = torch.nn.functional.pad(x, (pad, 0))
            out = self.conv(x_pad)
            return self.relu(out + self.proj(x))

    class TinyTCN(nn.Module):
        def __init__(self, input_dim: int, hidden: int) -> None:
            super().__init__()
            self.block1 = CausalConvBlock(input_dim, hidden, 1)
            self.block2 = CausalConvBlock(hidden, hidden, 2)
            self.head = nn.Linear(hidden, 3)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.block1(x)
            x = self.block2(x)
            return self.head(x[:, :, -1])

    model = TinyTCN(sequences.shape[2], channels)
    pred, training_seconds, inference_seconds, parameter_count = _torch_train_classifier(
        model,
        sequences,
        target,
        train_mask,
        epochs,
        batch_size,
        learning_rate,
    )
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        details={"feature_count": sequences.shape[2]},
    )


def run_transformer(
    sequences: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    hidden_dim: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    ff_dim: int = 64,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> ModelRun:
    try:
        require_torch()
    except SystemExit as exc:
        return ModelRun(
            predictions=np.full(target.shape[0], -1, dtype=int),
            training_seconds=0.0,
            inference_seconds=0.0,
            details={"skipped": True},
            warning=str(exc),
        )
    model = build_family_transformer(
        input_dim=sequences.shape[2],
        horizon=1,
        num_classes=3,
        config={
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout": 0.0,
            "rope_theta": 10000.0,
        },
    )
    pred, training_seconds, inference_seconds, parameter_count = _torch_train_classifier(
        model,
        sequences,
        target,
        train_mask,
        epochs,
        batch_size,
        learning_rate,
    )
    return ModelRun(
        predictions=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        details={"feature_count": sequences.shape[2]},
    )
