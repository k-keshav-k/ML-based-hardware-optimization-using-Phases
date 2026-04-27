"""Teacher training for family-wise multi-horizon transformer LMs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, write_csv_rows, write_json

from .data import load_scope_family_data, states_matrix
from .metrics import classification_metrics
from .transformer_model import build_family_transformer, require_torch


def _group_indices(metadata_rows: list[dict[str, str]]) -> list[list[int]]:
    """Rebuild temporal streams from run/core metadata."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata_rows):
        grouped[(str(row.get("run_id", "")), str(row.get("core_id", "")))].append(index)
    for key in grouped:
        grouped[key].sort(key=lambda i: float(metadata_rows[i].get("timestamp_ms", "0") or 0.0))
    return list(grouped.values())


def _encode_states(sequence: np.ndarray) -> np.ndarray:
    """Map raw state IDs into stable numeric inputs for the transformer."""

    encoded = sequence.astype(float).copy()
    encoded[encoded < 0] = -1.0
    # Values map from {-1,0,1,2} -> {0,1/3,2/3,1.0}
    return (encoded + 1.0) / 3.0


def _build_examples(
    family_index: int,
    current: np.ndarray,
    future: np.ndarray,
    split: np.ndarray,
    metadata_rows: list[dict[str, str]],
    history_length: int,
    context_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], np.ndarray]:
    """Build sequence examples for one target family and one context mode."""

    streams = _group_indices(metadata_rows)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    split_list: list[str] = []
    current_state_list: list[int] = []
    meta_list: list[dict[str, str]] = []
    row_ids: list[int] = []
    for stream in streams:
        if len(stream) < history_length:
            continue
        for local_pos in range(history_length - 1, len(stream)):
            row_index = stream[local_pos]
            target = future[row_index, family_index, :]
            if np.any(target < 0):
                continue
            history_indices = stream[local_pos - history_length + 1 : local_pos + 1]
            if context_mode == "with_context":
                history_states = current[history_indices, :]
            else:
                history_states = current[history_indices, family_index : family_index + 1]
            x_list.append(_encode_states(history_states))
            y_list.append(target.astype(int))
            split_list.append(str(split[row_index]))
            current_state_list.append(int(current[row_index, family_index]))
            meta_list.append(metadata_rows[row_index])
            row_ids.append(int(row_index))
    if not x_list:
        return (
            np.empty((0, history_length, 1), dtype=float),
            np.empty((0, future.shape[2]), dtype=int),
            np.empty(0, dtype=object),
            np.empty(0, dtype=int),
            [],
            np.empty(0, dtype=int),
        )
    return (
        np.asarray(x_list, dtype=float),
        np.asarray(y_list, dtype=int),
        np.asarray(split_list),
        np.asarray(current_state_list, dtype=int),
        meta_list,
        np.asarray(row_ids, dtype=int),
    )


def _eval_mask(split: np.ndarray) -> np.ndarray:
    val = split == "val"
    if np.any(val):
        return val
    test = split == "test"
    if np.any(test):
        return test
    return split != "train"


def _weighted_score(metrics: dict[str, float]) -> float:
    return 0.4 * float(metrics.get("accuracy", 0.0)) + 0.4 * float(metrics.get("phase_change_f1", 0.0)) + 0.2 * float(metrics.get("high_usage_recall", 0.0))


def _train_model(x: np.ndarray, y: np.ndarray, split: np.ndarray, model_cfg: dict[str, object], seed: int) -> tuple[object, np.ndarray]:
    """Train one transformer and return logits for all rows."""

    torch, nn = require_torch()
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_family_transformer(input_dim=x.shape[2], horizon=y.shape[1], num_classes=3, config=model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(model_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(model_cfg.get("epochs", 4))
    batch_size = int(model_cfg.get("batch_size", 256))
    train_mask = split == "train"
    train_index = np.where(train_mask)[0]
    if train_index.size == 0:
        train_index = np.arange(x.shape[0])
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        np.random.shuffle(train_index)
        model.train()
        for start in range(0, train_index.size, batch_size):
            idx = train_index[start : start + batch_size]
            xb = torch.tensor(x[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            # Average CE over horizons to train all forecast steps jointly.
            losses = [ce(logits[:, h, :], yb[:, h]) for h in range(y.shape[1])]
            loss = sum(losses) / max(1, len(losses))
            loss.backward()
            optimizer.step()
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            parts.append(logits.cpu().numpy())
    logits_np = np.concatenate(parts, axis=0) if parts else np.zeros((0, y.shape[1], 3), dtype=float)
    return model, logits_np


def train_teachers_for_experiment(
    experiment_dir: Path,
    scope: str,
    output_dir: Path,
    horizon: int,
    history_length: int,
    teacher_config: dict[str, object],
    seed: int,
) -> list[dict[str, object]]:
    """Train best-context teacher per family and export predictions/metrics."""

    output_dir = ensure_dir(output_dir)
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    families, current, future, split, metadata_rows = states_matrix(family_data, horizon)
    if not families:
        return []
    summaries: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        best = {
            "score": -1.0,
            "context_mode": "",
            "metrics": {},
            "pred": np.empty((0, horizon), dtype=int),
            "prob": np.empty((0, horizon, 3), dtype=float),
            "true": np.empty((0, horizon), dtype=int),
            "split": np.empty(0, dtype=object),
            "current": np.empty(0, dtype=int),
            "meta": [],
            "row_ids": np.empty(0, dtype=int),
        }
        for context_mode in teacher_config.get("context_modes", ["without_context"]):
            x, y, split_local, current_local, meta_local, row_ids = _build_examples(
                family_index,
                current,
                future,
                split,
                metadata_rows,
                history_length,
                str(context_mode),
            )
            if x.shape[0] == 0:
                continue
            _, logits = _train_model(x, y, split_local, teacher_config, seed + family_index)
            logits_shifted = logits - logits.max(axis=2, keepdims=True)
            exp = np.exp(logits_shifted)
            prob = exp / np.maximum(exp.sum(axis=2, keepdims=True), 1e-12)
            pred = logits.argmax(axis=2).astype(int)
            eval_mask = _eval_mask(split_local)
            metrics = classification_metrics(y[eval_mask, 0], pred[eval_mask, 0], current_state=current_local[eval_mask])
            score = _weighted_score(metrics)
            if score > best["score"]:
                best = {
                    "score": score,
                    "context_mode": str(context_mode),
                    "metrics": metrics,
                    "pred": pred,
                    "prob": prob,
                    "true": y,
                    "split": split_local,
                    "current": current_local,
                    "meta": meta_local,
                    "row_ids": row_ids,
                }
        if best["score"] < 0:
            summaries.append(
                {
                    "family": family,
                    "scope": scope,
                    "context_mode": "unavailable",
                    "rows": 0,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "phase_change_f1": 0.0,
                    "high_usage_recall": 0.0,
                    "validation_score": 0.0,
                }
            )
            continue

        pred = best["pred"]
        prob = best["prob"]
        true = best["true"]
        split_local = best["split"]
        current_local = best["current"]
        meta_local = best["meta"]
        row_ids = best["row_ids"]
        for idx in range(pred.shape[0]):
            item: dict[str, object] = {
                "family": family,
                "scope": scope,
                "context_mode": best["context_mode"],
                "split": split_local[idx],
                "workload": meta_local[idx].get("workload", ""),
                "run_id": meta_local[idx].get("run_id", ""),
                "core_id": meta_local[idx].get("core_id", ""),
                "row_index": int(row_ids[idx]),
                "family_state": int(current_local[idx]),
            }
            for step in range(1, horizon + 1):
                item[f"y_true_future_state_{step}"] = int(true[idx, step - 1])
                item[f"y_pred_future_state_{step}"] = int(pred[idx, step - 1])
                for cls in range(3):
                    item[f"p_future_state_{step}_class_{cls}"] = float(prob[idx, step - 1, cls])
            prediction_rows.append(item)
        summary_row = {
            "family": family,
            "scope": scope,
            "context_mode": best["context_mode"],
            "rows": int(pred.shape[0]),
            "validation_score": float(best["score"]),
        }
        summary_row.update(best["metrics"])
        summaries.append(summary_row)

    write_csv_rows(output_dir / "teacher_predictions.csv", prediction_rows)
    write_csv_rows(output_dir / "teacher_summary.csv", summaries)
    write_json(
        output_dir / "teacher_manifest.json",
        {
            "scope": scope,
            "horizon": horizon,
            "history_length": history_length,
            "families": families,
            "rows": len(prediction_rows),
        },
    )
    return summaries
