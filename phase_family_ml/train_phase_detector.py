"""Train a shallow phase detector against offline clustered phase labels."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import write_csv_rows, write_json

from .config import load_config
from .data import load_scope_family_data, states_matrix
from .metrics import classification_metrics
from .orchestration import experiment_dirs, scopes_for_experiment
from .tree import DecisionTree, TreeNode

TREE_MODEL = "online_phase_history_tree"


def _eval_mask(split: np.ndarray) -> np.ndarray:
    val = split == "val"
    if np.any(val):
        return val
    test = split == "test"
    if np.any(test):
        return test
    return split != "train"


def _fill_train_medians(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col in range(filled.shape[1]):
        values = filled[train_mask, col]
        clean = values[np.isfinite(values)]
        median = float(np.median(clean)) if clean.size else 0.0
        filled[~np.isfinite(filled[:, col]), col] = median
    return filled


def _node_counts(node: TreeNode | None, depth: int = 0) -> tuple[int, int, int]:
    if node is None:
        return 0, 0, depth
    if node.feature < 0 or node.left is None or node.right is None:
        return 0, 1, depth
    left_internal, left_leaves, left_depth = _node_counts(node.left, depth + 1)
    right_internal, right_leaves, right_depth = _node_counts(node.right, depth + 1)
    return 1 + left_internal + right_internal, left_leaves + right_leaves, max(left_depth, right_depth)


def _tree_storage_bytes(internal_nodes: int, leaves: int, feature_count: int) -> float:
    if feature_count <= 0:
        return 0.0
    total_nodes = internal_nodes + leaves
    feature_bits = max(1, math.ceil(math.log2(feature_count)))
    child_bits = 2 * max(1, math.ceil(math.log2(max(1, total_nodes))))
    threshold_bits = 16
    phase_bits = 2
    bits = internal_nodes * (feature_bits + threshold_bits + child_bits) + leaves * phase_bits
    return bits / 8.0


def _group_indices(metadata_rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(metadata_rows):
        key = (str(row.get("run_id", "")), str(row.get("core_id", "")))
        grouped.setdefault(key, []).append(index)
    for key in grouped:
        grouped[key].sort(key=lambda idx: float(metadata_rows[idx].get("timestamp_ms", "0") or 0.0))
    return list(grouped.values())


def _history_examples(
    values: np.ndarray,
    phases: np.ndarray,
    split: np.ndarray,
    metadata_rows: list[dict[str, str]],
    history_length: int,
    prediction_horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[list[int]] = []
    splits: list[str] = []
    rows: list[dict[str, str]] = []
    row_ids: list[int] = []
    current_phase: list[int] = []
    for stream in _group_indices(metadata_rows):
        if len(stream) < history_length + 1:
            continue
        for pos in range(history_length - 1, len(stream)):
            row_index = stream[pos]
            target_values: list[int] = []
            for step in range(1, prediction_horizon + 1):
                target_pos = pos + step
                target_values.append(int(phases[stream[target_pos]]) if target_pos < len(stream) else -1)
            window = stream[pos - history_length + 1 : pos + 1]
            features.append(values[window].reshape(-1))
            targets.append(target_values)
            splits.append(str(split[row_index]))
            rows.append(metadata_rows[row_index])
            row_ids.append(row_index)
            current_phase.append(int(phases[row_index]))
    if not features:
        width = values.shape[1] * history_length
        return (
            np.empty((0, width), dtype=float),
            np.empty((0, prediction_horizon), dtype=int),
            np.empty(0, dtype=object),
            [],
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=int),
        np.asarray(splits, dtype=object),
        rows,
        np.asarray(row_ids, dtype=int),
        np.asarray(current_phase, dtype=int),
    )


def _topk_accuracy(y_true: np.ndarray, topk: np.ndarray) -> float:
    valid = y_true >= 0
    if not np.any(valid):
        return 0.0
    hits = [int(y_true[index]) in set(topk[index].astype(int).tolist()) for index in np.where(valid)[0]]
    return float(np.mean(hits)) if hits else 0.0


def _subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & (y_true >= 0)
    if not np.any(usable):
        return 0.0
    return float(np.mean(y_true[usable] == y_pred[usable]))


def _last_state_baseline(current_phase: np.ndarray) -> np.ndarray:
    return current_phase.astype(int).copy()


def _majority_baseline(target: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_target = target[train_mask & (target >= 0)]
    if train_target.size == 0:
        train_target = target[target >= 0]
    majority = int(np.bincount(train_target, minlength=3).argmax()) if train_target.size else 0
    return np.full(target.shape[0], majority, dtype=int)


def _state_conditioned_baseline(current_phase: np.ndarray, target: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    majority = _majority_baseline(target, train_mask)
    fallback = int(majority[0]) if majority.size else 0
    mapping: dict[int, int] = {}
    for state in sorted(set(current_phase[train_mask].astype(int).tolist())):
        mask = train_mask & (current_phase == state) & (target >= 0)
        if np.any(mask):
            mapping[int(state)] = int(np.bincount(target[mask], minlength=3).argmax())
    pred = np.full(target.shape[0], fallback, dtype=int)
    for index, state in enumerate(current_phase.astype(int)):
        pred[index] = mapping.get(int(state), fallback)
    return pred


def _summary_row(
    *,
    experiment: str,
    scope: str,
    model: str,
    prediction_step: int,
    history_length: int,
    prediction_horizon: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_phase: np.ndarray,
    eval_mask: np.ndarray,
    selected_counter_map: str,
    train_rows: int,
    eval_rows: int,
    tree_depth: int | str = "",
    tree_max_depth: int | str = "",
    tree_internal_nodes: int | str = "",
    tree_leaves: int | str = "",
    estimated_storage_bytes: float | str = "",
    feature_count: int | str = "",
) -> dict[str, object]:
    metrics = classification_metrics(y_true[eval_mask], y_pred[eval_mask], current_state=current_phase[eval_mask])
    stable_mask = eval_mask & (y_true == current_phase)
    transition_mask = eval_mask & (y_true != current_phase)
    return {
        "experiment": experiment,
        "scope": scope,
        "model": model,
        "feature_source": "selected_counter_history" if model == TREE_MODEL else "baseline",
        "phase_label_source": "train_split_kmeans_full_safe_counters",
        "prediction_step": prediction_step,
        "history_length": history_length,
        "prediction_horizon": prediction_horizon,
        "top1_accuracy": float(np.mean(y_true[eval_mask] == y_pred[eval_mask])) if np.any(eval_mask) else 0.0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "high_usage_recall": metrics["high_usage_recall"],
        "stable_case_accuracy": _subset_accuracy(y_true, y_pred, stable_mask),
        "transition_case_accuracy": _subset_accuracy(y_true, y_pred, transition_mask),
        "stable_eval_rows": int(np.sum(stable_mask)),
        "transition_eval_rows": int(np.sum(transition_mask)),
        "tree_depth": tree_depth,
        "tree_max_depth": tree_max_depth,
        "tree_internal_nodes": tree_internal_nodes,
        "tree_leaves": tree_leaves,
        "estimated_storage_bytes": estimated_storage_bytes,
        "feature_count": feature_count,
        "selected_counter_map": selected_counter_map,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def _mean_metric(rows: list[dict[str, object]], model: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row.get("model") == model and key in row]
    return float(np.mean(values)) if values else 0.0


def _prepare_phase_detector_data(
    *,
    experiment_dir: Path,
    scope: str,
    horizon: int,
    history_length: int,
    prediction_horizon: int,
) -> dict[str, object]:
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    families, current, _future, split, metadata_rows = states_matrix(family_data, horizon)
    if not families:
        raise SystemExit(
            f"No selected family sequence CSVs found for experiment={experiment_dir.name} scope={scope}. "
            f"Run ablation and rebuild counter sequences first."
        )

    features = np.full((current.shape[0], len(families)), np.nan, dtype=float)
    selected: dict[str, str] = {}
    for family_index, family in enumerate(families):
        rows = family_data[family].rows[: current.shape[0]]
        for row_index, row in enumerate(rows):
            if row_index == 0:
                selected[family] = str(row.get("selected_counter", row.get("counter_name", "")))
            value = row.get("counter_value", "")
            try:
                features[row_index, family_index] = float(value)
            except (TypeError, ValueError):
                features[row_index, family_index] = math.nan

    phases = current[:, 0].astype(int)
    valid = phases >= 0
    if not np.any(valid):
        raise SystemExit(f"No valid clustered labels for experiment={experiment_dir.name} scope={scope}")
    raw_features, targets, example_split, example_rows, row_ids, current_phase = _history_examples(
        features,
        phases,
        split,
        metadata_rows,
        history_length,
        prediction_horizon,
    )
    if raw_features.shape[0] == 0:
        raise SystemExit(f"No history windows for experiment={experiment_dir.name} scope={scope}")
    train_base = example_split == "train"
    if not np.any(train_base):
        train_base = np.ones(example_split.shape[0], dtype=bool)
    hist_features = _fill_train_medians(raw_features, train_base)
    eval_base = _eval_mask(example_split)
    selected_counter_map = ";".join(f"{family}:{selected.get(family, '')}" for family in families)
    return {
        "families": families,
        "features": features,
        "targets": targets,
        "example_split": example_split,
        "example_rows": example_rows,
        "row_ids": row_ids,
        "current_phase": current_phase,
        "hist_features": hist_features,
        "train_base": train_base,
        "eval_base": eval_base,
        "selected_counter_map": selected_counter_map,
        "experiment": experiment_dir.name,
        "scope": scope,
        "history_length": history_length,
        "prediction_horizon": prediction_horizon,
    }


def _train_phase_detector_from_prepared(
    prepared: dict[str, object],
    *,
    tree_max_depth: int,
    tree_min_leaf: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    hist_features = np.asarray(prepared["hist_features"], dtype=float)
    targets = np.asarray(prepared["targets"], dtype=int)
    example_split = np.asarray(prepared["example_split"], dtype=object)
    example_rows = list(prepared["example_rows"])
    row_ids = np.asarray(prepared["row_ids"], dtype=int)
    current_phase = np.asarray(prepared["current_phase"], dtype=int)
    train_base = np.asarray(prepared["train_base"], dtype=bool)
    eval_base = np.asarray(prepared["eval_base"], dtype=bool)
    features = np.asarray(prepared["features"], dtype=float)
    families = list(prepared["families"])
    selected_counter_map = str(prepared["selected_counter_map"])
    experiment = str(prepared["experiment"])
    scope = str(prepared["scope"])
    history_length = int(prepared["history_length"])
    prediction_horizon = int(prepared["prediction_horizon"])

    preds = np.full((hist_features.shape[0], prediction_horizon), -1, dtype=int)
    summary_rows: list[dict[str, object]] = []
    total_internal = 0
    total_leaves = 0
    max_actual_depth = 0

    for step in range(1, prediction_horizon + 1):
        target = targets[:, step - 1].astype(int)
        step_valid = target >= 0
        train_mask = train_base & step_valid
        if not np.any(train_mask):
            train_mask = step_valid
        if not np.any(train_mask):
            continue
        tree = DecisionTree(max_depth=tree_max_depth, min_samples_leaf=tree_min_leaf)
        tree.fit(hist_features[train_mask], target[train_mask])
        pred = tree.predict(hist_features)
        preds[:, step - 1] = pred

        eval_mask = eval_base & step_valid
        if not np.any(eval_mask):
            eval_mask = step_valid
        internal_nodes, leaves, actual_depth = _node_counts(tree.root)
        total_internal += internal_nodes
        total_leaves += leaves
        max_actual_depth = max(max_actual_depth, actual_depth)
        tree_storage = _tree_storage_bytes(internal_nodes, leaves, hist_features.shape[1])
        tree_row = _summary_row(
            experiment=experiment,
            scope=scope,
            model=TREE_MODEL,
            prediction_step=step,
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            y_true=target,
            y_pred=pred,
            current_phase=current_phase,
            eval_mask=eval_mask,
            selected_counter_map=selected_counter_map,
            train_rows=int(np.sum(train_mask)),
            eval_rows=int(np.sum(eval_mask)),
            tree_depth=actual_depth,
            tree_max_depth=tree_max_depth,
            tree_internal_nodes=internal_nodes,
            tree_leaves=leaves,
            estimated_storage_bytes=tree_storage,
            feature_count=hist_features.shape[1],
        )

        last_pred = _last_state_baseline(current_phase)
        majority_pred = _majority_baseline(target, train_mask)
        state_conditioned_pred = _state_conditioned_baseline(current_phase, target, train_mask)

        last_row = _summary_row(
            experiment=experiment,
            scope=scope,
            model="baseline_last_state",
            prediction_step=step,
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            y_true=target,
            y_pred=last_pred,
            current_phase=current_phase,
            eval_mask=eval_mask,
            selected_counter_map=selected_counter_map,
            train_rows=int(np.sum(train_mask)),
            eval_rows=int(np.sum(eval_mask)),
        )
        majority_row = _summary_row(
            experiment=experiment,
            scope=scope,
            model="baseline_majority",
            prediction_step=step,
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            y_true=target,
            y_pred=majority_pred,
            current_phase=current_phase,
            eval_mask=eval_mask,
            selected_counter_map=selected_counter_map,
            train_rows=int(np.sum(train_mask)),
            eval_rows=int(np.sum(eval_mask)),
        )
        state_conditioned_row = _summary_row(
            experiment=experiment,
            scope=scope,
            model="baseline_state_conditioned_majority",
            prediction_step=step,
            history_length=history_length,
            prediction_horizon=prediction_horizon,
            y_true=target,
            y_pred=state_conditioned_pred,
            current_phase=current_phase,
            eval_mask=eval_mask,
            selected_counter_map=selected_counter_map,
            train_rows=int(np.sum(train_mask)),
            eval_rows=int(np.sum(eval_mask)),
        )

        tree_row["accuracy_gain_vs_last_state"] = float(tree_row["accuracy"]) - float(last_row["accuracy"])
        tree_row["macro_f1_gain_vs_last_state"] = float(tree_row["macro_f1"]) - float(last_row["macro_f1"])
        tree_row["transition_accuracy_gain_vs_last_state"] = float(tree_row["transition_case_accuracy"]) - float(last_row["transition_case_accuracy"])
        tree_row["accuracy_gain_vs_majority"] = float(tree_row["accuracy"]) - float(majority_row["accuracy"])
        tree_row["macro_f1_gain_vs_majority"] = float(tree_row["macro_f1"]) - float(majority_row["macro_f1"])
        tree_row["accuracy_gain_vs_state_conditioned_majority"] = float(tree_row["accuracy"]) - float(state_conditioned_row["accuracy"])
        tree_row["macro_f1_gain_vs_state_conditioned_majority"] = float(tree_row["macro_f1"]) - float(state_conditioned_row["macro_f1"])

        summary_rows.extend([tree_row, last_row, majority_row, state_conditioned_row])

    prediction_rows: list[dict[str, object]] = []
    for example_index, row in enumerate(example_rows):
        item: dict[str, object] = {
            "model": TREE_MODEL,
            "split": example_split[example_index],
            "workload": row.get("workload", ""),
            "run_id": row.get("run_id", ""),
            "core_id": row.get("core_id", ""),
            "row_index": int(row.get("row_index", row_ids[example_index]) or row_ids[example_index]),
            "current_phase": int(current_phase[example_index]),
            "history_length": history_length,
        }
        for family_index, family in enumerate(families):
            item[f"current_value_{family}"] = features[row_ids[example_index], family_index]
        for step in range(1, prediction_horizon + 1):
            item[f"y_true_phase_t_plus_{step}"] = int(targets[example_index, step - 1])
            item[f"y_pred_phase_t_plus_{step}"] = int(preds[example_index, step - 1])
        prediction_rows.append(item)

    aggregate = {
        "experiment": experiment,
        "scope": scope,
        "model": TREE_MODEL,
        "history_length": history_length,
        "prediction_horizon": prediction_horizon,
        "tree_count": len([row for row in summary_rows if row.get("model") == TREE_MODEL]),
        "tree_depth": max_actual_depth,
        "tree_max_depth": tree_max_depth,
        "tree_internal_nodes": total_internal,
        "tree_leaves": total_leaves,
        "estimated_storage_bytes": _tree_storage_bytes(total_internal, total_leaves, hist_features.shape[1]),
        "mean_top1_accuracy": _mean_metric(summary_rows, TREE_MODEL, "top1_accuracy"),
        "mean_accuracy": _mean_metric(summary_rows, TREE_MODEL, "accuracy"),
        "mean_macro_f1": _mean_metric(summary_rows, TREE_MODEL, "macro_f1"),
        "mean_high_usage_recall": _mean_metric(summary_rows, TREE_MODEL, "high_usage_recall"),
        "mean_stable_case_accuracy": _mean_metric(summary_rows, TREE_MODEL, "stable_case_accuracy"),
        "mean_transition_case_accuracy": _mean_metric(summary_rows, TREE_MODEL, "transition_case_accuracy"),
        "mean_accuracy_gain_vs_last_state": _mean_metric(summary_rows, TREE_MODEL, "accuracy_gain_vs_last_state"),
        "mean_macro_f1_gain_vs_last_state": _mean_metric(summary_rows, TREE_MODEL, "macro_f1_gain_vs_last_state"),
        "mean_transition_accuracy_gain_vs_last_state": _mean_metric(summary_rows, TREE_MODEL, "transition_accuracy_gain_vs_last_state"),
        "feature_count": hist_features.shape[1],
        "selected_counter_map": selected_counter_map,
    }
    return summary_rows, prediction_rows, aggregate


def run_phase_detector_depth_sweep_for_experiment(
    *,
    experiment_dir: Path,
    scope: str,
    output_dir: Path,
    horizon: int,
    history_length: int,
    prediction_horizon: int,
    tree_depths: list[int],
    tree_min_leaf: int,
) -> list[dict[str, object]]:
    prepared = _prepare_phase_detector_data(
        experiment_dir=experiment_dir,
        scope=scope,
        horizon=horizon,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, object]] = []
    for depth in sorted({int(item) for item in tree_depths if int(item) > 0}):
        summary_rows, _prediction_rows, aggregate = _train_phase_detector_from_prepared(
            prepared,
            tree_max_depth=depth,
            tree_min_leaf=tree_min_leaf,
        )
        sweep_rows.append(
            {
                "experiment": aggregate["experiment"],
                "scope": aggregate["scope"],
                "model": aggregate["model"],
                "history_length": aggregate["history_length"],
                "prediction_horizon": aggregate["prediction_horizon"],
                "tree_max_depth": depth,
                "actual_max_depth": aggregate["tree_depth"],
                "tree_count": aggregate["tree_count"],
                "tree_internal_nodes": aggregate["tree_internal_nodes"],
                "tree_leaves": aggregate["tree_leaves"],
                "estimated_storage_bytes": aggregate["estimated_storage_bytes"],
                "feature_count": aggregate["feature_count"],
                "selected_counter_map": aggregate["selected_counter_map"],
                "mean_top1_accuracy": aggregate["mean_top1_accuracy"],
                "mean_accuracy": aggregate["mean_accuracy"],
                "mean_macro_f1": aggregate["mean_macro_f1"],
                "mean_high_usage_recall": aggregate["mean_high_usage_recall"],
                "mean_stable_case_accuracy": aggregate["mean_stable_case_accuracy"],
                "mean_transition_case_accuracy": aggregate["mean_transition_case_accuracy"],
                "mean_accuracy_gain_vs_last_state": aggregate["mean_accuracy_gain_vs_last_state"],
                "mean_macro_f1_gain_vs_last_state": aggregate["mean_macro_f1_gain_vs_last_state"],
                "mean_transition_accuracy_gain_vs_last_state": aggregate["mean_transition_accuracy_gain_vs_last_state"],
                "step_1_accuracy": _mean_metric([row for row in summary_rows if int(row.get("prediction_step", 0)) == 1], TREE_MODEL, "accuracy"),
                "step_5_accuracy": _mean_metric([row for row in summary_rows if int(row.get("prediction_step", 0)) == prediction_horizon], TREE_MODEL, "accuracy"),
            }
        )
    write_csv_rows(output_dir / "phase_detector_depth_sweep_summary.csv", sweep_rows)
    return sweep_rows


def train_phase_detector_for_experiment(
    *,
    experiment_dir: Path,
    scope: str,
    output_dir: Path,
    horizon: int,
    history_length: int,
    prediction_horizon: int,
    tree_max_depth: int,
    tree_min_leaf: int,
) -> list[dict[str, object]]:
    """Train reduced-counter history trees to predict future clustered labels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_phase_detector_data(
        experiment_dir=experiment_dir,
        scope=scope,
        horizon=horizon,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
    )
    summary_rows, prediction_rows, aggregate = _train_phase_detector_from_prepared(
        prepared,
        tree_max_depth=tree_max_depth,
        tree_min_leaf=tree_min_leaf,
    )
    write_csv_rows(output_dir / "phase_detector_predictions.csv", prediction_rows)
    write_csv_rows(output_dir / "phase_detector_summary.csv", summary_rows)
    write_json(output_dir / "phase_detector_manifest.json", aggregate)
    return summary_rows


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--prediction-horizon", type=int, default=0)
    parser.add_argument("--tree-max-depth", type=int, default=0)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=0)
    parser.add_argument("--depth-sweep-depths", default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "phase_detector"))
    detector_cfg = dict(config["phase_detector"])
    history_length = args.history_length or int(detector_cfg["history_length"])
    prediction_horizon = args.prediction_horizon or int(detector_cfg["prediction_horizon"])
    tree_max_depth = args.tree_max_depth or int(detector_cfg["decision_tree_max_depth"])
    tree_min_leaf = args.tree_min_samples_leaf or int(detector_cfg["decision_tree_min_samples_leaf"])
    raw_depths = str(args.depth_sweep_depths or "")
    if not raw_depths:
        raw_cfg = detector_cfg.get("depth_sweep_depths", [])
        if isinstance(raw_cfg, list):
            raw_depths = ",".join(str(item) for item in raw_cfg)
    sweep_depths = [int(item.strip()) for item in raw_depths.split(",") if item.strip()]
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for exp_dir in experiment_dirs(sequences_root):
        for scope in scopes_for_experiment(exp_dir):
            rows = train_phase_detector_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                output_dir=output_root / exp_dir.name / scope,
                horizon=int(dataset_cfg["horizon"]),
                history_length=history_length,
                prediction_horizon=prediction_horizon,
                tree_max_depth=tree_max_depth,
                tree_min_leaf=tree_min_leaf,
            )
            all_rows.extend(rows)
            if sweep_depths:
                sweep_rows.extend(
                    run_phase_detector_depth_sweep_for_experiment(
                        experiment_dir=exp_dir,
                        scope=scope,
                        output_dir=output_root / exp_dir.name / scope,
                        horizon=int(dataset_cfg["horizon"]),
                        history_length=history_length,
                        prediction_horizon=prediction_horizon,
                        tree_depths=sweep_depths,
                        tree_min_leaf=tree_min_leaf,
                    )
                )
    write_csv_rows(output_root / "phase_detector_summary_all.csv", all_rows)
    if sweep_rows:
        write_csv_rows(output_root / "phase_detector_depth_sweep_summary_all.csv", sweep_rows)
    print(f"[phase_detector] wrote_summary={output_root / 'phase_detector_summary_all.csv'} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
