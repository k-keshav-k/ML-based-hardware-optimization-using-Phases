"""Train a compact decision-tree student from transformer teacher predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, write_csv_rows, write_json

from .baselines import DecisionTree, TreeNode, aligned_arrays
from .config import load_config
from .metrics import classification_metrics


def node_to_dict(node: TreeNode | None) -> dict[str, object] | None:
    if node is None:
        return None
    payload = asdict(node)
    payload["left"] = node_to_dict(node.left)
    payload["right"] = node_to_dict(node.right)
    return payload


def load_teacher_predictions(path: Path) -> dict[int, dict[str, str]]:
    rows = load_csv_rows(path)
    if not rows:
        raise SystemExit(f"No teacher predictions found at {path}")
    return {int(row["window_id"]): row for row in rows}


def student_feature_matrix(flat: np.ndarray, current: np.ndarray, run_length: np.ndarray) -> np.ndarray:
    return np.c_[flat, current.astype(float), np.log1p(run_length.astype(float))]


def train_student(
    dataset_dir: Path,
    label_dir: Path,
    teacher_predictions: Path,
    output_dir: Path,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    arrays = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    x = arrays["X"].astype(float)
    label_rows = load_csv_rows(label_dir / "window_labels.csv")
    label_model = read_json(label_dir / "label_model.json")
    medians = np.asarray(label_model["model"]["standardizer"]["medians"], dtype=float)  # type: ignore[index]
    flat, true_next, current, true_change, run_length = aligned_arrays(x, label_rows, medians)

    # Distillation target: teacher predictions keyed by window_id.
    teacher_by_window = load_teacher_predictions(teacher_predictions)
    keep_indices: list[int] = []
    teacher_next: list[int] = []
    teacher_change: list[int] = []
    kept_rows: list[dict[str, str]] = []
    for index, row in enumerate(label_rows):
        teacher_row = teacher_by_window.get(int(row["window_id"]))
        if teacher_row is None:
            continue
        keep_indices.append(index)
        teacher_next.append(int(teacher_row["y_pred_next_phase"]))
        teacher_change.append(int(teacher_row["y_pred_phase_change"]))
        kept_rows.append(row)
    if not keep_indices:
        raise SystemExit("Teacher predictions did not overlap with labeled windows.")

    keep = np.asarray(keep_indices, dtype=int)
    flat = student_feature_matrix(flat[keep], current[keep], run_length[keep])
    true_next = true_next[keep]
    true_change = true_change[keep]
    teacher_next_array = np.asarray(teacher_next, dtype=int)
    teacher_change_array = np.asarray(teacher_change, dtype=int)
    split = np.asarray([row.get("split", "train") for row in kept_rows])
    train_mask = split == "train"
    eval_mask = split != "train"
    if not np.any(train_mask):
        raise SystemExit("Student training requires at least one train split row.")
    if not np.any(eval_mask):
        eval_mask = np.ones(split.shape[0], dtype=bool)

    phase_tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf).fit(flat[train_mask], teacher_next_array[train_mask])
    change_tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf).fit(flat[train_mask], teacher_change_array[train_mask])
    pred_next = phase_tree.predict(flat)
    pred_change = change_tree.predict(flat)

    # Student quality is reported against true labels on eval split, with
    # additional agreement metrics versus the teacher outputs.
    metrics = classification_metrics(true_next[eval_mask], pred_next[eval_mask], true_change[eval_mask], pred_change[eval_mask])
    metrics["teacher_next_agreement"] = float(np.mean(pred_next[eval_mask] == teacher_next_array[eval_mask]))
    metrics["teacher_change_agreement"] = float(np.mean(pred_change[eval_mask] == teacher_change_array[eval_mask]))
    metrics["train_windows"] = int(train_mask.sum())
    metrics["eval_windows"] = int(eval_mask.sum())

    prediction_rows = []
    for row, y_next, p_next, y_change, p_change in zip(kept_rows, true_next, pred_next, true_change, pred_change):
        prediction_rows.append(
            {
                "model": "student_decision_tree",
                "window_id": row.get("window_id", ""),
                "split": row.get("split", ""),
                "workload": row.get("workload", ""),
                "run_id": row.get("run_id", ""),
                "phase_id": row.get("phase_id", ""),
                "y_true_next_phase": int(y_next),
                "y_pred_next_phase": int(p_next),
                "y_true_phase_change": int(y_change),
                "y_pred_phase_change": int(p_change),
            }
        )
    write_csv_rows(output_dir / "student_decision_tree_predictions.csv", prediction_rows)
    write_json(
        output_dir / "student_decision_tree_model.json",
        {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "teacher_predictions": str(teacher_predictions),
            "feature_suffix": ["current_phase_id", "log1p_current_run_length"],
            "feature_count": int(flat.shape[1]),
            "phase_tree": node_to_dict(phase_tree.root),
            "phase_change_tree": node_to_dict(change_tree.root),
        },
    )
    write_json(output_dir / "student_decision_tree_summary.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--teacher-predictions", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config or None)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["output_dir"])
    label_dir = Path(args.label_dir or dataset_dir.parent / "labels")
    teacher_predictions = Path(args.teacher_predictions or dataset_dir.parent / "transformer" / "transformer_predictions.csv")
    output_dir = Path(args.output_dir or dataset_dir.parent / "student_tree")
    student_config = dict(config["student"])
    metrics = train_student(
        dataset_dir,
        label_dir,
        teacher_predictions,
        output_dir,
        args.max_depth or int(student_config["decision_tree_max_depth"]),
        args.min_samples_leaf or int(student_config["decision_tree_min_samples_leaf"]),
    )
    print(
        "Trained student decision tree; "
        f"teacher next-phase agreement={metrics['teacher_next_agreement']:.3f}, "
        f"teacher phase-change agreement={metrics['teacher_change_agreement']:.3f}."
    )


if __name__ == "__main__":
    main()
