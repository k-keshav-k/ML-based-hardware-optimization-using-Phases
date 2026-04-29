"""CLI for direct true-label decision-tree training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import write_csv_rows, write_json

from .config import load_config
from .data import load_scope_family_data, states_matrix
from .metrics import classification_metrics
from .orchestration import experiment_dirs, scopes_for_experiment
from .teacher import _build_examples
from .tree import DecisionTree


def _eval_mask(split: np.ndarray) -> np.ndarray:
    mask = split != "train"
    return mask if np.any(mask) else np.ones(split.shape[0], dtype=bool)


def train_direct_trees_for_experiment(
    *,
    experiment_dir: Path,
    scope: str,
    output_dir: Path,
    horizon: int,
    history_length: int,
    context_mode: str,
    tree_max_depth: int,
    tree_min_leaf: int,
) -> list[dict[str, object]]:
    """Train true-label history decision trees for each family."""

    output_dir.mkdir(parents=True, exist_ok=True)
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    families, current, future, split, metadata_rows = states_matrix(family_data, horizon)
    if not families:
        raise SystemExit(
            f"No family counter sequence CSVs found for experiment={experiment_dir.name} scope={scope}. "
            f"Expected files under {experiment_dir / f'threshold_{scope}'} like counter_sequence_L1.csv."
        )

    print(
        f"[direct_tree] experiment={experiment_dir.name} scope={scope} families={len(families)} rows={current.shape[0]}",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for family_index, family in enumerate(families):
        print(f"[direct_tree] family={family} start context_mode={context_mode}", flush=True)
        x_hist, y_hist, split_hist, current_hist, meta_hist, row_ids_hist = _build_examples(
            family_index,
            current,
            future,
            split,
            metadata_rows,
            history_length,
            context_mode,
        )
        if x_hist.shape[0] == 0:
            print(f"[direct_tree] family={family} skipped=no_history_examples", flush=True)
            continue

        features = x_hist.reshape(x_hist.shape[0], -1)
        train_mask = split_hist == "train"
        if not np.any(train_mask):
            train_mask = np.ones(split_hist.shape[0], dtype=bool)
        eval_mask = _eval_mask(split_hist)
        preds = np.full((x_hist.shape[0], horizon), -1, dtype=int)

        for step in range(1, horizon + 1):
            print(f"[direct_tree] family={family} step={step} start", flush=True)
            target = y_hist[:, step - 1].astype(int)
            valid = target >= 0
            fit_mask = train_mask & valid
            if not np.any(fit_mask):
                fit_mask = valid
            if not np.any(fit_mask):
                continue
            tree = DecisionTree(max_depth=tree_max_depth, min_samples_leaf=tree_min_leaf)
            tree.fit(features[fit_mask], target[fit_mask])
            preds[:, step - 1] = tree.predict(features)
            print(f"[direct_tree] family={family} step={step} done fit_rows={int(np.sum(fit_mask))}", flush=True)

        metrics = classification_metrics(y_hist[eval_mask, 0], preds[eval_mask, 0], current_state=current_hist[eval_mask])
        summary_rows.append(
            {
                "family": family,
                "scope": scope,
                "model": "decision_tree_true_history",
                "feature_source": "state_history",
                "context_mode": context_mode,
                "history_length": history_length,
                "tree_depth": tree_max_depth,
                "rows": int(x_hist.shape[0]),
                **metrics,
            }
        )

        for i, row in enumerate(meta_hist):
            item: dict[str, object] = {
                "family": family,
                "scope": scope,
                "split": split_hist[i],
                "workload": row.get("workload", ""),
                "run_id": row.get("run_id", ""),
                "core_id": row.get("core_id", ""),
                "row_index": int(row_ids_hist[i]),
                "family_state": int(current_hist[i]),
                "model": "decision_tree_true_history",
                "feature_source": "state_history",
                "context_mode": context_mode,
            }
            for step in range(1, horizon + 1):
                item[f"y_true_future_state_{step}"] = int(y_hist[i, step - 1])
                item[f"y_pred_future_state_{step}"] = int(preds[i, step - 1])
            prediction_rows.append(item)
        print(f"[direct_tree] family={family} done accuracy={metrics['accuracy']:.4f}", flush=True)

    write_csv_rows(output_dir / "direct_tree_summary.csv", summary_rows)
    write_csv_rows(output_dir / "direct_tree_predictions.csv", prediction_rows)
    write_json(
        output_dir / "direct_tree_manifest.json",
        {
            "scope": scope,
            "horizon": horizon,
            "history_length": history_length,
            "context_mode": context_mode,
            "tree_max_depth": tree_max_depth,
            "tree_min_leaf": tree_min_leaf,
            "rows": len(prediction_rows),
        },
    )
    return summary_rows


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--context-mode", choices=["with_context", "without_context"], default="with_context")
    parser.add_argument("--tree-max-depth", type=int, default=0)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "direct_tree"))
    tree_max_depth = args.tree_max_depth or int(config["student"]["decision_tree_max_depth"])
    tree_min_leaf = args.tree_min_samples_leaf or int(config["student"]["decision_tree_min_samples_leaf"])
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[direct_tree] config={args.config or '<default>'}", flush=True)
    print(f"[direct_tree] sequences_root={sequences_root}", flush=True)
    print(f"[direct_tree] output_root={output_root}", flush=True)
    print("[direct_tree] device=cpu", flush=True)
    print(f"[direct_tree] context_mode={args.context_mode}", flush=True)
    print(f"[direct_tree] tree_max_depth={tree_max_depth}", flush=True)
    print(f"[direct_tree] tree_min_samples_leaf={tree_min_leaf}", flush=True)

    exp_dirs = experiment_dirs(sequences_root)
    if not exp_dirs:
        raise SystemExit(f"No experiment directories with counter_sequence_manifest.json found in {sequences_root}")

    all_rows: list[dict[str, object]] = []
    for exp_dir in exp_dirs:
        scopes = scopes_for_experiment(exp_dir)
        print(f"[direct_tree] experiment={exp_dir.name} scopes={','.join(scopes) or '<none>'}", flush=True)
        for scope in scopes:
            rows = train_direct_trees_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                output_dir=output_root / exp_dir.name / scope,
                horizon=int(dataset_cfg["horizon"]),
                history_length=int(dataset_cfg["history_length"]),
                context_mode=args.context_mode,
                tree_max_depth=tree_max_depth,
                tree_min_leaf=tree_min_leaf,
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                all_rows.append(item)
            print(f"[direct_tree] done experiment={exp_dir.name} scope={scope} summary_rows={len(rows)}", flush=True)

    write_csv_rows(output_root / "direct_tree_summary_all.csv", all_rows)
    print(f"[direct_tree] wrote_summary={output_root / 'direct_tree_summary_all.csv'} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
