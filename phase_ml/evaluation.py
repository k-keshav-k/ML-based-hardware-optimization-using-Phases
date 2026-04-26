"""Evaluation entry points for baseline and transformer predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, write_csv_rows, write_json

from .metrics import classification_metrics, confusion_matrix, per_workload_metrics


def evaluate_predictions(prediction_files: list[Path], output_dir: Path) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    summary_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {}
    for path in prediction_files:
        rows = load_csv_rows(path)
        if not rows:
            continue
        model = rows[0].get("model", path.stem.replace("_predictions", ""))
        # By default we report test/val quality; if split is missing we fall
        # back to the full file so external predictions still evaluate.
        eval_rows = [row for row in rows if row.get("split", "") != "train"] or rows
        y_true = np.asarray([int(row["y_true_next_phase"]) for row in eval_rows], dtype=int)
        y_pred = np.asarray([int(row["y_pred_next_phase"]) for row in eval_rows], dtype=int)
        c_true = np.asarray([int(row["y_true_phase_change"]) for row in eval_rows], dtype=int)
        c_pred = np.asarray([int(row["y_pred_phase_change"]) for row in eval_rows], dtype=int)
        metrics = classification_metrics(y_true, y_pred, c_true, c_pred)
        item: dict[str, object] = {"model": model, "samples": len(eval_rows)}
        item.update(metrics)
        summary_rows.append(item)
        cm = confusion_matrix(y_true, y_pred).tolist()
        workload_rows = per_workload_metrics(eval_rows, y_true, y_pred, c_true, c_pred)
        write_csv_rows(output_dir / f"{model}_per_workload.csv", workload_rows)
        write_json(output_dir / f"{model}_confusion_matrix.json", {"matrix": cm})
        payload[model] = {"metrics": metrics, "confusion_matrix": cm}
    write_csv_rows(output_dir / "model_comparison.csv", summary_rows)
    write_json(output_dir / "evaluation_summary.json", payload)
    return payload
