"""CLI for training the PyTorch transformer teacher."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, write_csv_rows, write_json

from .config import load_config
from .features import fill_nan_with_medians
from .transformer_model import build_model, require_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    torch, nn = require_torch()
    config = load_config(args.config or None)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["output_dir"])
    label_dir = Path(args.label_dir or dataset_dir.parent / "labels")
    output_dir = ensure_dir(Path(args.output_dir or dataset_dir.parent / "transformer"))
    arrays = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    labels = load_csv_rows(label_dir / "window_labels.csv")
    label_model = read_json(label_dir / "label_model.json")
    medians = np.asarray(label_model["model"]["standardizer"]["medians"], dtype=float)
    used_indices = np.asarray([int(row["window_id"]) for row in labels], dtype=int)
    x = arrays["X"].astype(float)[used_indices]
    x = fill_nan_with_medians(x.reshape(-1, x.shape[-1]), medians).reshape(x.shape)
    y = np.asarray([int(row["next_phase_id"]) for row in labels], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in labels], dtype=float)
    split = np.asarray([row.get("split", "train") for row in labels])
    train_mask = split == "train"
    eval_mask = split != "train"
    if not np.any(eval_mask):
        eval_mask = np.ones(split.shape[0], dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_count = int(max((int(np.max(y)) + 1) if y.size else 1, int(label_model.get("phase_count", 1))))
    model = build_model(x.shape[-1], phase_count, config["transformer"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["transformer"].get("learning_rate", 0.0003)),
        weight_decay=float(config["transformer"].get("weight_decay", 0.01)),
    )
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    batch_size = int(config["transformer"].get("batch_size", 256))
    epochs = int(config["transformer"].get("epochs", 8))
    train_indices = np.where(train_mask)[0]
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(config["transformer"].get("amp_dtype", "bf16")) == "bf16" else torch.float16
    for _ in range(epochs):
        np.random.shuffle(train_indices)
        model.train()
        for start in range(0, train_indices.size, batch_size):
            idx = train_indices[start : start + batch_size]
            xb = torch.tensor(x[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y[idx], dtype=torch.long, device=device)
            cb = torch.tensor(change[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, change_logits = model(xb)
                loss = ce(logits, yb) + bce(change_logits, cb)
            loss.backward()
            optimizer.step()
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        logits, change_logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        pred_change = (torch.sigmoid(change_logits) >= 0.5).cpu().numpy().astype(int)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    torch.save(model.state_dict(), output_dir / "transformer.pt")
    prediction_rows = []
    for row, true_phase, predicted, true_change, predicted_change in zip(labels, y, pred, change.astype(int), pred_change):
        prediction_rows.append(
            {
                "model": "transformer",
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
    write_csv_rows(output_dir / "transformer_predictions.csv", prediction_rows)
    write_json(
        output_dir / "transformer_summary.json",
        {
            "device": str(device),
            "phase_count": phase_count,
            "inference_ms_total": elapsed_ms,
            "inference_us_per_sample": elapsed_ms * 1000.0 / max(1, len(labels)),
        },
    )
    print(f"Trained transformer on {device}; wrote predictions for {len(labels)} windows.")


if __name__ == "__main__":
    main()
