"""CLI for training the PyTorch transformer teacher."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import ensure_dir, load_csv_rows, read_json, write_csv_rows, write_json

from .config import load_config
from .features import fill_nan_with_medians
from .transformer_model import build_model, require_torch


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-every", type=int, default=1, help="Print transformer loss every N training steps.")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Batch size for prediction export after training (0 -> use training batch_size).",
    )
    args = parser.parse_args()
    torch, nn = require_torch()
    config = load_config(args.config or None)
    dataset_dir = Path(args.dataset_dir or config["dataset"]["output_dir"])
    label_dir = Path(args.label_dir or dataset_dir.parent / "labels")
    output_dir = ensure_dir(Path(args.output_dir or dataset_dir.parent / "transformer"))

    # Load windows and align them to labeled window ids so split logic is
    # driven by the canonical label CSV.
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
    eval_batch_size = int(args.eval_batch_size or batch_size)
    epochs = int(config["transformer"].get("epochs", 8))
    train_indices = np.where(train_mask)[0]
    steps_per_epoch = int(math.ceil(train_indices.size / max(1, batch_size)))
    total_steps = epochs * steps_per_epoch
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(config["transformer"].get("amp_dtype", "bf16")) == "bf16" else torch.float16
    history_rows = []
    train_started = time.perf_counter()
    global_step = 0
    print(
        f"[transformer] device={device} train_windows={train_indices.size} eval_windows={int(np.sum(eval_mask))} "
        f"epochs={epochs} batch_size={batch_size} steps_per_epoch={steps_per_epoch}",
        file=sys.stderr,
        flush=True,
    )
    for epoch_index in range(1, epochs + 1):
        # Shuffle per epoch for SGD-style training.
        np.random.shuffle(train_indices)
        model.train()
        epoch_losses = []
        for step_index, start in enumerate(range(0, train_indices.size, batch_size), start=1):
            idx = train_indices[start : start + batch_size]
            xb = torch.tensor(x[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y[idx], dtype=torch.long, device=device)
            cb = torch.tensor(change[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, change_logits = model(xb)
                next_loss = ce(logits, yb)
                change_loss = bce(change_logits, cb)
                loss = next_loss + change_loss
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_value = float(loss.detach().cpu())
            next_loss_value = float(next_loss.detach().cpu())
            change_loss_value = float(change_loss.detach().cpu())
            epoch_losses.append(loss_value)
            elapsed = time.perf_counter() - train_started
            rate = global_step / elapsed if elapsed > 0 else 0.0
            eta = (total_steps - global_step) / rate if rate > 0 else 0.0
            history_rows.append(
                {
                    "epoch": epoch_index,
                    "step": step_index,
                    "global_step": global_step,
                    "batch_size": len(idx),
                    "loss": loss_value,
                    "next_phase_loss": next_loss_value,
                    "phase_change_loss": change_loss_value,
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                }
            )
            if args.log_every > 0 and (global_step % args.log_every == 0 or step_index == steps_per_epoch):
                print(
                    f"[transformer] {progress_bar(global_step, total_steps)} "
                    f"epoch={epoch_index}/{epochs} step={step_index}/{steps_per_epoch} "
                    f"global={global_step}/{total_steps} loss={loss_value:.6f} "
                    f"next={next_loss_value:.6f} change={change_loss_value:.6f} "
                    f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )
        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"[transformer] epoch {epoch_index}/{epochs} mean_loss={mean_epoch_loss:.6f}", file=sys.stderr, flush=True)

    # Emit predictions in mini-batches to avoid GPU OOM on large datasets.
    model.eval()
    start_time = time.perf_counter()
    pred_parts: list[np.ndarray] = []
    pred_change_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], max(1, eval_batch_size)):
            xb = torch.tensor(x[start : start + eval_batch_size], dtype=torch.float32, device=device)
            logits, change_logits = model(xb)
            pred_parts.append(logits.argmax(dim=1).cpu().numpy())
            pred_change_parts.append((torch.sigmoid(change_logits) >= 0.5).cpu().numpy().astype(int))
    pred = np.concatenate(pred_parts, axis=0) if pred_parts else np.zeros(0, dtype=int)
    pred_change = np.concatenate(pred_change_parts, axis=0) if pred_change_parts else np.zeros(0, dtype=int)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    torch.save(model.state_dict(), output_dir / "transformer.pt")
    write_csv_rows(
        output_dir / "training_history.csv",
        history_rows,
        ["epoch", "step", "global_step", "batch_size", "loss", "next_phase_loss", "phase_change_loss", "elapsed_s", "eta_s"],
    )
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
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "train_windows": int(train_indices.size),
            "eval_windows": int(np.sum(eval_mask)),
            "training_steps": int(global_step),
            "final_training_loss": float(history_rows[-1]["loss"]) if history_rows else 0.0,
            "training_history_csv": str(output_dir / "training_history.csv"),
            "inference_ms_total": elapsed_ms,
            "inference_us_per_sample": elapsed_ms * 1000.0 / max(1, len(labels)),
        },
    )
    print(f"Trained transformer on {device}; wrote predictions for {len(labels)} windows.")


if __name__ == "__main__":
    main()
