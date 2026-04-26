"""CLI for training the PyTorch RoPE phase-LM teacher."""

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
    parser.add_argument("--log-every", type=int, default=1, help="Print phase-LM loss every N training steps.")
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
    means = np.asarray(label_model["model"]["standardizer"]["means"], dtype=float)
    scales = np.asarray(label_model["model"]["standardizer"]["scales"], dtype=float)
    scales[scales < 1e-12] = 1.0
    used_indices = np.asarray([int(row["window_id"]) for row in labels], dtype=int)
    x = arrays["X"].astype(float)[used_indices]
    x = fill_nan_with_medians(x.reshape(-1, x.shape[-1]), medians).reshape(x.shape)
    x = (x - means) / scales
    y = np.asarray([int(row["next_phase_id"]) for row in labels], dtype=int)
    change = np.asarray([int(row["phase_change"]) for row in labels], dtype=float)
    split = np.asarray([row.get("split", "train") for row in labels])
    train_mask = split == "train"
    eval_mask = split != "train"
    if not np.any(eval_mask):
        eval_mask = np.ones(split.shape[0], dtype=bool)
    val_mask = split == "val"
    if not np.any(val_mask):
        val_mask = eval_mask
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_count = int(max((int(np.max(y)) + 1) if y.size else 1, int(label_model.get("phase_count", 1))))
    model = build_model(x.shape[-1], phase_count, config["transformer"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["transformer"].get("learning_rate", 0.0003)),
        weight_decay=float(config["transformer"].get("weight_decay", 0.01)),
    )
    class_weight_power = float(config["transformer"].get("class_weight_power", 0.0))
    ce_weight = None
    if class_weight_power > 0.0 and train_indices.size > 0:
        counts = np.bincount(y[train_indices], minlength=phase_count).astype(float)
        counts[counts < 1.0] = 1.0
        weights = (counts.sum() / counts) ** class_weight_power
        weights = weights / max(1e-12, weights.mean())
        ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=ce_weight)
    bce = nn.BCEWithLogitsLoss()
    change_loss_weight = float(config["transformer"].get("change_loss_weight", 1.0))
    batch_size = int(config["transformer"].get("batch_size", 256))
    eval_batch_size = int(args.eval_batch_size or batch_size)
    epochs = int(config["transformer"].get("epochs", 8))
    steps_per_epoch = int(math.ceil(train_indices.size / max(1, batch_size)))
    total_steps = epochs * steps_per_epoch
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(config["transformer"].get("amp_dtype", "bf16")) == "bf16" else torch.float16
    history_rows = []
    validation_rows = []
    train_started = time.perf_counter()
    global_step = 0
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = None
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
                loss = next_loss + change_loss_weight * change_loss
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
        model.eval()
        val_losses = []
        val_next_losses = []
        val_change_losses = []
        val_correct = 0
        val_change_correct = 0
        val_seen = 0
        with torch.no_grad():
            for start in range(0, val_indices.size, max(1, eval_batch_size)):
                idx = val_indices[start : start + eval_batch_size]
                xb = torch.tensor(x[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(y[idx], dtype=torch.long, device=device)
                cb = torch.tensor(change[idx], dtype=torch.float32, device=device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits, change_logits = model(xb)
                    next_loss = ce(logits, yb)
                    change_loss = bce(change_logits, cb)
                    loss = next_loss + change_loss_weight * change_loss
                pred = logits.argmax(dim=1)
                pred_change = torch.sigmoid(change_logits) >= 0.5
                val_correct += int((pred == yb).sum().detach().cpu())
                val_change_correct += int((pred_change == cb.bool()).sum().detach().cpu())
                val_seen += int(yb.numel())
                val_losses.append(float(loss.detach().cpu()))
                val_next_losses.append(float(next_loss.detach().cpu()))
                val_change_losses.append(float(change_loss.detach().cpu()))
        val_accuracy = val_correct / max(1, val_seen)
        val_change_accuracy = val_change_correct / max(1, val_seen)
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        validation_rows.append(
            {
                "epoch": epoch_index,
                "loss": mean_val_loss,
                "next_phase_loss": sum(val_next_losses) / len(val_next_losses) if val_next_losses else 0.0,
                "phase_change_loss": sum(val_change_losses) / len(val_change_losses) if val_change_losses else 0.0,
                "accuracy": val_accuracy,
                "phase_change_accuracy": val_change_accuracy,
                "samples": val_seen,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch_index
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[transformer] epoch {epoch_index}/{epochs} mean_loss={mean_epoch_loss:.6f} "
            f"val_loss={mean_val_loss:.6f} val_acc={val_accuracy:.4f} "
            f"val_change_acc={val_change_accuracy:.4f}",
            file=sys.stderr,
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"[transformer] restored best epoch={best_epoch} val_acc={best_val_accuracy:.4f}", file=sys.stderr, flush=True)

    summary_payload = {
        "device": str(device),
        "phase_count": phase_count,
        "model_type": "phase_lm_rope",
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "feature_standardization": "label_model_standardizer",
        "class_weight_power": class_weight_power,
        "change_loss_weight": change_loss_weight,
        "train_windows": int(train_indices.size),
        "eval_windows": int(np.sum(eval_mask)),
        "training_steps": int(global_step),
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_val_accuracy),
        "validation_history_csv": str(output_dir / "validation_history.csv"),
        "final_training_loss": float(history_rows[-1]["loss"]) if history_rows else 0.0,
        "training_history_csv": str(output_dir / "training_history.csv"),
        "prediction_export_complete": False,
        "inference_ms_total": 0.0,
        "inference_us_per_sample": 0.0,
    }
    torch.save(model.state_dict(), output_dir / "transformer.pt")
    write_csv_rows(
        output_dir / "training_history.csv",
        history_rows,
        ["epoch", "step", "global_step", "batch_size", "loss", "next_phase_loss", "phase_change_loss", "elapsed_s", "eta_s"],
    )
    write_csv_rows(
        output_dir / "validation_history.csv",
        validation_rows,
        ["epoch", "loss", "next_phase_loss", "phase_change_loss", "accuracy", "phase_change_accuracy", "samples"],
    )
    write_json(output_dir / "transformer_summary.json", summary_payload)

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
    summary_payload["prediction_export_complete"] = True
    summary_payload["inference_ms_total"] = elapsed_ms
    summary_payload["inference_us_per_sample"] = elapsed_ms * 1000.0 / max(1, len(labels))
    write_json(output_dir / "transformer_summary.json", summary_payload)
    print(f"Trained phase LM on {device}; wrote predictions for {len(labels)} windows.")


if __name__ == "__main__":
    main()
