"""Configuration helpers for the phase-ML command line tools."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from hpc_phase_analysis.constants import PROJECT_ROOT
from hpc_phase_analysis.io_utils import read_json

DEFAULT_PHASE_ML_CONFIG: dict[str, Any] = {
    "random_seed": 17,
    "dataset": {
        "input_csv": str(PROJECT_ROOT / "results" / "processed" / "merged_interval_dataset.csv"),
        "output_dir": str(PROJECT_ROOT / "results" / "phase_ml" / "dataset"),
        "sequence_length": 16,
        "stride": 1,
        "prediction_horizon": 20,
    },
    "labeling": {
        "method": "fgmm",
        "split_policy": "run_grouped",
        "holdout_workload": "",
        "pca_enabled": True,
        "pca_variance": 0.95,
        "k_min": 2,
        "k_max": 12,
        "max_iter": 80,
        "smoothing_window": 3,
        "min_phase_duration": 3,
    },
    "splits": {
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "baselines": {
        "decision_tree_max_depth": 5,
        "decision_tree_min_samples_leaf": 3,
        "logistic_epochs": 180,
        "logistic_lr": 0.08,
        "logistic_l2": 0.0005,
        "include_linear_svm": False,
    },
    "transformer": {
        "epochs": 8,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "ff_dim": 256,
        "dropout": 0.1,
        "rope_theta": 10000.0,
        "class_weight_power": 0.5,
        "change_loss_weight": 0.25,
        "amp_dtype": "bf16",
    },
    "student": {
        "decision_tree_max_depth": 6,
        "decision_tree_min_samples_leaf": 8,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_update(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(DEFAULT_PHASE_ML_CONFIG)
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return deep_update(DEFAULT_PHASE_ML_CONFIG, payload)
