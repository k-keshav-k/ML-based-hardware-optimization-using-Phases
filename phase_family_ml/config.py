"""Configuration helpers for the factorized family-wise phase LM pipeline.

The defaults are tuned to keep local development runs lightweight while still
covering all required stages and outputs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from hpc_phase_analysis.constants import PROJECT_ROOT
from hpc_phase_analysis.io_utils import read_json


DEFAULT_FAMILY_ML_CONFIG: dict[str, Any] = {
    "random_seed": 17,
    "dataset": {
        "input_csv": str(PROJECT_ROOT / "results" / "processed" / "merged_interval_dataset.csv"),
        "output_dir": str(PROJECT_ROOT / "results" / "phase_family_ml"),
        "horizon": 1,
        "history_length": 16,
    },
    "families": {
        "threshold_mode": "global",
        "names": ["L1", "L2", "LLC", "memory_offcore", "branch_control", "core_fp"],
    },
    "splits": {
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
    },
    "ablation": {
        "score_weights": {
            "accuracy": 0.7,
            "high_usage_recall": 0.3,
        },
        "tree_max_depth": 5,
        "tree_min_samples_leaf": 3,
        "global_exhaustive_one_per_family": True,
    },
    "teacher": {
        "epochs": 15,
        "batch_size": 1024,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "hidden_dim": 192,
        "num_layers": 5,
        "num_heads": 6,
        "ff_dim": 768,
        "dropout": 0.15,
        "rope_theta": 10000.0,
        "context_modes": ["without_context", "with_context"],
        "early_stopping_patience": 3,
        "warmup_fraction": 0.05,
        "scheduler": "cosine",
        "class_weight_power": 0.5,
        "transition_loss_weight": 1.5,
        "high_usage_loss_weight": 0.5,
    },
    "student": {
        "blend_alpha": 0.25,
        "decision_tree_max_depth": 6,
        "decision_tree_min_samples_leaf": 8,
        "run_length_buckets": [1, 3, 7, 15],
        "synthetic_examples_per_family": 20000,
        "synthetic_mutation_rate": 0.05,
    },
    "phase_detector": {
        "history_length": 20,
        "prediction_horizon": 1,
        "decision_tree_max_depth": 6,
        "decision_tree_min_samples_leaf": 8,
    },
    "experiments": {
        "modes": ["config_group_holdout"],
        "default_mode": "config_group_holdout",
    },
    "runtime": {
        "profile": "quick",
        "quick": {
            "teacher_epochs": 2,
        },
        "full": {
            "teacher_epochs": 15,
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries recursively while preserving unknown fields."""

    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_update(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a JSON override file on top of the package defaults."""

    if not path:
        return copy.deepcopy(DEFAULT_FAMILY_ML_CONFIG)
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return deep_update(DEFAULT_FAMILY_ML_CONFIG, payload)


def apply_runtime_profile(config: dict[str, Any], full: bool = False) -> dict[str, Any]:
    """Apply quick/full runtime overrides and return a new config object."""

    output = copy.deepcopy(config)
    runtime = dict(output.get("runtime", {}))
    profile = "full" if full else str(runtime.get("profile", "quick"))
    profile_payload = runtime.get(profile, {})
    if isinstance(profile_payload, dict):
        teacher_epochs = profile_payload.get("teacher_epochs")
        if teacher_epochs is not None:
            output.setdefault("teacher", {})["epochs"] = int(teacher_epochs)
    output.setdefault("runtime", {})["profile"] = profile
    return output
