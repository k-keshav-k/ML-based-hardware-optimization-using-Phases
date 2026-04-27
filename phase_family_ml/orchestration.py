"""Helpers for iterating experiment/scope artifacts in the new pipeline."""

from __future__ import annotations

from pathlib import Path


def experiment_dirs(label_root: Path) -> list[Path]:
    """Return experiment directories that contain a label manifest."""

    if not label_root.exists():
        return []
    return sorted(path for path in label_root.iterdir() if path.is_dir() and (path / "family_label_manifest.json").exists())


def scopes_for_experiment(experiment_dir: Path) -> list[str]:
    """Return available threshold scopes for one experiment directory."""

    scopes = []
    if (experiment_dir / "threshold_global").exists():
        scopes.append("global")
    if (experiment_dir / "threshold_per_workload").exists():
        scopes.append("per_workload")
    return scopes
