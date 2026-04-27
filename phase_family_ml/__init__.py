"""Factorized family-wise phase language modeling pipeline.

This package is intentionally separate from ``phase_ml`` so we can evolve
family-specific modeling without changing the original pipeline modules.
"""

from __future__ import annotations

__all__ = [
    "ablation",
    "collect",
    "config",
    "evaluation",
    "families",
    "labels",
    "splits",
    "students",
    "teacher",
    "transformer_model",
]
