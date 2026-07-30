"""Approximate hardware budget estimation for online phase predictors."""

from __future__ import annotations

import math
from typing import Any


def _complexity_and_recommendation(memory_bytes: float, operations: float, operator_type: str) -> tuple[str, str]:
    if operator_type in {"register", "table_lookup"} and memory_bytes <= 256 and operations <= 16:
        return "Very Low", "feasible in simple hardware"
    if operator_type in {"table_lookup", "comparisons", "integer_arithmetic"} and memory_bytes <= 2048 and operations <= 128:
        return "Low", "feasible in simple hardware"
    if operator_type in {"integer_arithmetic", "comparisons", "MACs"} and memory_bytes <= 8192 and operations <= 1024:
        return "Medium", "feasible in firmware/software runtime"
    if operator_type in {"MACs", "softmax"} and memory_bytes <= 65536 and operations <= 100000:
        return "High", "feasible in firmware/software runtime"
    if operator_type == "softmax":
        return "Very High", "offline upper bound only"
    return "Very High", "likely needs accelerator"


def estimate_hardware_budget(model: str, details: dict[str, Any]) -> dict[str, object]:
    """Return approximate deployment cost figures for one trained model."""

    parameter_count = int(details.get("parameter_count", 0) or 0)
    table_entries = int(details.get("table_entries", 0) or 0)
    feature_count = int(details.get("feature_count", 0) or 0)
    tree_internal = int(details.get("tree_internal_nodes", 0) or 0)
    tree_leaves = int(details.get("tree_leaves", 0) or 0)
    tree_storage = float(details.get("estimated_storage_bytes", 0.0) or 0.0)
    bucket_count = int(details.get("bucket_count", 0) or 0)
    kernel_count = int(details.get("kernel_count", 0) or 0)
    online_family_count = int(details.get("online_family_count", 0) or 0)
    history_length = int(details.get("effective_history_length", details.get("history_length", 0)) or 0)
    uses_clustered_counter_state = bool(details.get("uses_clustered_counter_state", False))

    memory_bytes = 0.0
    operations = 0.0
    operator_type = "integer_arithmetic"

    if model == "baseline_last_state":
        table_entries = max(table_entries, 1)
        memory_bytes = 1.0
        operations = 1.0
        operator_type = "register"
    elif model == "markov_phase_predictor":
        table_entries = max(table_entries, 9)
        memory_bytes = max(memory_bytes, table_entries * 2.0)
        operations = 3.0
        operator_type = "table_lookup"
    elif model == "rle_markov_phase_predictor":
        table_entries = max(table_entries, max(1, bucket_count) * 9)
        memory_bytes = max(memory_bytes, table_entries * 2.0)
        operations = 6.0
        operator_type = "table_lookup"
    elif model == "hsmm_duration_phase_predictor":
        table_entries = max(table_entries, 9 + max(1, bucket_count) * 6)
        memory_bytes = max(memory_bytes, table_entries * 2.0)
        operations = 12.0
        operator_type = "integer_arithmetic"
    elif model.startswith("online_") and "tree" in model:
        table_entries = max(table_entries, tree_internal + tree_leaves)
        memory_bytes = max(memory_bytes, tree_storage)
        operations = max(1.0, float(tree_internal) * 2.0)
        operator_type = "comparisons"
    elif model.startswith("rocket_phase_classifier"):
        parameter_count = max(parameter_count, kernel_count * max(1, feature_count))
        memory_bytes = max(memory_bytes, parameter_count * 4.0)
        operations = max(64.0, float(kernel_count) * max(1, feature_count) * 8.0)
        operator_type = "MACs"
    elif model.startswith("tcn_phase_classifier"):
        memory_bytes = max(memory_bytes, parameter_count * 4.0)
        operations = max(128.0, float(parameter_count) * 2.0)
        operator_type = "MACs"
    elif model.startswith("tiny_transformer_phase_classifier"):
        memory_bytes = max(memory_bytes, parameter_count * 4.0)
        operations = max(256.0, float(parameter_count) * 4.0)
        operator_type = "softmax"
    elif model == "graph_tcn_phase_classifier":
        memory_bytes = max(memory_bytes, parameter_count * 4.0)
        operations = max(256.0, float(parameter_count) * 3.0)
        operator_type = "MACs"
    else:
        if parameter_count > 0:
            memory_bytes = max(memory_bytes, parameter_count * 4.0)
            operations = max(8.0, float(parameter_count))
            operator_type = "integer_arithmetic"
        elif table_entries > 0:
            memory_bytes = max(memory_bytes, table_entries * 2.0)
            operations = max(4.0, float(table_entries))
            operator_type = "table_lookup"

    model_memory_bytes = memory_bytes
    history_storage_bytes = 0.0
    discretizer_storage_bytes = 0.0
    if uses_clustered_counter_state and online_family_count > 0:
        # Three clustered states require two bits per family and interval.
        history_storage_bytes = float(math.ceil(online_family_count * max(1, history_length) * 2 / 8))
        # Two 16-bit boundaries map each selected counter into one of three states.
        discretizer_storage_bytes = float(online_family_count * 2 * 2)
        memory_bytes += history_storage_bytes + discretizer_storage_bytes

    complexity, recommendation = _complexity_and_recommendation(memory_bytes, operations, operator_type)
    if bool(details.get("requires_oracle_current_phase", False)):
        recommendation = "oracle diagnostic only; requires the current offline-teacher phase"
    return {
        "stored_parameters_or_entries": int(max(parameter_count, table_entries)),
        "parameter_count": parameter_count,
        "table_entries": table_entries,
        "estimated_memory_bytes": float(memory_bytes),
        "model_storage_bytes": float(model_memory_bytes),
        "history_storage_bytes": history_storage_bytes,
        "discretizer_storage_bytes": discretizer_storage_bytes,
        "approx_operations_per_prediction": float(operations),
        "operator_type": operator_type,
        "hardware_complexity_category": complexity,
        "deployment_recommendation": recommendation,
    }
