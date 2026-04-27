"""Tests for the factorized family-wise phase LM pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows
from phase_family_ml.ablation import run_ablation
from phase_family_ml.config import deep_update, load_config
from phase_family_ml.families import FAMILY_COUNTERS, assert_no_forbidden_counter_columns
from phase_family_ml.labels import (
    _bucketize,
    _tertiles,
    build_counter_sequences,
    family_usage_scores,
    parsec_rows,
    states_for_scope,
    thresholds_for_family,
)
from phase_family_ml.students import LookupBackoffModel
from phase_family_ml.teacher import train_teachers_for_experiment
from phase_family_ml.transformer_model import build_family_transformer, require_torch
from phase_family_ml.collect import family_lm_events, family_lm_readiness


HEADER = [
    "timestamp_ms",
    "interval_duration_ms",
    "workload",
    "suite",
    "run_id",
    "threads",
    "cpu_or_core_id",
    "counter__instructions_retired",
    "counter__branch_instructions",
    "counter__branch_mispredictions",
    "counter__l1d_loads",
    "counter__l1d_stores",
    "counter__l2_misses",
    "counter__llc_references",
    "counter__llc_misses",
    "counter__offcore_demand_data_reads",
    "counter__memory_read_bandwidth",
    "counter__memory_write_bandwidth",
    "counter__total_memory_bandwidth",
    "counter__fp_arithmetic",
]


def write_fixture(path: Path) -> None:
    runs = [
        ("run_black_r1", "blackscholes", "0"),
        ("run_black_r2", "blackscholes", "0"),
        ("run_cann_r1", "canneal", "1"),
        ("run_cann_r2", "canneal", "1"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for run_id, workload, core in runs:
            for i in range(28):
                high = i >= 14
                inst = 100000 + i * 1000
                branch = 20000 if not high else 7000
                branch_miss = 200 if not high else 900
                l1_loads = 22000 if not high else 90000
                l1_stores = 10000 if not high else 50000
                l2_misses = 150 if not high else 2000
                llc_refs = 1300 if not high else 7500
                llc_misses = 120 if not high else 4000
                offcore = 250 if not high else 6000
                mem_read = 1.2 if not high else 7.0
                mem_write = 0.4 if not high else 3.1
                fp = 4000 if not high else 500
                writer.writerow(
                    {
                        "timestamp_ms": i * 10.0,
                        "interval_duration_ms": 10.0,
                        "workload": workload,
                        "suite": "parsec",
                        "run_id": run_id,
                        "threads": 2,
                        "cpu_or_core_id": core,
                        "counter__instructions_retired": inst,
                        "counter__branch_instructions": branch,
                        "counter__branch_mispredictions": branch_miss,
                        "counter__l1d_loads": l1_loads,
                        "counter__l1d_stores": l1_stores,
                        "counter__l2_misses": l2_misses,
                        "counter__llc_references": llc_refs,
                        "counter__llc_misses": llc_misses,
                        "counter__offcore_demand_data_reads": offcore,
                        "counter__memory_read_bandwidth": mem_read,
                        "counter__memory_write_bandwidth": mem_write,
                        "counter__total_memory_bandwidth": mem_read + mem_write,
                        "counter__fp_arithmetic": fp,
                    }
                )


class PhaseFamilyMLTests(unittest.TestCase):
    def test_family_mapping_and_forbidden_policy(self) -> None:
        self.assertIn("L2", FAMILY_COUNTERS)
        self.assertIn("core_fp", FAMILY_COUNTERS)
        assert_no_forbidden_counter_columns(["counter__instructions_retired", "counter__l2_misses"])
        with self.assertRaises(ValueError):
            assert_no_forbidden_counter_columns(["counter__cycles"])

    def test_family_profile_event_selection(self) -> None:
        alias_map = {
            "instructions_retired": {"supported": True, "collection_scope": "task_local", "analysis_confident": True, "selected_event": "inst_retired.any"},
            "l2_misses": {"supported": True, "collection_scope": "task_local", "analysis_confident": True, "selected_event": "l2_rqsts.miss"},
            "fp_arithmetic": {"supported": True, "collection_scope": "task_local", "analysis_confident": True, "selected_event": "fp_arith_inst_retired.scalar_double"},
        }
        events = family_lm_events(alias_map)
        self.assertIn("inst_retired.any", events)
        self.assertIn("l2_rqsts.miss", events)
        self.assertTrue(family_lm_readiness(alias_map)[0])

    def test_train_only_thresholds_and_bucketization(self) -> None:
        rows = [
            {"run_id": "r1", "workload": "w"},
            {"run_id": "r1", "workload": "w"},
            {"run_id": "r2", "workload": "w"},
            {"run_id": "r2", "workload": "w"},
        ]
        values = np.asarray([1.0, 2.0, 50.0, 60.0])
        split = {"r1": "train", "r2": "test"}
        thresholds = thresholds_for_family(values, rows, split, "global")
        states = states_for_scope(values, rows, "global", thresholds)
        # Train tertiles come from [1,2], so test values are high bucket.
        self.assertEqual(states.tolist(), [0, 2, 2, 2])

    def test_counter_sequence_file_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)
            summaries = build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
            )
            self.assertEqual(len(summaries), 1)
            sequence_path = root / "sequences" / "pooled_run_group" / "threshold_global" / "counter_sequence_L1.csv"
            rows = load_csv_rows(sequence_path)
            self.assertTrue(rows)
            first = rows[0]
            self.assertIn("family_state", first)
            self.assertIn("future_state_20", first)
            self.assertIn("will_change_within_horizon", first)
            self.assertIn("time_to_change", first)
            self.assertIn("split", first)
            self.assertIn("workload", first)
            self.assertIn("run_id", first)
            self.assertIn("core_id", first)
            counter_rows = load_csv_rows(sequence_path.parent / "counter_value_sequences" / "counter_sequence_counter__branch_mispredictions.csv")
            self.assertTrue(counter_rows)
            self.assertIn("counter_value", counter_rows[0])
            self.assertIn("future_counter_value_20", counter_rows[0])

    def test_sequences_use_ablation_selected_counters_for_teacher_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)

            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences_default",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
            )
            default_rows = load_csv_rows(root / "sequences_default" / "pooled_run_group" / "threshold_global" / "counter_sequence_core_fp.csv")
            default_states = [int(row.get("family_state", "-1") or -1) for row in default_rows]

            # Provide one selected counter per family exactly as the ablation output
            # schema does, then rebuild sequences from that selection.
            write_csv_rows(
                root / "ablation.csv",
                [
                    {"experiment": "pooled_run_group", "scope": "global", "family": "L1", "candidate_type": "singleton", "counter_set": "counter__l1d_loads", "selected": 1, "validation_score": 0.7},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "L2", "candidate_type": "singleton", "counter_set": "counter__l2_misses", "selected": 1, "validation_score": 0.7},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "LLC", "candidate_type": "singleton", "counter_set": "counter__llc_references", "selected": 1, "validation_score": 0.7},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "memory_offcore", "candidate_type": "singleton", "counter_set": "counter__total_memory_bandwidth", "selected": 1, "validation_score": 0.7},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "branch_control", "candidate_type": "singleton", "counter_set": "counter__branch_instructions", "selected": 1, "validation_score": 0.7},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "core_fp", "candidate_type": "singleton", "counter_set": "counter__fp_arithmetic", "selected": 1, "validation_score": 0.7},
                ],
            )

            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences_selected",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
                ablation_results=root / "ablation.csv",
            )
            selected_rows = load_csv_rows(root / "sequences_selected" / "pooled_run_group" / "threshold_global" / "counter_sequence_core_fp.csv")
            selected_states = [int(row.get("family_state", "-1") or -1) for row in selected_rows]

            self.assertEqual(len(default_states), len(selected_states))
            self.assertTrue(any(a != b for a, b in zip(default_states, selected_states)))

            summary_rows = load_csv_rows(root / "sequences_selected" / "pooled_run_group" / "counter_sequence_summary.csv")
            core_row = next(row for row in summary_rows if row.get("family", "") == "core_fp" and row.get("scope", "") == "global")
            self.assertEqual(core_row.get("sequence_source", ""), "ablation_selected")
            self.assertEqual(core_row.get("selected_counters", ""), "counter__fp_arithmetic")
            selected_first = selected_rows[0]
            self.assertEqual(selected_first.get("selected_counter", ""), "counter__fp_arithmetic")
            self.assertIn("counter_value", selected_first)
            self.assertIn("future_counter_value_20", selected_first)

    def test_sequences_prefer_global_one_per_family_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)
            write_csv_rows(
                root / "ablation.csv",
                [
                    {"experiment": "pooled_run_group", "scope": "global", "family": "L1", "candidate_type": "singleton", "counter_set": "counter__l1d_stores", "selected": 1, "validation_score": 0.8},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "LLC", "candidate_type": "singleton", "counter_set": "counter__llc_misses", "selected": 1, "validation_score": 0.8},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "memory_offcore", "candidate_type": "singleton", "counter_set": "counter__total_memory_bandwidth", "selected": 1, "validation_score": 0.8},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "branch_control", "candidate_type": "singleton", "counter_set": "counter__branch_mispredictions", "selected": 1, "validation_score": 0.8},
                    {"experiment": "pooled_run_group", "scope": "global", "family": "core_fp", "candidate_type": "singleton", "counter_set": "counter__fp_arithmetic", "selected": 1, "validation_score": 0.8},
                    {
                        "experiment": "pooled_run_group",
                        "scope": "global",
                        "family": "__global__",
                        "candidate_type": "one_per_family_exhaustive",
                        "counter_set": "counter__l1d_loads,counter__l2_misses,counter__llc_references,counter__offcore_demand_data_reads,counter__branch_instructions,counter__instructions_retired",
                        "family_counter_map": "L1:counter__l1d_loads;L2:counter__l2_misses;LLC:counter__llc_references;memory_offcore:counter__offcore_demand_data_reads;branch_control:counter__branch_instructions;core_fp:counter__instructions_retired",
                        "selected": 1,
                        "validation_score": 0.9,
                    },
                ],
            )
            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
                ablation_results=root / "ablation.csv",
            )
            summary_rows = load_csv_rows(root / "sequences" / "pooled_run_group" / "counter_sequence_summary.csv")
            core_row = next(row for row in summary_rows if row.get("family", "") == "core_fp" and row.get("scope", "") == "global")
            self.assertEqual(core_row.get("selected_counters", ""), "counter__instructions_retired")
            self.assertEqual(core_row.get("sequence_source", ""), "ablation_selected")

    def test_ablation_singleton_all_and_global_exhaustive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)
            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
            )
            rows = run_ablation(
                input_csv=merged,
                experiment_dir=root / "sequences" / "pooled_run_group",
                scope="global",
                horizon=20,
                output_csv=root / "ablation.csv",
                weights={"accuracy": 0.4, "phase_change_f1": 0.4, "high_usage_recall": 0.2},
                tree_max_depth=3,
                tree_min_samples_leaf=2,
                run_global_exhaustive=True,
            )
            self.assertTrue(any(row.get("candidate_type") == "singleton" for row in rows))
            self.assertTrue(any(row.get("candidate_type") == "all_counters" for row in rows))
            self.assertTrue(any(row.get("family") == "__global__" for row in rows))

    def test_transformer_output_shape(self) -> None:
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch not available")
        model = build_family_transformer(
            input_dim=4,
            horizon=20,
            num_classes=3,
            config={"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "ff_dim": 64, "dropout": 0.0, "rope_theta": 10000.0},
        )
        logits = model(torch.zeros(3, 8, 4))
        self.assertEqual(tuple(logits.shape), (3, 20, 3))

    def test_lookup_backoff(self) -> None:
        model = LookupBackoffModel([1, 3, 7])
        current = np.asarray([0, 0, 1, 1])
        prev = np.asarray([-1, 0, -1, 1])
        run_len = np.asarray([1, 2, 1, 4])
        context = np.asarray([10, 10, 11, 11])
        target = np.asarray([1, 1, 2, 2])
        model.fit(current, prev, run_len, context, target)
        pred = model.predict(np.asarray([0, 1, 2]), np.asarray([0, 1, -1]), np.asarray([2, 4, 1]), np.asarray([10, 11, 99]))
        self.assertEqual(pred.tolist()[0], 1)
        self.assertEqual(pred.tolist()[1], 2)
        self.assertIn(pred.tolist()[2], [1, 2])

    def test_smoke_teacher_training_artifacts(self) -> None:
        try:
            require_torch()
        except SystemExit:
            self.skipTest("PyTorch not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)
            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences",
                horizon=20,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
            )
            rows = train_teachers_for_experiment(
                experiment_dir=root / "sequences" / "pooled_run_group",
                scope="global",
                output_dir=root / "teacher",
                horizon=20,
                history_length=4,
                teacher_config={
                    "epochs": 1,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "ff_dim": 64,
                    "dropout": 0.0,
                    "rope_theta": 10000.0,
                    "context_modes": ["without_context"],
                },
                seed=3,
            )
            self.assertTrue(rows)
            self.assertTrue((root / "teacher" / "teacher_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
