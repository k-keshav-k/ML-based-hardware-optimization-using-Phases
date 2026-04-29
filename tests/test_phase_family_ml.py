"""Tests for the factorized family-wise phase LM pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, read_json, write_csv_rows
from phase_family_ml.ablation import run_ablation
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
from phase_family_ml.students import HistoryLookupModel, LookupBackoffModel, train_students_for_experiment
from phase_family_ml.teacher import _build_examples, train_teachers_for_experiment
from phase_family_ml.train_phase_detector import (
    run_phase_detector_depth_sweep_for_experiment,
    train_phase_detector_for_experiment,
)
from phase_family_ml.transformer_model import build_family_transformer, require_torch
from phase_family_ml.collect import family_lm_events, family_lm_readiness
from phase_family_ml.splits import build_experiment_splits


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

    def test_config_group_holdout_keeps_reps_together(self) -> None:
        rows = []
        for threads in ["2", "4", "8"]:
            for rep in range(1, 5):
                run_id = f"w_t{threads}_r{rep}"
                rows.append(
                    {
                        "run_id": run_id,
                        "workload": "w",
                        "threads": threads,
                        "experiment_set": "set1_single_process_multithread",
                        "requested_input_size": "simsmall",
                        "process_index": "0",
                        "process_count": "1",
                        "co_running_workloads": "w",
                        "core_collection_scope": "system_wide_physical_core",
                    }
                )
        [split] = build_experiment_splits(rows, "config_group_holdout", 0.67, 0.0, 7)
        by_threads: dict[str, set[str]] = {}
        for row in rows:
            by_threads.setdefault(row["threads"], set()).add(split.split_by_run[row["run_id"]])
        self.assertTrue(all(len(values) == 1 for values in by_threads.values()))

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
            self.assertFalse(sequence_path.exists())
            manifest = read_json(root / "sequences" / "pooled_run_group" / "counter_sequence_manifest.json")
            self.assertFalse(manifest.get("family_sequences_written"))
            counter_rows = load_csv_rows(sequence_path.parent / "counter_value_sequences" / "counter_sequence_counter__branch_mispredictions.csv")
            self.assertTrue(counter_rows)
            first = counter_rows[0]
            self.assertIn("family_state", first)
            self.assertIn("future_state_20", first)
            self.assertIn("will_change_within_horizon", first)
            self.assertIn("time_to_change", first)
            self.assertIn("split", first)
            self.assertIn("workload", first)
            self.assertIn("run_id", first)
            self.assertIn("core_id", first)
            self.assertIn("counter_value", first)
            self.assertIn("future_counter_value_20", first)

            write_csv_rows(
                root / "ablation.csv",
                [
                    {"experiment": "pooled_run_group", "scope": "global", "family": "L1", "candidate_type": "singleton", "counter_set": "counter__l1d_loads", "selected": 1, "validation_score": 0.7},
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
            self.assertEqual(rows, load_csv_rows(sequence_path.parent / "counter_value_sequences" / "counter_sequence_counter__l1d_loads.csv"))

    def test_sequences_use_ablation_selected_counters_for_teacher_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)

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
            self.assertTrue(selected_rows)
            raw_rows = load_csv_rows(root / "sequences_selected" / "pooled_run_group" / "threshold_global" / "counter_value_sequences" / "counter_sequence_counter__fp_arithmetic.csv")
            self.assertEqual(selected_rows, raw_rows)

            summary_rows = load_csv_rows(root / "sequences_selected" / "pooled_run_group" / "counter_sequence_summary.csv")
            core_row = next(row for row in summary_rows if row.get("family", "") == "core_fp" and row.get("scope", "") == "global")
            self.assertEqual(core_row.get("sequence_source", ""), "ablation_selected")
            self.assertEqual(core_row.get("selected_counters", ""), "counter__fp_arithmetic")
            selected_first = selected_rows[0]
            self.assertEqual(selected_first.get("counter_name", ""), "counter__fp_arithmetic")
            self.assertEqual(selected_first.get("phase_label_source", ""), "train_split_kmeans_full_safe_counters")
            self.assertIn("counter_value", selected_first)
            self.assertIn("future_counter_value_20", selected_first)

            summary_rows = train_phase_detector_for_experiment(
                experiment_dir=root / "sequences_selected" / "pooled_run_group",
                scope="global",
                output_dir=root / "phase_detector",
                horizon=20,
                history_length=4,
                prediction_horizon=5,
                tree_max_depth=3,
                tree_min_leaf=2,
            )
            summary = summary_rows[0]
            self.assertEqual(summary.get("phase_label_source"), "train_split_kmeans_full_safe_counters")
            self.assertEqual(summary.get("history_length"), 4)
            self.assertEqual(summary.get("prediction_horizon"), 5)
            self.assertIn("top1_accuracy", summary)
            self.assertIn("stable_case_accuracy", summary)
            self.assertIn("transition_case_accuracy", summary)
            models = {row.get("model") for row in summary_rows}
            self.assertIn("online_phase_history_tree", models)
            self.assertIn("baseline_last_state", models)
            self.assertIn("baseline_majority", models)
            self.assertIn("baseline_state_conditioned_majority", models)
            self.assertTrue((root / "phase_detector" / "phase_detector_summary.csv").exists())

            sweep_rows = run_phase_detector_depth_sweep_for_experiment(
                experiment_dir=root / "sequences_selected" / "pooled_run_group",
                scope="global",
                output_dir=root / "phase_detector",
                horizon=20,
                history_length=4,
                prediction_horizon=5,
                tree_depths=[1, 2, 3],
                tree_min_leaf=2,
            )
            self.assertEqual(len(sweep_rows), 3)
            self.assertEqual([int(row["tree_max_depth"]) for row in sweep_rows], [1, 2, 3])
            self.assertTrue(all("mean_accuracy" in row for row in sweep_rows))
            self.assertTrue((root / "phase_detector" / "phase_detector_depth_sweep_summary.csv").exists())

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

    def test_ablation_singleton_and_global_exhaustive(self) -> None:
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
                weights={"accuracy": 0.7, "high_usage_recall": 0.3},
                tree_max_depth=3,
                tree_min_samples_leaf=2,
                run_global_exhaustive=True,
            )
            self.assertTrue(any(row.get("candidate_type") == "singleton" for row in rows))
            self.assertFalse(any(row.get("candidate_type") == "all_counters" for row in rows))
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

    def test_teacher_examples_use_state_history_inputs(self) -> None:
        current = np.asarray(
            [
                [0, -1],
                [1, 0],
                [2, 1],
                [1, 2],
                [0, -1],
            ],
            dtype=int,
        )
        future = np.asarray([[[1], [0]], [[2], [1]], [[1], [2]], [[0], [1]], [[2], [0]]], dtype=int)
        split = np.asarray(["train", "train", "train", "val", "val"])
        metadata_rows = [
            {"run_id": "r1", "core_id": "0", "timestamp_ms": str(i * 10)}
            for i in range(current.shape[0])
        ]

        x, y, *_ = _build_examples(
            family_index=0,
            current=current,
            future=future,
            split=split,
            metadata_rows=metadata_rows,
            history_length=3,
            context_mode="without_context",
        )
        self.assertEqual(tuple(x.shape), (3, 3, 3))
        np.testing.assert_array_equal(x[0], np.eye(3))
        self.assertEqual(y[0].tolist(), [1])

        x_context, *_ = _build_examples(
            family_index=0,
            current=current,
            future=future,
            split=split,
            metadata_rows=metadata_rows,
            history_length=3,
            context_mode="with_context",
        )
        self.assertEqual(tuple(x_context.shape), (3, 3, 6))
        np.testing.assert_array_equal(x_context[0, 0, :3], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(x_context[0, 0, 3:], [0.0, 0.0, 0.0])

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

    def test_history_lookup_suffix_backoff(self) -> None:
        history = np.asarray(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[0, 0, 1], [0, 1, 0]],
                [[1, 0, 0], [0, 0, 1]],
            ],
            dtype=float,
        )
        target = np.asarray([1, 1, 2], dtype=int)
        model = HistoryLookupModel().fit(history, target)
        query = np.asarray(
            [
                [[0, 1, 0], [0, 1, 0]],
                [[0, 1, 0], [0, 0, 1]],
            ],
            dtype=float,
        )
        self.assertEqual(model.predict(query).tolist(), [1, 2])

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

    def test_student_training_writes_history_distillation_models(self) -> None:
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged = root / "merged.csv"
            write_fixture(merged)
            ablation_rows = [
                {"experiment": "pooled_run_group", "scope": "global", "family": "L1", "candidate_type": "singleton", "counter_set": "counter__l1d_loads", "selected": 1, "validation_score": 0.7},
                {"experiment": "pooled_run_group", "scope": "global", "family": "L2", "candidate_type": "singleton", "counter_set": "counter__l2_misses", "selected": 1, "validation_score": 0.7},
                {"experiment": "pooled_run_group", "scope": "global", "family": "LLC", "candidate_type": "singleton", "counter_set": "counter__llc_references", "selected": 1, "validation_score": 0.7},
                {"experiment": "pooled_run_group", "scope": "global", "family": "memory_offcore", "candidate_type": "singleton", "counter_set": "counter__total_memory_bandwidth", "selected": 1, "validation_score": 0.7},
                {"experiment": "pooled_run_group", "scope": "global", "family": "branch_control", "candidate_type": "singleton", "counter_set": "counter__branch_instructions", "selected": 1, "validation_score": 0.7},
                {"experiment": "pooled_run_group", "scope": "global", "family": "core_fp", "candidate_type": "singleton", "counter_set": "counter__fp_arithmetic", "selected": 1, "validation_score": 0.7},
            ]
            write_csv_rows(root / "ablation.csv", ablation_rows)
            build_counter_sequences(
                input_csv=merged,
                output_root=root / "sequences",
                horizon=1,
                threshold_mode="global",
                experiment_mode="pooled_run_group",
                train_fraction=0.7,
                val_fraction=0.15,
                seed=7,
                ablation_results=root / "ablation.csv",
            )

            teacher_rows = []
            scope_dir = root / "sequences" / "pooled_run_group" / "threshold_global"
            for family in FAMILY_COUNTERS:
                for row in load_csv_rows(scope_dir / f"counter_sequence_{family}.csv"):
                    target = int(row.get("future_state_1", "-1") or -1)
                    if target < 0:
                        continue
                    item = {
                        "family": family,
                        "scope": "global",
                        "context_mode": "with_context",
                        "split": row.get("split", ""),
                        "workload": row.get("workload", ""),
                        "run_id": row.get("run_id", ""),
                        "core_id": row.get("core_id", ""),
                        "row_index": row.get("row_index", ""),
                        "family_state": row.get("family_state", ""),
                        "y_true_future_state_1": target,
                        "y_pred_future_state_1": target,
                    }
                    for cls in range(3):
                        item[f"p_future_state_1_class_{cls}"] = 1.0 if cls == target else 0.0
                    teacher_rows.append(item)
            write_csv_rows(root / "teacher_predictions.csv", teacher_rows)
            checkpoint_dir = root / "teacher_checkpoints"
            checkpoint_dir.mkdir()
            teacher_cfg = {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "ff_dim": 64, "dropout": 0.0, "rope_theta": 10000.0}
            input_dim = len(FAMILY_COUNTERS) * 3
            for family in FAMILY_COUNTERS:
                model = build_family_transformer(input_dim=input_dim, horizon=1, num_classes=3, config=teacher_cfg)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "horizon": 1,
                        "num_classes": 3,
                        "config": teacher_cfg,
                        "context_mode": "with_context",
                        "family": family,
                        "history_length": 4,
                    },
                    checkpoint_dir / f"{family}.pt",
                )

            rows = train_students_for_experiment(
                experiment_dir=root / "sequences" / "pooled_run_group",
                scope="global",
                teacher_predictions_path=root / "teacher_predictions.csv",
                output_dir=root / "students",
                horizon=1,
                history_length=4,
                blend_alpha=0.25,
                tree_max_depth=3,
                tree_min_leaf=2,
                run_length_buckets=[1, 3, 7],
                synthetic_examples_per_family=8,
                synthetic_mutation_rate=0.05,
                seed=7,
            )
            models = {row.get("model") for row in rows}
            self.assertIn("decision_tree_distilled_history", models)
            self.assertIn("lookup_distilled_history", models)
            self.assertIn("decision_tree_scratch_history", models)
            self.assertIn("synthetic_distilled_history_tree", models)
            pred_models = {row.get("model") for row in load_csv_rows(root / "students" / "student_predictions.csv")}
            self.assertIn("decision_tree_scratch_history", pred_models)
            self.assertIn("synthetic_distilled_history_tree", pred_models)


if __name__ == "__main__":
    unittest.main()
