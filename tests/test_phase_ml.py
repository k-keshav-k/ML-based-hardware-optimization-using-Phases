"""Integration-style tests for the phase-ML data/model pipeline."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hpc_phase_analysis.collector import parse_perf_csv
from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows, write_json
from phase_ml.baselines import train_baselines
from phase_ml.clustering import enforce_min_duration
from phase_ml.collect_experiment_sets import chunked
from phase_ml.dataset import build_dataset
from phase_ml.evaluation import evaluate_predictions
from phase_ml.labeling import label_dataset
from phase_ml.train_student_tree import train_student
from phase_ml.transformer_model import build_model, require_torch
from scripts.merge_runs import merge_interval_rows
from scripts.run_workloads import events_for_profile, phase_ml_readiness


HEADER = [
    "timestamp_ms",
    "interval_duration_ms",
    "workload",
    "suite",
    "run_id",
    "threads",
    "cpu_or_core_id",
    "experiment_set",
    "rep",
    "concurrent_group_id",
    "process_index",
    "process_count",
    "co_running_workloads",
    "phase_label",
    "counter__instructions_retired",
    "counter__cycles",
    "counter__branch_instructions",
    "counter__branch_mispredictions",
    "counter__l1d_loads",
    "counter__l1d_stores",
    "counter__llc_references",
    "counter__llc_misses",
    "counter__offcore_demand_data_reads",
    "counter__memory_read_bandwidth",
    "counter__memory_write_bandwidth",
    "counter__total_memory_bandwidth",
    "counter__resource_stalls",
]


def write_fixture(path: Path) -> None:
    # Two workloads with two thread settings each so split/group logic sees
    # multiple run ids and execution modes.
    runs = [
        ("parsec_blackscholes_t1_r1", "blackscholes", 1, "0"),
        ("parsec_blackscholes_t4_r2", "blackscholes", 4, "0,1,2,3"),
        ("parsec_canneal_t1_r1", "canneal", 1, "0"),
        ("parsec_canneal_t4_r2", "canneal", 4, "0,1,2,3"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for run_id, workload, threads, cpu_text in runs:
            for index in range(18):
                phase = 0 if index < 9 else 1
                inst = 100000 + index * 1000
                branches = 30000 if phase == 0 else 7000
                branch_miss = 300 if phase == 0 else 900
                loads = 20000 if phase == 0 else 90000
                stores = 10000 if phase == 0 else 45000
                llc_refs = 1000 if phase == 0 else 7000
                llc_miss = 100 if phase == 0 else 3500
                offcore = 200 if phase == 0 else 5000
                imc_read = 1.5 if phase == 0 else 7.5
                imc_write = 0.5 if phase == 0 else 2.5
                writer.writerow(
                    {
                        "timestamp_ms": index * 10.0,
                        "interval_duration_ms": 10.0,
                        "workload": workload,
                        "suite": "parsec",
                        "run_id": run_id,
                        "threads": threads,
                        "cpu_or_core_id": cpu_text,
                        "experiment_set": "set1_single_process_multithread" if threads > 1 else "",
                        "rep": 1,
                        "concurrent_group_id": f"fixture_{run_id}",
                        "process_index": 0,
                        "process_count": 1,
                        "co_running_workloads": workload,
                        "phase_label": "",
                        "counter__instructions_retired": inst,
                        "counter__cycles": 999999,
                        "counter__branch_instructions": branches,
                        "counter__branch_mispredictions": branch_miss,
                        "counter__l1d_loads": loads,
                        "counter__l1d_stores": stores,
                        "counter__llc_references": llc_refs,
                        "counter__llc_misses": llc_miss,
                        "counter__offcore_demand_data_reads": offcore,
                        "counter__memory_read_bandwidth": imc_read,
                        "counter__memory_write_bandwidth": imc_write,
                        "counter__total_memory_bandwidth": imc_read + imc_write,
                        "counter__resource_stalls": 12345,
                    }
                )
        writer.writerow(
            {
                # Non-PARSEC row verifies dataset filtering by suite column.
                "timestamp_ms": 0,
                "interval_duration_ms": 10,
                "workload": "compute",
                "suite": "synthetic",
                "run_id": "synthetic_compute",
                "threads": 1,
                "cpu_or_core_id": "0",
                "experiment_set": "",
                "rep": "",
                "concurrent_group_id": "",
                "process_index": "",
                "process_count": "",
                "co_running_workloads": "",
                "phase_label": "",
                "counter__instructions_retired": 1,
                "counter__cycles": 1,
                "counter__branch_instructions": 1,
                "counter__branch_mispredictions": 1,
                "counter__l1d_loads": 1,
                "counter__l1d_stores": 1,
                "counter__llc_references": 1,
                "counter__llc_misses": 1,
                "counter__offcore_demand_data_reads": 1,
                "counter__memory_read_bandwidth": "",
                "counter__memory_write_bandwidth": "",
                "counter__total_memory_bandwidth": "",
                "counter__resource_stalls": 1,
            }
        )


class PhaseMLTests(unittest.TestCase):
    def test_dataset_features_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "merged.csv"
            write_fixture(input_csv)
            manifest = build_dataset(input_csv, root / "dataset", sequence_length=4, stride=2, prediction_horizon=2)
            self.assertEqual(manifest["interval_rows"], 72)
            self.assertGreater(manifest["windows"], 0)
            forbidden = ["cycle", "per_ms", "ipc", "cpi", "stall", "interval_duration"]
            for column in manifest["feature_columns"]:
                self.assertFalse(any(part in column for part in forbidden), column)
            self.assertIn("shared_imc_total_mib", manifest["feature_columns"])
            self.assertEqual(manifest["shared_context_scope"], "system_wide_uncore")
            rows = load_csv_rows(root / "dataset" / "interval_features.csv")
            units = {row["collection_unit_type"] for row in rows}
            self.assertIn("single_cpu_task_local", units)
            self.assertIn("affinity_group_task_local", units)
            self.assertIn("experiment_set", rows[0])
            self.assertEqual(rows[0]["shared_context_scope"], "system_wide_uncore")
            window_rows = load_csv_rows(root / "dataset" / "window_metadata.csv")
            self.assertIn("experiment_set", window_rows[0])
            arrays = np.load(root / "dataset" / "window_tensors.npz", allow_pickle=True)
            self.assertEqual(arrays["X"].shape[1], 4)

    def test_label_baseline_and_evaluate_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "merged.csv"
            write_fixture(input_csv)
            build_dataset(input_csv, root / "dataset", sequence_length=4, stride=1, prediction_horizon=2)
            label_summary = label_dataset(
                root / "dataset",
                root / "labels",
                {
                    "method": "fgmm",
                    "split_policy": "run_grouped",
                    "pca_enabled": False,
                    "k_min": 2,
                    "k_max": 3,
                    "max_iter": 15,
                    "smoothing_window": 1,
                    "min_phase_duration": 1,
                },
                {"train_fraction": 0.5, "val_fraction": 0.0},
                seed=3,
            )
            self.assertGreaterEqual(label_summary["phase_count"], 1)
            window_labels = load_csv_rows(root / "labels" / "window_labels.csv")
            self.assertTrue(window_labels)
            self.assertTrue({row["split"] for row in window_labels} <= {"train", "test", "val"})
            baseline_summary = train_baselines(
                root / "dataset",
                root / "labels",
                root / "baselines",
                {
                    "decision_tree_max_depth": 3,
                    "decision_tree_min_samples_leaf": 2,
                    "logistic_epochs": 8,
                    "logistic_lr": 0.05,
                    "logistic_l2": 0.001,
                    "include_linear_svm": True,
                },
            )
            self.assertIn("last_value", baseline_summary)
            self.assertIn("linear_svm", baseline_summary)
            teacher_rows = []
            for row in window_labels:
                teacher_rows.append(
                    {
                        "model": "transformer",
                        "window_id": row["window_id"],
                        "split": row["split"],
                        "workload": row["workload"],
                        "run_id": row["run_id"],
                        "phase_id": row["phase_id"],
                        "y_true_next_phase": row["next_phase_id"],
                        "y_pred_next_phase": row["next_phase_id"],
                        "y_true_phase_change": row["phase_change"],
                        "y_pred_phase_change": row["phase_change"],
                    }
                )
            write_csv_rows(root / "transformer" / "transformer_predictions.csv", teacher_rows)
            student_summary = train_student(
                root / "dataset",
                root / "labels",
                root / "transformer" / "transformer_predictions.csv",
                root / "student_tree",
                max_depth=2,
                min_samples_leaf=2,
            )
            self.assertIn("teacher_next_agreement", student_summary)
            self.assertTrue((root / "student_tree" / "student_decision_tree_predictions.csv").exists())
            evaluation = evaluate_predictions(sorted((root / "baselines").glob("*_predictions.csv")), root / "evaluation")
            self.assertIn("last_value", evaluation)
            self.assertTrue((root / "evaluation" / "model_comparison.csv").exists())

    def test_transition_filter(self) -> None:
        labels = np.asarray([1, 1, 2, 1, 1])
        filtered, changed = enforce_min_duration(labels, 2)
        self.assertEqual(filtered.tolist(), [1, 1, 1, 1, 1])
        self.assertTrue(changed[2])

    def test_phase_ml_event_profile_avoids_cycles(self) -> None:
        alias_map = {
            "instructions_retired": {"supported": True, "analysis_confident": True, "collection_scope": "task_local", "selected_event": "inst_retired.any"},
            "cycles": {"supported": True, "analysis_confident": True, "collection_scope": "task_local", "selected_event": "cycles"},
            "branch_mispredictions": {"supported": True, "analysis_confident": True, "collection_scope": "task_local", "selected_event": "branch-misses"},
            "resource_stalls": {"supported": True, "analysis_confident": True, "collection_scope": "task_local", "selected_event": "resource_stalls.any"},
        }
        events = events_for_profile(alias_map, "phase_ml")
        self.assertIn("inst_retired.any", events)
        self.assertIn("branch-misses", events)
        self.assertNotIn("cycles", events)
        self.assertNotIn("resource_stalls.any", events)
        self.assertTrue(phase_ml_readiness(alias_map)[0])

    def test_phase_language_model_forward_shapes(self) -> None:
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch is not installed")
        model = build_model(
            input_dim=5,
            phase_count=3,
            config={
                "hidden_dim": 16,
                "num_layers": 2,
                "num_heads": 4,
                "ff_dim": 32,
                "dropout": 0.0,
                "rope_theta": 10000.0,
            },
        )
        logits, change_logits = model(torch.zeros(2, 7, 5))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(change_logits.shape), (2,))

    def test_transformer_dependency_is_guarded(self) -> None:
        try:
            torch, _ = require_torch()
        except SystemExit:
            return
        model = __import__("phase_ml.transformer_model", fromlist=["build_model"]).build_model(3, 2, {"hidden_dim": 8, "num_heads": 2, "num_layers": 1, "ff_dim": 16, "dropout": 0.0})
        x = torch.zeros((2, 4, 3), dtype=torch.float32)
        next_logits, change_logits = model(x)
        self.assertEqual(tuple(next_logits.shape), (2, 2))
        self.assertEqual(tuple(change_logits.shape), (2,))

    def test_experiment_set_grouping_keeps_leftover_workload(self) -> None:
        groups = chunked(["a", "b", "c", "d", "e"], 2)
        self.assertEqual(groups, [["a", "b"], ["c", "d"], ["e", "a"]])

    def test_perf_parser_accepts_system_wide_cpu_rows(self) -> None:
        text = "0.100,CPU3,1234,,inst_retired.any [cpu],1000000,100.00,,\n"
        rows = parse_perf_csv(text, interval_mode=True)
        self.assertEqual(rows[0]["cpu_or_core_id"], "3")
        self.assertEqual(rows[0]["event_name"], "inst_retired.any")
        self.assertEqual(rows[0]["event_name_normalized"], "inst_retired.any")

    def test_merge_aggregates_smt_siblings_for_physical_core_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(
                run_dir / "metadata.json",
                {
                    "run_id": "parsec_fixture",
                    "suite": "parsec",
                    "workload": "blackscholes",
                    "threads": 2,
                    "cpu_or_core_id": "0,10",
                    "collection_scope": "system_wide_physical_core",
                    "core_collection_scope": "system_wide_physical_core",
                    "cpu_topology": {
                        "0": {"physical_core_id": "0:0"},
                        "10": {"physical_core_id": "0:0"},
                    },
                    "uncore_events": [],
                    "alias_map": {
                        "instructions_retired": {
                            "collection_scope": "task_local",
                            "selected_event": "inst_retired.any",
                            "selected_via": "inst_retired.any",
                        }
                    },
                },
            )
            write_csv_rows(
                run_dir / "interval_perf.csv",
                [
                    {
                        "timestamp_ms": 10.0,
                        "cpu_or_core_id": "0",
                        "value": 100,
                        "event_name": "inst_retired.any",
                    },
                    {
                        "timestamp_ms": 10.0,
                        "cpu_or_core_id": "10",
                        "value": 40,
                        "event_name": "inst_retired.any",
                    },
                ],
            )
            rows, attached = merge_interval_rows(run_dir)
            self.assertEqual(attached, 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cpu_or_core_id"], "0:0")
            self.assertEqual(rows[0]["physical_core_id"], "0:0")
            self.assertEqual(rows[0]["collection_scope"], "system_wide_physical_core")
            self.assertEqual(rows[0]["counter__instructions_retired"], 140.0)


if __name__ == "__main__":
    unittest.main()
