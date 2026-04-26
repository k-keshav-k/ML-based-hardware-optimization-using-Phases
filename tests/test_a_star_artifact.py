"""Tests for analysis-level detector, replay, and artifact validation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.detector import CentroidDetector, apply_persistence, ewma_update, quantize_fixed_point
from analysis.trace_policy_replay import replay_policy
from analysis.validate_artifact import validate
from hpc_phase_analysis.io_utils import write_csv_rows, write_json


class AStarArtifactTests(unittest.TestCase):
    def test_ewma_quantization_and_persistence(self) -> None:
        prev = np.asarray([0.0, 1.0])
        cur = np.asarray([1.0, 3.0])
        self.assertTrue(np.allclose(ewma_update(prev, cur, 0.25), [0.25, 1.5]))
        quantized = quantize_fixed_point(np.asarray([-5.0, 0.0, 5.0]), 8)
        self.assertGreaterEqual(float(quantized.min()), -4.0)
        self.assertLessEqual(float(quantized.max()), 4.0)
        labels = apply_persistence(np.asarray([1, 2, 1, 2, 2, 2]), 3)
        self.assertEqual(labels.tolist(), [1, 1, 1, 1, 1, 2])

    def test_centroid_detector(self) -> None:
        x = np.asarray([[0.0], [0.1], [5.0], [5.1]])
        y = np.asarray([0, 0, 1, 1])
        detector = CentroidDetector.fit(x, y)
        self.assertEqual(detector.predict(np.asarray([[0.2], [4.9]])).tolist(), [0, 1])

    def test_trace_policy_conflict_metrics(self) -> None:
        rows = [
            {"window_id": "1", "run_id": "a", "cpu_or_core_id": "0", "concurrent_group_id": "g", "end_timestamp_ms": "10", "process_count": "2"},
            {"window_id": "2", "run_id": "b", "cpu_or_core_id": "1", "concurrent_group_id": "g", "end_timestamp_ms": "10", "process_count": "2"},
            {"window_id": "3", "run_id": "a", "cpu_or_core_id": "0", "concurrent_group_id": "g", "end_timestamp_ms": "20", "process_count": "2"},
            {"window_id": "4", "run_id": "b", "cpu_or_core_id": "1", "concurrent_group_id": "g", "end_timestamp_ms": "20", "process_count": "2"},
        ]
        metrics = replay_policy(rows, {1: "memory", 2: "memory", 3: "branch", 4: "memory"})
        self.assertEqual(metrics["pair_count"], 2)
        self.assertEqual(metrics["conflict_count"], 1)
        self.assertEqual(metrics["migration_count"], 1)

    def test_validation_catches_clean_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            dataset = root / "dataset"
            labels = root / "labels"
            evaluation = root / "evaluation"
            processed.mkdir()
            dataset.mkdir()
            labels.mkdir()
            evaluation.mkdir()
            write_json(processed / "merge_summary.json", {"manifest_run_count": 1, "merged_run_count": 1})
            write_json(dataset / "feature_manifest.json", {"interval_rows": 1, "windows": 1, "feature_columns": ["branch_miss_rate"]})
            write_csv_rows(dataset / "interval_features.csv", [{"interval_index": 0, "run_id": "r1", "branch_miss_rate": 0.1}])
            write_csv_rows(dataset / "window_metadata.csv", [{"window_id": 0, "run_id": "r1"}])
            np.savez_compressed(dataset / "window_tensors.npz", X=np.zeros((1, 1, 1)), feature_columns=np.asarray(["branch_miss_rate"]))
            write_csv_rows(labels / "window_labels.csv", [{"window_id": 0, "run_id": "r1", "split": "train"}])
            write_csv_rows(evaluation / "model_comparison.csv", [{"model": "m"}])
            payload = validate(processed, dataset, labels, evaluation, root / "validation.json")
            self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
