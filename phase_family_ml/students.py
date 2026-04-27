"""Student distillation models for family-wise phase prediction.

Students include:
- decision-tree student (distilled with blended teacher/true targets)
- lookup/RLE-style backoff student for cheap table inference
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows, write_json
from phase_ml.baselines import DecisionTree

from .data import load_scope_family_data, states_matrix
from .metrics import classification_metrics


class LookupBackoffModel:
    """Lookup table with progressive backoff for unseen keys."""

    def __init__(self, buckets: list[int]) -> None:
        self.buckets = list(sorted(set(int(item) for item in buckets)))
        self.full: dict[tuple[int, int, int, int], int] = {}
        self.mid: dict[tuple[int, int, int], int] = {}
        self.simple: dict[tuple[int, int], int] = {}
        self.by_current: dict[int, int] = {}
        self.fallback = 1

    def _bucket(self, run_length: int) -> int:
        for bucket in self.buckets:
            if run_length <= bucket:
                return bucket
        return self.buckets[-1] if self.buckets else 15

    def fit(self, current: np.ndarray, prev: np.ndarray, run_length: np.ndarray, context_hash: np.ndarray, target: np.ndarray) -> "LookupBackoffModel":
        counter_full: dict[tuple[int, int, int, int], Counter[int]] = defaultdict(Counter)
        counter_mid: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
        counter_simple: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        counter_current: dict[int, Counter[int]] = defaultdict(Counter)
        for cur, prv, rl, ctx, tgt in zip(current, prev, run_length, context_hash, target):
            key_full = (int(cur), int(prv), self._bucket(int(rl)), int(ctx))
            key_mid = (int(cur), self._bucket(int(rl)), int(ctx))
            key_simple = (int(cur), self._bucket(int(rl)))
            counter_full[key_full][int(tgt)] += 1
            counter_mid[key_mid][int(tgt)] += 1
            counter_simple[key_simple][int(tgt)] += 1
            counter_current[int(cur)][int(tgt)] += 1
        self.full = {key: cnt.most_common(1)[0][0] for key, cnt in counter_full.items()}
        self.mid = {key: cnt.most_common(1)[0][0] for key, cnt in counter_mid.items()}
        self.simple = {key: cnt.most_common(1)[0][0] for key, cnt in counter_simple.items()}
        self.by_current = {key: cnt.most_common(1)[0][0] for key, cnt in counter_current.items()}
        self.fallback = Counter(target.tolist()).most_common(1)[0][0] if target.size else 1
        return self

    def predict(self, current: np.ndarray, prev: np.ndarray, run_length: np.ndarray, context_hash: np.ndarray) -> np.ndarray:
        output = np.full(current.shape[0], self.fallback, dtype=int)
        for index, (cur, prv, rl, ctx) in enumerate(zip(current, prev, run_length, context_hash)):
            key_full = (int(cur), int(prv), self._bucket(int(rl)), int(ctx))
            key_mid = (int(cur), self._bucket(int(rl)), int(ctx))
            key_simple = (int(cur), self._bucket(int(rl)))
            if key_full in self.full:
                output[index] = self.full[key_full]
            elif key_mid in self.mid:
                output[index] = self.mid[key_mid]
            elif key_simple in self.simple:
                output[index] = self.simple[key_simple]
            elif int(cur) in self.by_current:
                output[index] = self.by_current[int(cur)]
            else:
                output[index] = self.fallback
        return output


def _group_indices(metadata_rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata_rows):
        grouped[(str(row.get("run_id", "")), str(row.get("core_id", "")))].append(index)
    for key in grouped:
        grouped[key].sort(key=lambda i: float(metadata_rows[i].get("timestamp_ms", "0") or 0.0))
    return list(grouped.values())


def _temporal_features(current: np.ndarray, metadata_rows: list[dict[str, str]], family_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return current/prev/run-length/context-hash features for one family."""

    n = current.shape[0]
    cur = current[:, family_index].astype(int)
    prev = np.full(n, -1, dtype=int)
    run_len = np.ones(n, dtype=int)
    for stream in _group_indices(metadata_rows):
        last = -1
        length = 0
        for idx in stream:
            state = int(cur[idx])
            prev[idx] = last
            if state == last:
                length += 1
            else:
                length = 1
            run_len[idx] = length
            last = state
    context_cols = [index for index in range(current.shape[1]) if index != family_index]
    if not context_cols:
        context_hash = np.zeros(n, dtype=int)
    else:
        # Compress context states into a compact integer key.
        context_hash = np.zeros(n, dtype=int)
        for row_index in range(n):
            key = tuple(int(current[row_index, col]) for col in context_cols)
            context_hash[row_index] = hash(key) % 1000003
    return cur, prev, run_len, context_hash


def _eval_mask(split: np.ndarray) -> np.ndarray:
    val = split == "val"
    if np.any(val):
        return val
    test = split == "test"
    if np.any(test):
        return test
    return split != "train"


def _teacher_tables(path: Path, horizon: int) -> tuple[dict[tuple[str, int], dict[int, int]], dict[tuple[str, int], dict[int, np.ndarray]]]:
    """Load teacher hard predictions and optional probabilities keyed by row index."""

    rows = load_csv_rows(path)
    hard: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    soft: dict[tuple[str, int], dict[int, np.ndarray]] = defaultdict(dict)
    for row in rows:
        family = str(row.get("family", ""))
        row_index = int(row.get("row_index", "0") or 0)
        for step in range(1, horizon + 1):
            hard[(family, row_index)][step] = int(row.get(f"y_pred_future_state_{step}", "0") or 0)
            probs = []
            for cls in range(3):
                key = f"p_future_state_{step}_class_{cls}"
                value = row.get(key, "")
                if value == "":
                    probs = []
                    break
                probs.append(float(value))
            if probs:
                soft[(family, row_index)][step] = np.asarray(probs, dtype=float)
    return hard, soft


def train_students_for_experiment(
    experiment_dir: Path,
    scope: str,
    teacher_predictions_path: Path,
    output_dir: Path,
    horizon: int,
    blend_alpha: float,
    tree_max_depth: int,
    tree_min_leaf: int,
    run_length_buckets: list[int],
) -> list[dict[str, object]]:
    """Train decision-tree and lookup students for each family."""

    output_dir.mkdir(parents=True, exist_ok=True)
    family_data = load_scope_family_data(experiment_dir, scope, horizon)
    families, current, future, split, metadata_rows = states_matrix(family_data, horizon)
    if not families:
        return []
    hard_teacher, soft_teacher = _teacher_tables(teacher_predictions_path, horizon)
    eval_mask = _eval_mask(split)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for family_index, family in enumerate(families):
        row_indices = np.arange(current.shape[0], dtype=int)
        cur, prev, run_len, context_hash = _temporal_features(current, metadata_rows, family_index)
        # Core cheap student features.
        features = np.c_[cur.astype(float), prev.astype(float), np.log1p(run_len.astype(float)), context_hash.astype(float)]
        train_mask = split == "train"
        if not np.any(train_mask):
            train_mask = np.ones(split.shape[0], dtype=bool)

        tree_preds = np.full((current.shape[0], horizon), -1, dtype=int)
        lookup_preds = np.full((current.shape[0], horizon), -1, dtype=int)
        for step in range(1, horizon + 1):
            true_target = future[:, family_index, step - 1].astype(int)
            valid = true_target >= 0
            if not np.any(valid):
                continue
            pseudo = true_target.copy()
            for i, row_index in enumerate(row_indices):
                key = (family, int(row_index))
                if key in hard_teacher and step in hard_teacher[key]:
                    teacher_hard = int(hard_teacher[key][step])
                    teacher_prob = soft_teacher.get(key, {}).get(step)
                    if teacher_prob is None:
                        teacher_prob = np.full(3, (1.0 - 0.9) / 2.0, dtype=float)
                        teacher_prob[teacher_hard] = 0.9
                    true_one_hot = np.zeros(3, dtype=float)
                    if 0 <= true_target[i] < 3:
                        true_one_hot[true_target[i]] = 1.0
                    blend = (1.0 - blend_alpha) * teacher_prob + blend_alpha * true_one_hot
                    pseudo[i] = int(np.argmax(blend))
            fit_mask = valid & train_mask
            if not np.any(fit_mask):
                fit_mask = valid

            tree = DecisionTree(max_depth=tree_max_depth, min_samples_leaf=tree_min_leaf)
            tree.fit(features[fit_mask], pseudo[fit_mask])
            tree_preds[:, step - 1] = tree.predict(features)

            lookup = LookupBackoffModel(run_length_buckets)
            lookup.fit(cur[fit_mask], prev[fit_mask], run_len[fit_mask], context_hash[fit_mask], pseudo[fit_mask])
            lookup_preds[:, step - 1] = lookup.predict(cur, prev, run_len, context_hash)

        tree_metrics = classification_metrics(future[eval_mask, family_index, 0], tree_preds[eval_mask, 0], current_state=cur[eval_mask])
        lookup_metrics = classification_metrics(future[eval_mask, family_index, 0], lookup_preds[eval_mask, 0], current_state=cur[eval_mask])

        teacher_h1 = np.full(cur.shape[0], -1, dtype=int)
        for i, row_index in enumerate(row_indices):
            teacher_h1[i] = hard_teacher.get((family, int(row_index)), {}).get(1, -1)
        valid_teacher = eval_mask & (teacher_h1 >= 0)
        tree_ret = float(np.mean(tree_preds[valid_teacher, 0] == teacher_h1[valid_teacher])) if np.any(valid_teacher) else 0.0
        lookup_ret = float(np.mean(lookup_preds[valid_teacher, 0] == teacher_h1[valid_teacher])) if np.any(valid_teacher) else 0.0

        summary_rows.append(
            {
                "family": family,
                "scope": scope,
                "model": "decision_tree_student",
                **tree_metrics,
                "teacher_retention": tree_ret,
                "tree_depth": tree_max_depth,
                "lookup_entries": "",
            }
        )
        summary_rows.append(
            {
                "family": family,
                "scope": scope,
                "model": "lookup_rle_student",
                **lookup_metrics,
                "teacher_retention": lookup_ret,
                "tree_depth": "",
                "lookup_entries": int(len(np.unique(np.c_[cur, prev, run_len, context_hash], axis=0))),
            }
        )

        for i, row in enumerate(metadata_rows):
            base = {
                "family": family,
                "scope": scope,
                "split": split[i],
                "workload": row.get("workload", ""),
                "run_id": row.get("run_id", ""),
                "core_id": row.get("core_id", ""),
                "row_index": i,
                "family_state": int(cur[i]),
            }
            tree_row = dict(base)
            tree_row["model"] = "decision_tree_student"
            lookup_row = dict(base)
            lookup_row["model"] = "lookup_rle_student"
            for step in range(1, horizon + 1):
                tree_row[f"y_true_future_state_{step}"] = int(future[i, family_index, step - 1])
                tree_row[f"y_pred_future_state_{step}"] = int(tree_preds[i, step - 1])
                lookup_row[f"y_true_future_state_{step}"] = int(future[i, family_index, step - 1])
                lookup_row[f"y_pred_future_state_{step}"] = int(lookup_preds[i, step - 1])
            prediction_rows.append(tree_row)
            prediction_rows.append(lookup_row)

    write_csv_rows(output_dir / "student_predictions.csv", prediction_rows)
    write_csv_rows(output_dir / "student_summary.csv", summary_rows)
    write_json(
        output_dir / "student_manifest.json",
        {
            "scope": scope,
            "horizon": horizon,
            "blend_alpha": blend_alpha,
            "rows": len(prediction_rows),
        },
    )
    return summary_rows
