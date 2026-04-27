"""Student distillation models for family-wise phase prediction.

Students include:
- decision-tree student (distilled with blended teacher/true targets)
- lookup/RLE-style backoff student for cheap table inference
- history-window decision tree and lookup students trained on the same examples
  used by the transformer teacher
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hpc_phase_analysis.io_utils import load_csv_rows, write_csv_rows, write_json

from .data import load_scope_family_data, states_matrix
from .metrics import classification_metrics
from .teacher import _build_examples
from .tree import DecisionTree


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


class HistoryLookupModel:
    """Lookup table over recent state-history windows with suffix backoff."""

    def __init__(self) -> None:
        self.tables: list[dict[tuple[int, ...], int]] = []
        self.fallback = 1

    def fit(self, history: np.ndarray, target: np.ndarray) -> "HistoryLookupModel":
        if history.shape[0] == 0 or target.size == 0:
            self.tables = []
            self.fallback = 1
            return self
        self.fallback = Counter(target.astype(int).tolist()).most_common(1)[0][0]
        width = history.shape[2]
        tables: list[dict[tuple[int, ...], int]] = []
        for suffix_len in range(history.shape[1], 0, -1):
            counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
            for row, tgt in zip(history, target):
                key = tuple(row[-suffix_len:, :].astype(int).reshape(suffix_len * width).tolist())
                counts[key][int(tgt)] += 1
            tables.append({key: counter.most_common(1)[0][0] for key, counter in counts.items()})
        self.tables = tables
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        output = np.full(history.shape[0], self.fallback, dtype=int)
        if not self.tables:
            return output
        width = history.shape[2]
        for index, row in enumerate(history):
            for table_index, table in enumerate(self.tables):
                suffix_len = history.shape[1] - table_index
                key = tuple(row[-suffix_len:, :].astype(int).reshape(suffix_len * width).tolist())
                if key in table:
                    output[index] = table[key]
                    break
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


def _family_context_modes(path: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    for row in load_csv_rows(path):
        family = str(row.get("family", ""))
        mode = str(row.get("context_mode", ""))
        if family and mode:
            modes.setdefault(family, mode)
    return modes


def _teacher_targets(
    family: str,
    row_ids: np.ndarray,
    step: int,
    hard_teacher: dict[tuple[str, int], dict[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    target = np.full(row_ids.shape[0], -1, dtype=int)
    valid = np.zeros(row_ids.shape[0], dtype=bool)
    for index, row_id in enumerate(row_ids):
        value = hard_teacher.get((family, int(row_id)), {}).get(step, -1)
        if value >= 0:
            target[index] = int(value)
            valid[index] = True
    return target, valid


def train_students_for_experiment(
    experiment_dir: Path,
    scope: str,
    teacher_predictions_path: Path,
    output_dir: Path,
    horizon: int,
    history_length: int,
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
    context_modes = _family_context_modes(teacher_predictions_path)
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
                "feature_source": "run_length",
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
                "feature_source": "run_length",
                **lookup_metrics,
                "teacher_retention": lookup_ret,
                "tree_depth": "",
                "lookup_entries": int(len(np.unique(np.c_[cur, prev, run_len, context_hash], axis=0))),
            }
        )

        context_mode = context_modes.get(family, "without_context")
        x_hist, y_hist, split_hist, current_hist, meta_hist, row_ids_hist = _build_examples(
            family_index,
            current,
            future,
            split,
            metadata_rows,
            history_length,
            context_mode,
        )
        if x_hist.shape[0] > 0:
            hist_features = x_hist.reshape(x_hist.shape[0], -1)
            eval_hist = _eval_mask(split_hist)
            train_hist = split_hist == "train"
            if not np.any(train_hist):
                train_hist = np.ones(split_hist.shape[0], dtype=bool)

            hist_preds: dict[str, np.ndarray] = {
                "decision_tree_distilled_history": np.full((x_hist.shape[0], horizon), -1, dtype=int),
                "lookup_distilled_history": np.full((x_hist.shape[0], horizon), -1, dtype=int),
                "decision_tree_scratch_history": np.full((x_hist.shape[0], horizon), -1, dtype=int),
            }
            lookup_entries_by_step: list[int] = []

            for step in range(1, horizon + 1):
                true_target = y_hist[:, step - 1].astype(int)
                valid_true = true_target >= 0
                scratch_fit = train_hist & valid_true
                if not np.any(scratch_fit):
                    scratch_fit = valid_true
                if np.any(scratch_fit):
                    scratch_tree = DecisionTree(max_depth=tree_max_depth, min_samples_leaf=tree_min_leaf)
                    scratch_tree.fit(hist_features[scratch_fit], true_target[scratch_fit])
                    hist_preds["decision_tree_scratch_history"][:, step - 1] = scratch_tree.predict(hist_features)

                teacher_target, valid_teacher = _teacher_targets(family, row_ids_hist, step, hard_teacher)
                distill_fit = train_hist & valid_teacher
                if not np.any(distill_fit):
                    distill_fit = valid_teacher
                if np.any(distill_fit):
                    distill_tree = DecisionTree(max_depth=tree_max_depth, min_samples_leaf=tree_min_leaf)
                    distill_tree.fit(hist_features[distill_fit], teacher_target[distill_fit])
                    hist_preds["decision_tree_distilled_history"][:, step - 1] = distill_tree.predict(hist_features)

                    lookup = HistoryLookupModel()
                    lookup.fit(x_hist[distill_fit], teacher_target[distill_fit])
                    hist_preds["lookup_distilled_history"][:, step - 1] = lookup.predict(x_hist)
                    lookup_entries_by_step.append(sum(len(table) for table in lookup.tables))

            teacher_h1, valid_teacher_h1 = _teacher_targets(family, row_ids_hist, 1, hard_teacher)
            for model_name, preds in hist_preds.items():
                metrics = classification_metrics(y_hist[eval_hist, 0], preds[eval_hist, 0], current_state=current_hist[eval_hist])
                valid_retention = eval_hist & valid_teacher_h1
                retention = float(np.mean(preds[valid_retention, 0] == teacher_h1[valid_retention])) if np.any(valid_retention) else 0.0
                summary_rows.append(
                    {
                        "family": family,
                        "scope": scope,
                        "model": model_name,
                        "feature_source": "state_history",
                        "context_mode": context_mode,
                        "history_length": history_length,
                        **metrics,
                        "teacher_retention": retention,
                        "tree_depth": tree_max_depth if "decision_tree" in model_name else "",
                        "lookup_entries": max(lookup_entries_by_step) if model_name == "lookup_distilled_history" and lookup_entries_by_step else "",
                    }
                )

            for i, row in enumerate(meta_hist):
                base = {
                    "family": family,
                    "scope": scope,
                    "split": split_hist[i],
                    "workload": row.get("workload", ""),
                    "run_id": row.get("run_id", ""),
                    "core_id": row.get("core_id", ""),
                    "row_index": int(row_ids_hist[i]),
                    "family_state": int(current_hist[i]),
                    "context_mode": context_mode,
                    "feature_source": "state_history",
                }
                for model_name, preds in hist_preds.items():
                    item = dict(base)
                    item["model"] = model_name
                    for step in range(1, horizon + 1):
                        item[f"y_true_future_state_{step}"] = int(y_hist[i, step - 1])
                        item[f"y_pred_future_state_{step}"] = int(preds[i, step - 1])
                    prediction_rows.append(item)

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
                "feature_source": "run_length",
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
            "history_length": history_length,
            "blend_alpha": blend_alpha,
            "rows": len(prediction_rows),
        },
    )
    return summary_rows
