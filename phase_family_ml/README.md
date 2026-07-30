# Factorized Counter-Family Phase LM

This package builds a clustered offline phase labeler and a reduced-counter
hardware-friendly detector. The current flow is:

1. `phase_family_ml.collect` records PARSEC interval counter traces.
2. `scripts/merge_runs.py` merges raw run folders into one interval table.
3. `phase_family_ml.build_counter_sequences` fits train-only k-means over the full safe counter vector and writes one clustered-label/value sequence per raw counter.
4. `phase_family_ml.run_ablation` selects the single best representative counter per family for approximating those clustered labels.
5. `phase_family_ml.build_counter_sequences --ablation-results ...` refreshes the family streams with the selected counters while preserving the same offline clustered phase labels.
6. `phase_family_ml.train_phase_detector` trains per-family next-phase predictors over clustered online histories, including shallow decision trees, Markov/RLE/HSMM-style temporal baselines, a ROCKET-style classifier, and optional tiny PyTorch TCN/Transformer baselines when `torch` is installed.

The default split is `config_group_holdout`, which keeps all reps of the same
collection config in the same train/val/test split. This avoids the optimistic
case where rep 1 of a config is in train and rep 2 of the same config is in val
or test.

## Recommended Run

Collect all three sets with the final dataset layout. By default this command
creates `/scratch/kk6081/finals_dataset/set1`, `set2`, and `set3`; writes raw
runs under each set's `raw/`; merges each set into `processed/`; and runs the
phase-detector pipeline into `results/phase_family_ml/`.

```bash
python -m phase_family_ml.collect \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 10 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --set1-threads 2,4,8 \
  --group-size 2 \
  --hybrid-threads 2 \
  --core-collection-scope system_wide_physical_core \
  --collect-uncore
```

## Collection And Merge

Small example:

```bash
python -m phase_family_ml.collect \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 5 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --set1-threads 2,4,8 \
  --group-size 2 \
  --hybrid-threads 2 \
  --core-collection-scope system_wide_physical_core \
  --collect-uncore
```

For set2, that writes:

```text
/scratch/kk6081/finals_dataset/set2/raw
/scratch/kk6081/finals_dataset/set2/processed
/scratch/kk6081/finals_dataset/set2/results/phase_family_ml
```

Pass `--skip-postprocess` to collect raw data only. Pass `--output-dir` to use
the legacy single raw-output directory mode.

Large collection sketch:

```bash
RAW_ROOT=/scratch/kk6081/set1/phase_family_ml_raw_large
PROCESSED_ROOT=/scratch/kk6081/set1/results/processed_phase_family_ml_large
WORKLOADS=blackscholes,bodytrack,canneal,fluidanimate,freqmine,swaptions,streamcluster

for INPUT in simsmall simmedium simlarge; do
  uv run -m phase_family_ml.collect \
    --sets set1 \
    --workloads "$WORKLOADS" \
    --reps 10 \
    --interval-ms 10 \
    --parsec-input "$INPUT" \
    --set1-threads 1,2,4,8,16 \
    --group-size 2 \
    --hybrid-threads 4 \
    --core-collection-scope system_wide_physical_core \
    --collect-uncore \
    --output-dir "$RAW_ROOT"
done

uv run scripts/merge_runs.py \
  --input-dir "$RAW_ROOT" \
  --output-dir "$PROCESSED_ROOT" \
  --manifest-path "$RAW_ROOT/phase_family_ml_experiment_manifest.json" \
  --include-all-raw-runs
```

Re-merge old raw data with the latest `merge_runs.py` so
`requested_input_size` is preserved for config-group splitting.

`set1` will usually create more runs than `set2`/`set3`. That is expected with
the current setup because `set1` expands over every requested thread count,
while `set2` and `set3` expand over workload groups. The collector now prints a
per-set task-plan summary before running so this difference is visible up front.

## Set Dataset Outputs

This produces the paper-facing clustered phase detector artifacts:

```text
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/counter_sequences/config_group_holdout/phase_cluster_summary.csv
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/ablation/family_ablation_results.csv
```

## Stage Commands

These commands are useful for debugging individual stages.

```bash
uv run -m phase_family_ml.build_counter_sequences \
  --config config/phase_family_ml_defaults.json \
  --input /scratch/kk6081/set3/results/processed_phase_family_ml_large/merged_interval_dataset.csv \
  --output-dir /scratch/kk6081/set3/results/phase_family_ml/counter_sequences \
  --experiment-mode config_group_holdout

uv run -m phase_family_ml.run_ablation \
  --config config/phase_family_ml_defaults.json \
  --input /scratch/kk6081/set3/results/processed_phase_family_ml_large/merged_interval_dataset.csv \
  --sequences-root /scratch/kk6081/set3/results/phase_family_ml/counter_sequences \
  --output-dir /scratch/kk6081/set3/results/phase_family_ml/ablation

uv run -m phase_family_ml.build_counter_sequences \
  --config config/phase_family_ml_defaults.json \
  --input /scratch/kk6081/set3/results/processed_phase_family_ml_large/merged_interval_dataset.csv \
  --output-dir /scratch/kk6081/set3/results/phase_family_ml/counter_sequences \
  --ablation-results /scratch/kk6081/set3/results/phase_family_ml/ablation/family_ablation_results.csv

uv run -m phase_family_ml.train_phase_detector \
  --config config/phase_family_ml_defaults.json \
  --sequences-root /scratch/kk6081/set3/results/phase_family_ml/counter_sequences \
  --output-dir /scratch/kk6081/set3/results/phase_family_ml/phase_detector \
  --history-length 20 \
  --prediction-horizon 1 \
  --tree-max-depth 6 \
  --tree-min-samples-leaf 8

```

For the full finals workflow across all sets and all input sizes, use:

```bash
bash scripts/run_full_phase_family_pipeline.sh
```

## Artifacts

Before ablation:

```text
counter_sequences/<split>/threshold_<scope>/counter_value_sequences/counter_sequence_<raw_counter>.csv
counter_sequences/<split>/phase_cluster_summary.csv
```

After ablation refresh:

```text
counter_sequences/<split>/threshold_<scope>/counter_sequence_<family>.csv
```

The family CSV is a direct copy of the selected best raw-counter CSV. No
family aggregation is performed after ablation. The labels in both raw-counter
and selected-family CSVs come from train-only k-means over the full safe
counter set, then clusters are ordered by resource pressure.

Reduced phase detector outputs:

```text
phase_detector/<split>/<scope>/phase_detector_predictions.csv
phase_detector/<split>/<scope>/phase_detector_summary.csv
phase_detector/<split>/<scope>/phase_detector_confusion_matrices.csv
phase_detector/<split>/<scope>/phase_detector_per_workload_accuracy.csv
phase_detector/<split>/<scope>/phase_detector_per_core_accuracy.csv
phase_detector/<split>/<scope>/phase_detector_per_thread_accuracy.csv
phase_detector/<split>/<scope>/phase_detector_per_process_count_accuracy.csv
phase_detector/<split>/<scope>/phase_detector_per_thread_process_accuracy.csv
phase_detector/<split>/<scope>/phase_detector_phase_behavior_by_thread.csv
phase_detector/<split>/<scope>/phase_detector_phase_behavior_by_process_count.csv
phase_detector/<split>/<scope>/phase_detector_phase_behavior_by_thread_process.csv
phase_detector/<split>/<scope>/phase_detector_hardware_budget.csv
phase_detector/phase_detector_summary_all.csv
```

The detector writes one row per family for single-family models and one
`__all_families__` row for baselines/cross-family models. It does not duplicate a
global result across family names. With the default configuration,
`prediction_step=1` corresponds to the next interval after the current history
window.

The extra grouped CSVs let you study how thread count (`set1`), process count
(`set2`), and the joint thread/process configuration (`set3`) affect both model
accuracy and the underlying clustered phase behavior.

## Metrics

State IDs are ordered clustered phases:

```text
0 = low-pressure / compute-light
1 = moderate mixed
2 = memory/cache/interference-heavy
```

Reported detector metrics:

```text
top1_accuracy
accuracy
accuracy_ci95_low
accuracy_ci95_high
macro_f1
weighted_f1
balanced_accuracy
high_usage_recall
stable_case_accuracy
transition_case_accuracy
transition_event_precision
transition_event_recall
transition_event_f1
transition_false_alarm_rate
transition_accuracy_ci95_low
transition_accuracy_ci95_high
eval_group_count
bootstrap_resamples
training_seconds
inference_latency_us
history_length
prediction_horizon
prediction_step
tree_depth
tree_internal_nodes
tree_leaves
estimated_storage_bytes
stored_parameters_or_entries
estimated_memory_bytes
model_storage_bytes
history_storage_bytes
discretizer_storage_bytes
approx_operations_per_prediction
hardware_complexity_category
deployment_recommendation
```

Reported baseline rows in the same summary CSV:

```text
baseline_last_state
baseline_majority
baseline_state_conditioned_majority
markov_phase_predictor
rle_markov_phase_predictor
hsmm_duration_phase_predictor
rocket_phase_classifier
rocket_phase_classifier_all_families
tcn_phase_classifier
tcn_phase_classifier_all_families
tiny_transformer_phase_classifier
tiny_transformer_phase_classifier_all_families
```

`baseline_last_state`, the state-conditioned majority model, and the
Markov/duration models receive the offline teacher's current phase. They are marked
`requires_oracle_current_phase=1` and are diagnostic upper bounds, not deployable
counter-only baselines. `online_current_state_tree_all_families` is the fair
history-length-one baseline.

For publication runs, `phase_family_ml.collect` defaults to all unordered workload
combinations. Use `--pairing-mode chunked` only to reproduce exploratory collections.
The merge stage writes `pmu_collection_quality.csv`, which reports enabled-time and
multiplexing quality without exposing those audit fields as model features.

The tree rows also include gain columns against these baselines so the CSV shows
where the online history tree improves, especially on transition-heavy cases.

If PyTorch is unavailable, the TCN and transformer rows are skipped and the
reason is recorded in `phase_detector_manifest.json`.

Reliability note: representative counter selection is done on `val` when a
validation split exists, while the detector summary CSVs report held-out `test`
metrics when a test split exists. The summary rows expose both
`counter_selection_split` and `report_split`.
