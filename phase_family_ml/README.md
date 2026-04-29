# Factorized Counter-Family Phase LM

This package builds a clustered offline phase labeler and a reduced-counter
hardware-friendly detector. The current flow is:

1. `phase_family_ml.collect` records PARSEC interval counter traces.
2. `scripts/merge_runs.py` merges raw run folders into one interval table.
3. `phase_family_ml.build_counter_sequences` fits train-only k-means over the full safe counter vector and writes one clustered-label/value sequence per raw counter.
4. `phase_family_ml.run_ablation` selects the single best representative counter per family for approximating those clustered labels.
5. `phase_family_ml.build_counter_sequences --ablation-results ...` refreshes the family streams with the selected counters while preserving the same offline clustered phase labels.
6. `phase_family_ml.train_phase_detector` trains shallow global decision trees from the last 20 selected-counter intervals to the next 5 clustered phase labels.

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
  --prediction-horizon 5 \
  --tree-max-depth 6 \
  --tree-min-samples-leaf 8

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
phase_detector/phase_detector_summary_all.csv
```

The detector writes one summary row per future step. With the default
configuration, `prediction_step=1..5` corresponds to phases 1 through 5
intervals after the current online history window.

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
macro_f1
high_usage_recall
stable_case_accuracy
transition_case_accuracy
history_length
prediction_horizon
prediction_step
tree_depth
tree_internal_nodes
tree_leaves
estimated_storage_bytes
```

Reported baseline rows in the same summary CSV:

```text
baseline_last_state
baseline_majority
baseline_state_conditioned_majority
```

The tree rows also include gain columns against these baselines so the CSV shows
where the online history tree improves, especially on transition-heavy cases.

Phase-change metrics are intentionally not used.
