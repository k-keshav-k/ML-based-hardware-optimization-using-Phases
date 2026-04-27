# Factorized Counter-Family Phase LM

This package builds low/moderate/high state predictors for hardware-counter
families. The current flow is:

1. `phase_family_ml.collect` records PARSEC interval counter traces.
2. `scripts/merge_runs.py` merges raw run folders into one interval table.
3. `phase_family_ml.build_counter_sequences` writes one state/value sequence per raw counter.
4. `phase_family_ml.run_ablation` selects the single best counter per family from those raw-counter sequences.
5. `phase_family_ml.train_teacher` copies each selected counter sequence into the family stream, trains transformer teachers on state histories, and saves teacher checkpoints.
6. `phase_family_ml.train_students` trains run-length students, distilled history students, a scratch history tree, and a synthetic teacher-labeled history tree.
7. `phase_family_ml.evaluate` exports teacher/student comparison CSVs.

The default split is `config_group_holdout`, which keeps all reps of the same
collection config in the same train/val/test split. This avoids the optimistic
case where rep 1 of a config is in train and rep 2 of the same config is in val
or test.

## Recommended Run

Collect one experiment set per dataset, merge it, then run the normal pipeline
on that set-specific merged CSV. For example, collect only set1 with
`--sets set1`, only set2 with `--sets set2`, or only set3 with `--sets set3`.

```bash
FULL=1 DATASET_INPUT_CSV=results/processed/merged_interval_dataset.csv \
sbatch scripts/slurm/train_phase_family_ml_teacher_students.sbatch
```

## Collection And Merge

Small example:

```bash
python -m phase_family_ml.collect \
  --sets set1 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 5 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --set1-threads 2,4,8 \
  --group-size 2 \
  --hybrid-threads 2 \
  --core-collection-scope system_wide_physical_core \
  --collect-uncore \
  --output-dir results/raw_phase_family_ml_experiments

python scripts/merge_runs.py \
  --input-dir results/raw_phase_family_ml_experiments \
  --output-dir results/processed \
  --manifest-path results/raw_phase_family_ml_experiments/phase_family_ml_experiment_manifest.json
```

Large collection sketch:

```bash
RAW_ROOT=/scratch/kk6081/phase_family_ml_raw_large
PROCESSED_ROOT=results/processed_phase_family_ml_large
WORKLOADS=blackscholes,bodytrack,canneal,fluidanimate,freqmine,swaptions,streamcluster

for INPUT in simsmall simmedium simlarge; do
  python -m phase_family_ml.collect \
    --sets set1 \
    --workloads "$WORKLOADS" \
    --reps 30 \
    --interval-ms 10 \
    --parsec-input "$INPUT" \
    --set1-threads 1,2,4,8,16 \
    --group-size 2 \
    --hybrid-threads 4 \
    --core-collection-scope system_wide_physical_core \
    --collect-uncore \
    --output-dir "$RAW_ROOT"
done

python scripts/merge_runs.py \
  --input-dir "$RAW_ROOT" \
  --output-dir "$PROCESSED_ROOT" \
  --manifest-path "$RAW_ROOT/phase_family_ml_experiment_manifest.json" \
  --include-all-raw-runs
```

Re-merge old raw data with the latest `merge_runs.py` so
`requested_input_size` is preserved for config-group splitting.

Repeat the collection/merge/pipeline sequence separately for `set2` and `set3`
when you want independent set-level reports.

## Single Dataset Pipeline

For one merged CSV:

```bash
python -m phase_family_ml.run_pipeline \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml \
  --experiment-mode config_group_holdout
```

The single-dataset Slurm teacher/student wrapper is:

```bash
FULL=1 DATASET_INPUT_CSV=results/processed/merged_interval_dataset.csv \
sbatch scripts/slurm/train_phase_family_ml_teacher_students.sbatch
```

## Stage Commands

These commands are useful for debugging individual stages.

```bash
python -m phase_family_ml.build_counter_sequences \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_large/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml/counter_sequences \
  --experiment-mode config_group_holdout

python -m phase_family_ml.run_ablation \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_large/merged_interval_dataset.csv \
  --sequences-root results/phase_family_ml/counter_sequences \
  --output-dir results/phase_family_ml/ablation

python -m phase_family_ml.train_teacher \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --sequences-root results/phase_family_ml/counter_sequences \
  --ablation-results results/phase_family_ml/ablation/family_ablation_results.csv \
  --require-ablation-coverage \
  --experiment-mode config_group_holdout \
  --output-dir results/phase_family_ml/teacher

python -m phase_family_ml.train_students \
  --config config/phase_family_ml_defaults.json \
  --sequences-root results/phase_family_ml/counter_sequences \
  --teacher-root results/phase_family_ml/teacher \
  --output-dir results/phase_family_ml/students

python -m phase_family_ml.evaluate \
  --config config/phase_family_ml_defaults.json \
  --sequences-root results/phase_family_ml/counter_sequences \
  --teacher-root results/phase_family_ml/teacher \
  --students-root results/phase_family_ml/students \
  --ablation-dir results/phase_family_ml/ablation \
  --output-dir results/phase_family_ml/evaluation
```

## Artifacts

Before ablation:

```text
counter_sequences/<split>/threshold_<scope>/counter_value_sequences/counter_sequence_<raw_counter>.csv
```

After ablation refresh:

```text
counter_sequences/<split>/threshold_<scope>/counter_sequence_<family>.csv
```

The family CSV is a direct copy of the selected best raw-counter CSV. No
family aggregation is performed after ablation.

Teacher outputs:

```text
teacher/<split>/<scope>/teacher_predictions.csv
teacher/<split>/<scope>/teacher_summary.csv
teacher/<split>/<scope>/teacher_checkpoints/<family>.pt
teacher/teacher_summary_all.csv
```

Student models reported in `student_summary.csv`:

```text
decision_tree_student
lookup_rle_student
decision_tree_distilled_history
lookup_distilled_history
decision_tree_scratch_history
synthetic_distilled_history_tree
```

## Metrics

State IDs are:

```text
0 = low
1 = moderate
2 = high
```

Reported family metrics:

```text
accuracy
macro_f1
high_usage_recall
teacher_retention
```

Phase-change metrics are intentionally not used.

## Dependencies

Teacher training and synthetic teacher-labeled student generation need PyTorch:

```bash
uv sync --extra teacher
```

Transformer teacher training should use a GPU. Student-only distillation is
CPU-friendly, except `synthetic_distilled_history_tree` needs PyTorch to query
saved teacher checkpoints.
