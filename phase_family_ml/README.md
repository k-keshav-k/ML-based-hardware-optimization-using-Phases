# Factorized Family-Wise Phase LM

This package is a standalone pipeline for interpretable, family-wise hardware
phase modeling. It does not modify `phase_ml` behavior.

The pipeline uses these families:

- `L1`
- `L2`
- `LLC`
- `memory_offcore`
- `branch_control`
- `core_fp`

and keeps the existing forbidden-feature policy (no cycles, CPI/IPC, elapsed,
per-ms, or stall-derived signals in modeling features).

## How The Pipeline Is Structured

The family pipeline has seven runnable stages:

1. Collection (`phase_family_ml.collect`) or generic collection with profile `phase_family_lm`
2. Raw-merge (`scripts/merge_runs.py`)
3. Family label build (`phase_family_ml.build_family_labels`)
4. Counter ablation (`phase_family_ml.run_ablation`)
5. Transformer teacher training (`phase_family_ml.train_teacher`)
6. Student distillation (`phase_family_ml.train_students`)
7. Evaluation export (`phase_family_ml.evaluate`)

There is also one orchestrator that runs all stages:

- `phase_family_ml.run_pipeline`

## What Each Script Does

### `python -m phase_family_ml.collect`
Builds PARSEC set1/set2/set3 capture plans and runs `perf stat` interval
collection using the family-LM counter profile (includes L2/FP when supported).
Writes manifest and per-run metadata under the output directory.

Primary outputs:

- `phase_family_ml_experiment_manifest.json`
- `phase_family_ml_experiment_results.json`
- per-run folders with `interval_perf.csv`, `metadata.json`, `collection_results.json`

### `python scripts/merge_runs.py`
Merges raw per-run interval files into one wide table with `counter__*` columns
and run metadata.

Primary outputs:

- `merged_interval_dataset.csv`
- `merged_aggregate_dataset.csv`
- `merge_summary.json`

### `python -m phase_family_ml.build_family_labels`
Loads merged intervals, computes family usage scores, fits train-only tertile
thresholds, applies them to val/test, and writes one file per family:

- `family_labels_<family>.csv`

If `--ablation-results` is provided, label construction switches to the
selected per-family counter set from `family_ablation_results.csv` so teacher
training uses the ablation-finalized counter definitions.

Each file includes:

- `family_state`
- `future_state_1` ... `future_state_20`
- `will_change_within_horizon`
- `time_to_change`
- `split`, `workload`, `run_id`, `core_id`

### `python -m phase_family_ml.run_ablation`
Runs:

- per-family singleton sweeps plus all-counters upper bound
- global exhaustive one-counter-per-family search

Selection score is:

- `0.4 * accuracy + 0.4 * phase_change_f1 + 0.2 * high_usage_recall`

Primary output:

- `family_ablation_results.csv`

### `python -m phase_family_ml.train_teacher`
Trains family-wise transformer teachers (RoPE + causal multi-head attention +
feedforward + layer norm), tries context on/off, and keeps best validation mode
per family.

By default, this command refreshes labels from ablation selections when
`<dataset.output_dir>/ablation/family_ablation_results.csv` (or
`--ablation-results`) exists. Use `--skip-label-refresh` to train from prebuilt
labels unchanged.

During training, logs now print:
- current experiment/scope/family/context
- counter set used for that family label stream
- per-epoch `train_loss` and `val_loss`

Primary outputs:

- `teacher_predictions.csv`
- `teacher_summary.csv`
- `teacher_summary_all.csv`

### `python -m phase_family_ml.train_students`
Trains two students per family:

- decision-tree distilled student (blended teacher/true supervision)
- lookup/RLE-backoff student

Primary outputs:

- `student_predictions.csv`
- `student_summary.csv`
- `student_summary_all.csv`

### `python -m phase_family_ml.evaluate`
Aggregates teacher and student predictions into final report CSVs.

Primary outputs:

- `family_teacher_student_comparison.csv`
- `tuple_phase_prediction_results.csv`
- `hardware_cost_summary.csv`
- copy/export of `family_ablation_results.csv`

### `python -m phase_family_ml.run_pipeline`
Runs labels -> ablation -> labels(refresh from selected counters) -> teacher ->
students -> evaluation end-to-end.

## Setup

From repo root:

```bash
uv sync --extra phase-ml
```

`phase-ml` extra is needed for teacher transformer training (PyTorch).

## Counter Collection Recipes

## 1) Discover machine and PMU aliases

```bash
uv run python scripts/detect_platform.py
uv run python scripts/discover_events.py
```

## 2) Collect family-LM PARSEC sets

```bash
uv run -m phase_family_ml.collect \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 5 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --set1-threads 2,4,8 \
  --group-size 2 \
  --hybrid-threads 2 \
  --core-collection-scope system_wide_physical_core \
  --output-dir results/raw_phase_family_ml_experiments
```

Single-workload note: if you pass exactly one workload (for example
`--workloads blackscholes`) and include `set2` or `set3`, the collector now
automatically replicates that workload within each concurrent group so those
sets still run.

Alternative generic collector (same profile):

```bash
uv run python scripts/run_workloads.py \
  --suite parsec \
  --modes interval \
  --event-profile phase_family_lm \
  --require-confident-counters
```

## 3) Merge raw runs

```bash
uv run python scripts/merge_runs.py \
  --input-dir results/raw_phase_family_ml_experiments \
  --output-dir results/processed_phase_family_ml_experiments \
  --manifest-path results/raw_phase_family_ml_experiments/phase_family_ml_experiment_manifest.json
```

## Local Stage-By-Stage Run

```bash
# Build family labels (all experiment modes, both threshold scopes by default)
uv run -m phase_family_ml.build_family_labels \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_experiments/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml/family_labels

# Run ablations
uv run -m phase_family_ml.run_ablation \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_experiments/merged_interval_dataset.csv \
  --labels-root results/phase_family_ml/family_labels \
  --output-dir results/phase_family_ml/ablation

# Train teachers
uv run -m phase_family_ml.train_teacher \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_experiments/merged_interval_dataset.csv \
  --labels-root results/phase_family_ml/family_labels \
  --ablation-results results/phase_family_ml/ablation/family_ablation_results.csv \
  --output-dir results/phase_family_ml/teacher

# Train students
uv run -m phase_family_ml.train_students \
  --config config/phase_family_ml_defaults.json \
  --labels-root results/phase_family_ml/family_labels \
  --teacher-root results/phase_family_ml/teacher \
  --output-dir results/phase_family_ml/students

# Evaluate and write final CSVs
uv run -m phase_family_ml.evaluate \
  --config config/phase_family_ml_defaults.json \
  --labels-root results/phase_family_ml/family_labels \
  --teacher-root results/phase_family_ml/teacher \
  --students-root results/phase_family_ml/students \
  --ablation-dir results/phase_family_ml/ablation \
  --output-dir results/phase_family_ml/evaluation
```

## One-Command End-To-End Run

```bash
uv run -m phase_family_ml.run_pipeline \
  --config config/phase_family_ml_defaults.json \
  --input results/processed_phase_family_ml_experiments/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml
```

Use `--full` on training/orchestration commands for larger training budgets.

## Slurm: Transformer Teacher Training

A new A100 Slurm script is available:

- [train_phase_family_ml_transformer.sbatch](/home/kk6081/res/scripts/slurm/train_phase_family_ml_transformer.sbatch)

It is teacher-focused and does:

1. optional family-label build (auto-reuse when labels already exist)
2. family-wise transformer teacher training
3. summary validation (`teacher_summary_all.csv` must be non-empty)

Submit:

```bash
mkdir -p logs
sbatch scripts/slurm/train_phase_family_ml_transformer.sbatch
```

Common overrides:

```bash
PROJECT_DIR=/scratch/kk6081/res \
OVERLAY_PATH=/scratch/kk6081/overlay-25GB-500K.ext3:rw \
SIF_PATH=/scratch/kk6081/ubuntu-20.04.3.sif \
PHASE_FAMILY_ML_CONFIG=config/phase_family_ml_defaults.json \
DATASET_INPUT_CSV=/scratch/kk6081/indep/processed_phase_ml_core_uncore_large/merged_interval_dataset.csv \
FAMILY_LABELS_ROOT=/scratch/kk6081/indep/phase_family_ml/family_labels \
TEACHER_OUTPUT_ROOT=/scratch/kk6081/indep/phase_family_ml/teacher \
ABLATION_RESULTS_CSV=/scratch/kk6081/indep/phase_family_ml/ablation/family_ablation_results.csv \
EXPERIMENT_MODE=all \
THRESHOLD_MODE=both \
SKIP_LABEL_BUILD=auto \
SKIP_LABEL_REFRESH=0 \
FULL=1 \
sbatch scripts/slurm/train_phase_family_ml_transformer.sbatch
```

## Final Required CSV Artifacts

After full evaluation, these are under your chosen evaluation output root:

- `family_ablation_results.csv`
- `family_teacher_student_comparison.csv`
- `tuple_phase_prediction_results.csv`
- `hardware_cost_summary.csv`
