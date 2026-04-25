# PARSEC Phase ML

This module adds a PARSEC-only phase classification and prediction pipeline on top of the repository's existing `perf` interval traces.

The pipeline works on hosts where `kernel.perf_event_paranoid=2` by using task-local `perf stat` data. On hosts where `kernel.perf_event_paranoid=0` allows system-wide perf, it can collect logical-CPU streams with `perf stat -a -A -C` and merge SMT siblings into physical-core streams. Intel uncore IMC read/write counters can also be attached as real shared-memory traffic context when available. The schema keeps `collection_scope`, `collection_unit_type`, `cpu_or_core_id`, and optional `physical_core_id` fields so downstream labeling, baselines, transformer training, and evaluation use the same data contract in task-local and true per-core modes.

## Collection

Run the normal platform and event discovery first:

```bash
uv run python scripts/detect_platform.py
uv run python scripts/discover_events.py
```

For paranoid-2 collection, use task-local interval runs. At paranoid level 0 on this host, add `--collect-uncore` to attach actual IMC read/write traffic. The `phase_ml` event profile avoids cycle, stall, and timing-derived counters:

```bash
uv run python scripts/run_workloads.py \
  --suite parsec \
  --threads 1,2,4,8 \
  --reps 3 \
  --modes interval \
  --event-profile phase_ml \
  --require-confident-counters \
  --collect-uncore

uv run python scripts/merge_runs.py
```

The feature builder does not fabricate uncore data. If merged rows contain `counter__memory_read_bandwidth`, `counter__memory_write_bandwidth`, or `counter__total_memory_bandwidth`, it emits `shared_imc_read_mib`, `shared_imc_write_mib`, `shared_imc_total_mib`, and `shared_imc_write_fraction` with `shared_context_scope=system_wide_uncore`. Otherwise it falls back to task-local proxy context from LLC/offcore ratios.

For true per-CPU or per-physical-core streams at paranoid level 0, use `--core-collection-scope`:

```bash
uv run python scripts/run_workloads.py \
  --suite parsec \
  --threads 1,2,4,8 \
  --reps 3 \
  --modes interval \
  --event-profile phase_ml \
  --require-confident-counters \
  --collect-uncore \
  --core-collection-scope system_wide_physical_core

uv run python scripts/merge_runs.py
```

`system_wide_cpu` preserves one stream per logical CPU. `system_wide_physical_core` preserves raw logical-CPU artifacts but merges SMT siblings into one physical-core stream during `merge_runs.py` using `/sys/devices/system/cpu/cpu*/topology`. On the current host, the smoke probe produced physical-core IDs such as `0:0` and `1:0` and the dataset manifest reported `true_per_core_ready=true`.

## Three Experiment Sets

For phase-signature experiments, collect these PARSEC-only sets:

- `set1`: one PARSEC process at a time, each process using multiple threads.
- `set2`: multiple PARSEC processes running concurrently, one thread per process.
- `set3`: multiple PARSEC processes running concurrently, multiple threads per process.

The helper below uses task-local `perf` for each process and records experiment-set metadata so the merge and dataset stages can preserve it. With `perf_event_paranoid=0`, uncore collection is auto-enabled when discovery reports `study_ready_uncore=true`; pass `--no-collect-uncore` if you want the old task-local-only behavior.

```bash
uv run -m phase_ml.collect_experiment_sets \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 5 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --set1-threads 2,4,8 \
  --group-size 2 \
  --hybrid-threads 2 \
  --core-collection-scope system_wide_physical_core \
  --output-dir results/raw_phase_ml_experiments

uv run python scripts/merge_runs.py \
  --input-dir results/raw_phase_ml_experiments \
  --output-dir results/processed_phase_ml_experiments \
  --manifest-path results/raw_phase_ml_experiments/phase_ml_experiment_manifest.json
```

Use `--parsec-input test` only for smoke tests. Use `simsmall`, `simmedium`, or larger inputs for a real phase study.

## Build Dataset

```bash
uv run -m phase_ml.build_dataset \
  --input results/processed_phase_ml_experiments/merged_interval_dataset.csv \
  --output-dir results/phase_ml_experiments/dataset \
  --sequence-length 16 \
  --stride 1 \
  --prediction-horizon 20
```

Outputs are written under the selected output directory:

- `interval_features.csv`
- `window_metadata.csv`
- `window_tensors.npz`
- `feature_manifest.json`

Only instruction-normalized, access-normalized, and ratio features are emitted. Cycles, elapsed time, per-ms rates, CPI, IPC, and stall-derived features are excluded.

## Label Phases

Filtered GMM is the default label generator:

```bash
uv run -m phase_ml.label_phases \
  --dataset-dir results/phase_ml_experiments/dataset \
  --output-dir results/phase_ml_experiments/labels \
  --config config/phase_ml_defaults.json
```

Two-level k-means is available for the baseline phase labeler:

```bash
uv run -m phase_ml.label_phases \
  --dataset-dir results/phase_ml_experiments/dataset \
  --output-dir results/phase_ml_experiments/labels_2kmeans \
  --config config/phase_ml_defaults.json \
  --method 2kmeans
```

The default split is grouped by `run_id` to avoid interval leakage. A workload-held-out split can be generated separately:

```bash
uv run -m phase_ml.label_phases \
  --dataset-dir results/phase_ml_experiments/dataset \
  --config config/phase_ml_defaults.json \
  --split-policy workload_holdout \
  --holdout-workload freqmine \
  --output-dir results/phase_ml_experiments/labels_holdout_freqmine
```

## Train And Evaluate

For a fair model comparison, build the dataset, labels, baselines, and transformer with the same config. The A100 config uses a longer sequence length than the default config, so regenerate the baselines with `config/phase_ml_a100.json` before comparing against the transformer.

Classical baselines with the default local config:

```bash
uv run -m phase_ml.train_baselines \
  --dataset-dir results/phase_ml_experiments/dataset \
  --label-dir results/phase_ml_experiments/labels \
  --output-dir results/phase_ml_experiments/baselines \
  --config config/phase_ml_defaults.json
```

Transformer teacher and compact decision-tree student on the A100 machine:

```bash
uv sync --extra phase-ml

uv run -m phase_ml.build_dataset \
  --config config/phase_ml_a100.json

uv run -m phase_ml.label_phases \
  --config config/phase_ml_a100.json

uv run -m phase_ml.train_baselines \
  --config config/phase_ml_a100.json \
  --include-linear-svm

uv run -m phase_ml.train_transformer \
  --config config/phase_ml_a100.json

uv run -m phase_ml.train_student_tree \
  --config config/phase_ml_a100.json
```

Evaluation:

```bash
uv run -m phase_ml.evaluate \
  --prediction-dir results/phase_ml_experiments/baselines \
  --transformer-predictions results/phase_ml/transformer/transformer_predictions.csv \
  --extra-predictions results/phase_ml/student_tree/student_decision_tree_predictions.csv \
  --output-dir results/phase_ml_experiments/evaluation
```

The evaluator reports accuracy, macro-F1, phase-change precision/recall/F1, per-workload metrics, confusion matrices, and inference speed fields when the producer wrote them.

## Slurm A100 Job

The repository includes [scripts/slurm/train_phase_ml_transformer.sbatch](/home/kk6081/res/scripts/slurm/train_phase_ml_transformer.sbatch) for the NYU-style Singularity/A100 setup. Submit it from the repo root:

```bash
mkdir -p logs
sbatch scripts/slurm/train_phase_ml_transformer.sbatch
```

Useful overrides:

```bash
PROJECT_DIR=/scratch/kk6081/res \
OVERLAY_PATH=/scratch/kk6081/overlay-25GB-500K.ext3:rw \
SIF_PATH=/scratch/kk6081/ubuntu-20.04.3.sif \
PHASE_ML_CONFIG=config/phase_ml_a100.json \
sbatch scripts/slurm/train_phase_ml_transformer.sbatch
```

For the scratch-backed large dataset used on NYU burst, use:

```bash
PROJECT_DIR=/scratch/kk6081/res \
PHASE_ML_CONFIG=config/phase_ml_indep_a100.json \
sbatch scripts/slurm/train_phase_ml_transformer.sbatch
```

The scratch config expects merged intervals at `/scratch/kk6081/indep/processed_phase_ml_core_uncore_large/merged_interval_dataset.csv` and writes dataset, labels, baselines, transformer predictions, student-tree predictions, and evaluation outputs under `/scratch/kk6081/indep/phase_ml_core_uncore_large`.
