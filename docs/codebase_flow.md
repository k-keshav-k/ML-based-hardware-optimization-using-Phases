# Codebase Flow Guide

This document explains the full execution flow of the repository: what happens first, what happens later, how data is transformed at each stage, how models are trained, and how paper artifacts are generated.

It is written from the code paths currently used in this repository.

## 1. Big Picture

The project has two layers that work together:

1. Data and model pipeline:
- Collect raw `perf` traces from workloads.
- Merge them into interval datasets.
- Build timing-independent feature windows.
- Label phases unsupervised.
- Train predictors (baselines, transformer, student tree).
- Evaluate model outputs.

2. Paper/artifact pipeline:
- Validate generated artifacts.
- Run analysis experiments (replay, ablations, generalization, PMU-slot sensitivity, online/DVFS ingestion).
- Generate figures/tables.
- Generate paper LaTeX and PDF.

Core repository boundaries:

- Collection/reporting utilities: [hpc_phase_analysis](../hpc_phase_analysis)
- CLI entry points: [scripts](../scripts)
- Phase ML pipeline: [phase_ml](../phase_ml)
- Artifact analyses: [analysis](../analysis)
- Optional experiment drivers: [experiments](../experiments)
- Paper synthesis: [pipeline/run_all.py](../pipeline/run_all.py)
- Build shortcuts: [Makefile](../Makefile)

## 2. Directory Responsibilities

### 2.1 `hpc_phase_analysis/`

Shared utility library used by scripts and phase-ML modules.

Key files:

- [detect.py](../hpc_phase_analysis/detect.py): platform detection (`lscpu`, online CPUs, topology, PARSEC/SPEC detection, perf availability).
- [events.py](../hpc_phase_analysis/events.py): semantic counter family alias mapping, readiness checks, uncore support detection.
- [collector.py](../hpc_phase_analysis/collector.py): runs `perf stat`, parses interval/aggregate CSV output, handles sidecar uncore capture.
- [workloads.py](../hpc_phase_analysis/workloads.py): workload command builders and affinity helpers.
- [io_utils.py](../hpc_phase_analysis/io_utils.py): JSON/CSV helpers and shell wrappers.
- [constants.py](../hpc_phase_analysis/constants.py): default paths and constants.

### 2.2 `scripts/`

CLI wrappers around common operations.

Key files:

- [detect_platform.py](../scripts/detect_platform.py): writes platform JSON.
- [discover_events.py](../scripts/discover_events.py): writes semantic event alias and readiness artifacts.
- [run_workloads.py](../scripts/run_workloads.py): generic synthetic/PARSEC/SPEC collection CLI.
- [merge_runs.py](../scripts/merge_runs.py): merges per-run raw artifacts into wide interval/aggregate datasets.
- [slurm/train_phase_ml_transformer.sbatch](../scripts/slurm/train_phase_ml_transformer.sbatch): end-to-end GPU training/evaluation job on A100 setup.

Optional newer experiment wrappers:

- [online/run_burst_coscheduling.sh](../scripts/online/run_burst_coscheduling.sh)
- [online/run_dvfs_stress.sh](../scripts/online/run_dvfs_stress.sh)

### 2.3 `phase_ml/`

Main modeling pipeline for PARSEC phase signatures.

Key files:

- [collect_experiment_sets.py](../phase_ml/collect_experiment_sets.py): set1/set2/set3 PARSEC collection planner and executor.
- [build_dataset.py](../phase_ml/build_dataset.py): CLI for feature/window dataset build.
- [dataset.py](../phase_ml/dataset.py): interval feature creation + sliding window tensor generation.
- [features.py](../phase_ml/features.py): timing-independent feature engineering and forbidden-feature guardrails.
- [label_phases.py](../phase_ml/label_phases.py): phase labeling CLI.
- [labeling.py](../phase_ml/labeling.py): FGMM / 2-kmeans labeling, smoothing, minimum-duration filtering, split assignment.
- [clustering.py](../phase_ml/clustering.py): standardization, PCA, k-means, diagonal GMM.
- [splits.py](../phase_ml/splits.py): run-grouped and workload-holdout splits.
- [train_baselines.py](../phase_ml/train_baselines.py), [baselines.py](../phase_ml/baselines.py): classical models.
- [train_transformer.py](../phase_ml/train_transformer.py), [transformer_model.py](../phase_ml/transformer_model.py): teacher transformer.
- [train_student_tree.py](../phase_ml/train_student_tree.py): distilled student tree.
- [evaluate.py](../phase_ml/evaluate.py), [evaluation.py](../phase_ml/evaluation.py), [metrics.py](../phase_ml/metrics.py): metrics and report CSV/JSON outputs.

### 2.4 `analysis/`, `experiments/`, `pipeline/`

Artifact and publication layer.

Key files:

- [pipeline/run_all.py](../pipeline/run_all.py): orchestrates full artifact analysis and paper generation.
- [validate_artifact.py](../analysis/validate_artifact.py): leakage/count/consistency checks.
- [trace_policy_replay.py](../analysis/trace_policy_replay.py): trace-driven policy replay.
- [ablation.py](../analysis/ablation.py): detector ablations.
- [generalization.py](../analysis/generalization.py): transfer/generalization stress.
- [pmu_slot_sensitivity.py](../analysis/pmu_slot_sensitivity.py): constrained-counter sensitivity.
- [online_coscheduling.py](../analysis/online_coscheduling.py): analyzes measured online runs.
- [dvfs_stress.py](../analysis/dvfs_stress.py): cross-frequency drift analysis.
- [experiments/run_online_coscheduling.py](../experiments/run_online_coscheduling.py): live placement experiment runner.
- [experiments/run_dvfs_stress.py](../experiments/run_dvfs_stress.py): collection across frequency settings.

## 3. End-to-End Data Lifecycle

### Stage A: Platform and event discovery

Entry points:

- [scripts/detect_platform.py](../scripts/detect_platform.py)
- [scripts/discover_events.py](../scripts/discover_events.py)

What happens:

- Detect CPU topology, perf availability, PARSEC presence.
- Discover `perf list` aliases.
- Map aliases to semantic families (instructions, branches, L1/L3/offcore, uncore IMC).
- Compute readiness constraints (core and uncore).

Primary outputs:

- `results/platform_info.json`
- `results/event_alias_map.json`
- `results/study_readiness.json`

### Stage B: Raw trace collection

There are two collection styles.

1. Generic collector:
- [scripts/run_workloads.py](../scripts/run_workloads.py)

2. PARSEC experiment-set collector (recommended for phase-ML study):
- [phase_ml/collect_experiment_sets.py](../phase_ml/collect_experiment_sets.py)

`collect_experiment_sets.py` builds tasks for:

- `set1_single_process_multithread`
- `set2_multi_process_single_thread`
- `set3_hybrid_multi_process_multithread`

Collection internals:

- Each run executes workload command through `perf stat` via [hpc_phase_analysis/collector.py](../hpc_phase_analysis/collector.py).
- Per-run metadata is written before execution (`metadata.json`).
- Run results are written after execution (`collection_results.json`).
- Optional uncore sidecar stream can be captured when supported.
- Progress bar and ETA are printed to stderr by `collect_experiment_sets.py`.

Raw output structure:

- `results/raw_.../<run_id>/interval_perf.csv`
- `results/raw_.../<run_id>/interval_perf_raw.txt`
- `results/raw_.../<run_id>/metadata.json`
- `results/raw_.../<run_id>/collection_results.json`
- `results/raw_.../phase_ml_experiment_manifest.json`
- `results/raw_.../phase_ml_experiment_results.json`

### Stage C: Merge raw runs into unified datasets

Entry point:

- [scripts/merge_runs.py](../scripts/merge_runs.py)

What happens:

- Loads each run directory listed in manifest (unless overridden).
- Reads `interval_perf.csv`.
- Resolves normalized event family names using reverse alias map.
- Produces wide rows with `counter__<family>` columns.
- Preserves experiment metadata (`experiment_set`, `rep`, `concurrent_group_id`, process count, etc.).
- For system-wide modes, can consolidate logical CPUs to physical-core streams.
- Aligns uncore interval rows to core rows by timestamp tolerance.
- Writes merged interval and aggregate CSVs plus merge summary.

Primary outputs:

- `.../merged_interval_dataset.csv`
- `.../merged_aggregate_dataset.csv`
- `.../merge_summary.json`

### Stage D: Build phase-ML dataset (features + windows)

Entry points:

- [phase_ml/build_dataset.py](../phase_ml/build_dataset.py)
- [phase_ml/dataset.py](../phase_ml/dataset.py)

What happens:

1. Load merged interval CSV and keep `suite=parsec` rows.
2. Convert counters to timing-independent features in [phase_ml/features.py](../phase_ml/features.py):
- Instruction-normalized rates (per KI).
- Ratios (e.g., miss rates, load/store ratio).
- Shared context from true uncore IMC counters when available; fallback proxy otherwise.
3. Enforce forbidden feature policy (no cycles, per-ms, IPC/CPI, stalls, elapsed).
4. Build sliding windows per `(run_id, cpu_or_core_id, collection_unit_type)`.
5. Emit tensorized data and metadata.

Primary outputs:

- `dataset/interval_features.csv`
- `dataset/window_metadata.csv`
- `dataset/window_tensors.npz` (`X`, interval index arrays, `feature_columns`)
- `dataset/feature_manifest.json`

### Stage E: Unsupervised phase labeling

Entry points:

- [phase_ml/label_phases.py](../phase_ml/label_phases.py)
- [phase_ml/labeling.py](../phase_ml/labeling.py)

What happens:

1. Build split assignment by run (default) or workload holdout.
2. Standardize features (median fill + z-score).
3. Optional PCA.
4. Fit clustering model:
- `fgmm` (default diagonal GMM chosen by BIC), or
- `2kmeans`.
5. Post-process labels:
- smoothing window
- minimum phase duration enforcement
6. Attach interval and window targets:
- `phase_id`, `next_phase_id`, `phase_change`, run-length fields.

Primary outputs:

- `labels/interval_labels.csv`
- `labels/window_labels.csv`
- `labels/label_model.json` (standardizer/PCA/model payload + phase count)

### Stage F: Baseline model training

Entry points:

- [phase_ml/train_baselines.py](../phase_ml/train_baselines.py)
- [phase_ml/baselines.py](../phase_ml/baselines.py)

What happens:

- Loads windows and labels.
- Aligns arrays by `window_id`, flattens sequences, applies median fill.
- Splits train/eval using label split.
- Trains:
- `last_value`
- `nearest_centroid`
- `decision_tree`
- `logistic_regression`
- `rle_markov`
- optional `linear_svm`-style hinge model (`--include-linear-svm`)
- Produces per-window prediction CSV for each model and summary JSON.

Primary outputs:

- `baselines/*_predictions.csv`
- `baselines/baseline_summary.json`

### Stage G: Transformer teacher training (GPU optional)

Entry points:

- [phase_ml/train_transformer.py](../phase_ml/train_transformer.py)
- [phase_ml/transformer_model.py](../phase_ml/transformer_model.py)

What happens:

1. Load window tensors and labels.
2. Median-fill feature NaNs.
3. Build transformer encoder:
- input projection -> transformer encoder layers -> pooled last token
- two heads: next-phase classification and phase-change logits.
4. Optimize `CrossEntropy(next_phase) + BCE(phase_change)`.
5. Print live progress and loss at each configured step.
6. Save trained weights and prediction CSV for all windows.
7. Save `training_history.csv` and summary metrics including inference timing.

Primary outputs:

- `transformer/transformer.pt`
- `transformer/transformer_predictions.csv`
- `transformer/training_history.csv`
- `transformer/transformer_summary.json`

### Stage H: Student-tree distillation

Entry points:

- [phase_ml/train_student_tree.py](../phase_ml/train_student_tree.py)

What happens:

- Loads teacher predictions keyed by `window_id`.
- Trains two decision trees on train split:
- one for teacher next-phase outputs
- one for teacher phase-change outputs
- Evaluates on eval split against ground truth and teacher agreement.
- Emits student model JSON and prediction CSV.

Primary outputs:

- `student_tree/student_decision_tree_model.json`
- `student_tree/student_decision_tree_predictions.csv`
- `student_tree/student_decision_tree_summary.json`

### Stage I: Unified evaluation

Entry points:

- [phase_ml/evaluate.py](../phase_ml/evaluate.py)
- [phase_ml/evaluation.py](../phase_ml/evaluation.py)
- [phase_ml/metrics.py](../phase_ml/metrics.py)

What happens:

- Reads all prediction CSVs (baselines + transformer + student + extras).
- Filters to non-train split when available.
- Computes:
- next-phase accuracy
- macro-F1
- phase-change precision/recall/F1
- confusion matrix
- per-workload metrics
- Writes per-model artifacts and a common `model_comparison.csv`.

Primary outputs:

- `evaluation/model_comparison.csv`
- `evaluation/<model>_per_workload.csv`
- `evaluation/<model>_confusion_matrix.json`
- `evaluation/evaluation_summary.json`

## 4. Paper/Artifact Flow

Entry point:

- [pipeline/run_all.py](../pipeline/run_all.py)

What it orchestrates:

1. Validate consistency:
- split leakage checks, row counts, feature policy checks.
2. Run artifact analyses:
- uniqueness/equivalence classes
- trace-driven policy replay
- detector ablations
- transfer/generalization
- PMU-slot sensitivity
- hardware cost model
3. Optionally ingest measured:
- online co-scheduling summary (if present)
- DVFS stress summary (if present)
4. Render figures.
5. Generate `paper/main.tex` and references.
6. Write manifest for artifact reproducibility.

Relevant analysis modules:

- [analysis/validate_artifact.py](../analysis/validate_artifact.py)
- [analysis/trace_policy_replay.py](../analysis/trace_policy_replay.py)
- [analysis/ablation.py](../analysis/ablation.py)
- [analysis/generalization.py](../analysis/generalization.py)
- [analysis/pmu_slot_sensitivity.py](../analysis/pmu_slot_sensitivity.py)
- [analysis/hardware_cost.py](../analysis/hardware_cost.py)
- [analysis/online_coscheduling.py](../analysis/online_coscheduling.py)
- [analysis/dvfs_stress.py](../analysis/dvfs_stress.py)

## 5. Online and DVFS Experimental Extensions

These are optional extensions beyond offline replay:

- live online placement runner: [experiments/run_online_coscheduling.py](../experiments/run_online_coscheduling.py)
- DVFS collection runner: [experiments/run_dvfs_stress.py](../experiments/run_dvfs_stress.py)

Wrappers:

- [scripts/online/run_burst_coscheduling.sh](../scripts/online/run_burst_coscheduling.sh)
- [scripts/online/run_dvfs_stress.sh](../scripts/online/run_dvfs_stress.sh)

These produce measured results consumed by `pipeline/run_all.py` when files exist.

## 6. GPU Slurm End-to-End Path

GPU job script:

- [scripts/slurm/train_phase_ml_transformer.sbatch](../scripts/slurm/train_phase_ml_transformer.sbatch)

Runtime behavior (inside Singularity):

1. Activate conda and `uv`.
2. Resolve config path (`PHASE_ML_CONFIG`).
3. Optionally skip dataset build if artifacts already exist.
4. Optionally skip labeling if labels already exist.
5. Train baselines.
6. Train transformer (`--log-every` from `TRANSFORMER_LOG_EVERY`, default `1`).
7. Train student tree.
8. Run evaluation.

Outputs go to scratch paths configured in your config JSON (for example `config/phase_ml_indep_a100.json`).

## 7. Run Orders You Can Follow

### 7.1 Minimal sanity path (local)

1. `make detect`
2. `make discover`
3. `make collect-synth`
4. `make merge`
5. `make preprocess`
6. `make analyze`
7. `make report`

### 7.2 Full PARSEC phase-ML path (CPU + GPU)

1. Detect and discover.
2. Collect set1/set2/set3 with [phase_ml/collect_experiment_sets.py](../phase_ml/collect_experiment_sets.py).
3. Merge with [scripts/merge_runs.py](../scripts/merge_runs.py).
4. Submit GPU job [scripts/slurm/train_phase_ml_transformer.sbatch](../scripts/slurm/train_phase_ml_transformer.sbatch) for build/label/train/evaluate.
5. Sync outputs into repo `results/phase_ml_core_uncore_large/...` as needed.
6. Run `make paper` and compile PDF.

### 7.3 Publication artifact regeneration

1. Ensure `results/phase_ml_core_uncore_large/` and processed merge outputs exist.
2. Run:
- `make paper`
- `make test`
- `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## 8. Key Contracts Between Stages

- Merge -> dataset:
- Expects `counter__<family>` columns and metadata fields from merged interval CSV.

- Dataset -> labeling/training:
- Expects:
- `dataset/window_tensors.npz`
- `dataset/window_metadata.csv`
- `dataset/feature_manifest.json`

- Labeling -> training/evaluation:
- Expects:
- `labels/window_labels.csv`
- `labels/label_model.json`

- Baselines/transformer/student -> evaluation:
- Expects per-model `*_predictions.csv` with:
- `window_id`, `split`, `workload`, `run_id`, `phase_id`
- `y_true_next_phase`, `y_pred_next_phase`
- `y_true_phase_change`, `y_pred_phase_change`

- Evaluation/analysis -> paper:
- Expects `evaluation/model_comparison.csv` and analysis CSV/JSON outputs.

## 9. Where to Start Reading First

If you want a fast code reading sequence:

1. [scripts/run_workloads.py](../scripts/run_workloads.py)
2. [scripts/merge_runs.py](../scripts/merge_runs.py)
3. [phase_ml/dataset.py](../phase_ml/dataset.py)
4. [phase_ml/labeling.py](../phase_ml/labeling.py)
5. [phase_ml/baselines.py](../phase_ml/baselines.py)
6. [phase_ml/train_transformer.py](../phase_ml/train_transformer.py)
7. [phase_ml/train_student_tree.py](../phase_ml/train_student_tree.py)
8. [phase_ml/evaluation.py](../phase_ml/evaluation.py)
9. [pipeline/run_all.py](../pipeline/run_all.py)

That path gives the complete mental model from raw counters to final paper.

