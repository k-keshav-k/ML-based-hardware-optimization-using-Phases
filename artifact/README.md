# Artifact README

This artifact regenerates the conference-style draft and analysis tables from the available phase-ML outputs.

For a full architecture and stage-by-stage walkthrough, see [docs/codebase_flow.md](/home/kk6081/res/docs/codebase_flow.md).

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`
- Optional: `torch` only if retraining the transformer
- Optional: `latexmk` and `pdflatex` for PDF compilation

Install dependencies:

```bash
uv sync --extra phase-ml
```

## One-Command Reproduction

```bash
python -m pipeline.run_all --config experiments/configs/core_uncore_large.json
```

Or:

```bash
make paper
```

This regenerates:

- `results/a_star_analysis/validation_summary.json`
- `results/a_star_analysis/trace_policy_replay.csv`
- `results/a_star_analysis/detector_ablation.csv`
- `results/a_star_analysis/generalization.csv`
- `results/a_star_analysis/phase_uniqueness.csv`
- `results/a_star_analysis/hardware_cost.csv`
- `results/a_star_analysis/pmu_slot_sensitivity.csv`
- `figures/*.pdf`
- `paper/main.tex`
- `paper/references.bib`

Compile the paper:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## What Is Measured

- PARSEC phase-signature dataset with 300 merged runs.
- Run-grouped model evaluation from existing predictions.
- Trace-driven resource-policy replay.
- Detector ablations and transfer stress tests.
- Hardware cost estimates.
- PMU counter-slot sensitivity.

## Progress Logs

Dataset collection prints progress to stderr in the form:

```text
[collection] [########....................] 42/300 done, failed=0, elapsed=12.4m, eta=76.1m
```

Transformer training prints per-step losses to stderr and writes:

```bash
/scratch/kk6081/indep/phase_ml_core_uncore_large/transformer/training_history.csv
```

The Slurm output/error files under `logs/` contain the live progress stream.

## Live Online Co-Scheduling

Run this on a CPU node with PARSEC, `taskset`, and the phase artifact present:

```bash
export PARSEC_DIR=/scratch/kk6081/indep/third_party/bamos-parsec-benchmark
export OUTPUT_DIR=results/online_coscheduling_core_uncore_large
export PHASE_ARTIFACT=results/phase_ml_core_uncore_large
export WORKLOADS=blackscholes,canneal,fluidanimate,freqmine,swaptions,streamcluster
export THREADS=1,2
export REPS=5
export PARSEC_INPUT=simsmall
make online-cosched
```

The command writes:

- `results/online_coscheduling_core_uncore_large/online_runs.csv`
- `results/online_coscheduling_core_uncore_large/analysis/online_policy_summary.csv`

After copying those outputs back into the repository, run:

```bash
make paper
```

The paper generator includes measured online results only when `online_runs.csv`
exists.

## DVFS / Timing-Independence Stress

Run this only on a node where you are allowed to set CPU frequencies:

```bash
export PARSEC_DIR=/scratch/kk6081/indep/third_party/bamos-parsec-benchmark
export FREQUENCIES_KHZ=1200000,2200000
export OUTPUT_ROOT=results/dvfs_stress_core_uncore_large
export REPS=3
export PARSEC_INPUT=simsmall
export COLLECT_UNCORE=1
make dvfs-stress
```

The command collects, merges, builds per-frequency phase-ML artifacts, labels
phases, and writes `results/dvfs_stress_core_uncore_large/analysis/dvfs_stress.csv`.

## Claims Gated By Missing Measurements

- Do not claim online speedup/fairness until `online_policy_summary.csv` exists.
- Do not claim DVFS robustness until `dvfs_stress.csv` exists.
- Do not claim Intel CAT/resctrl or prefetch-control benefits unless those controls are actually run and logged.
