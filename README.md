# Hardware Counter Phase Analysis

This repo contains a Linux `perf` pipeline for collecting hardware-counter traces, turning them into phase-analysis datasets, and training/evaluating PARSEC phase-prediction models.

The important distinction is:

- `hpc_phase_analysis/`: shared collection, preprocessing, event discovery, and reporting library code.
- `scripts/`: command-line entry points for platform detection, workload collection, merge/preprocess/analyze/report steps, and PARSEC setup.
- `phase_ml/`: PARSEC phase-labeling, feature-building, baseline models, transformer training, student-tree training, and evaluation.
- `config/`: local and A100-oriented pipeline configs.
- `tests/`: regression tests for the phase-ML path.
- `synthetic_workloads/`: small C workload used for smoke tests and fallback data collection.

Generated data and reports belong under `results/`; local PARSEC checkouts belong under `third_party/`. Both are ignored so the repo stays readable.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Or, with `uv`:

```bash
uv sync
```

Install the optional phase-ML dependencies when training neural models:

```bash
uv sync --extra phase-ml
```

## Synthetic Smoke Pipeline

Use this path to verify the basic collection and reporting flow without PARSEC:

```bash
make example
```

Equivalent explicit steps:

```bash
python scripts/detect_platform.py
python scripts/discover_events.py
python scripts/run_workloads.py --suite synthetic --threads 1,2,4,8 --reps 3 --modes interval
python scripts/merge_runs.py
python scripts/preprocess.py
python scripts/analyze_correlation.py
python scripts/build_report.py
```

## PARSEC Setup

Bootstrap the repo-local PARSEC mirror:

```bash
python scripts/setup_parsec.py
```

Build a small subset:

```bash
python scripts/setup_parsec.py --benchmarks blackscholes,canneal,streamcluster
```

`scripts/run_workloads.py --suite parsec` auto-detects `PARSEC_DIR`, `third_party/parsec-benchmark`, or common install paths. If no workload list is provided, it defaults to the validated strict subset:

- `blackscholes`
- `bodytrack`
- `canneal`
- `fluidanimate`
- `freqmine`

## Phase-ML Workflow

The phase-ML pipeline uses PARSEC interval traces and avoids timing-derived/cycle-derived features. See [phase_ml/README.md](/home/kk6081/res/phase_ml/README.md) for detailed collection, labeling, training, and evaluation commands.

Typical high-level flow:

```bash
uv run -m phase_ml.collect_experiment_sets --sets set1,set2,set3 --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine
uv run python scripts/merge_runs.py --input-dir results/raw_phase_ml_experiments --output-dir results/processed_phase_ml_experiments
uv run -m phase_ml.build_dataset --config config/phase_ml_defaults.json
uv run -m phase_ml.label_phases --config config/phase_ml_defaults.json
uv run -m phase_ml.train_baselines --config config/phase_ml_defaults.json
uv run -m phase_ml.evaluate --prediction-dir results/phase_ml/baselines
```

For A100 or scratch-backed runs, use `config/phase_ml_a100.json` or `config/phase_ml_indep_a100.json`.

## Useful Make Targets

```bash
make synthetic
make detect
make discover
make collect-synth
make merge
make preprocess
make analyze
make report
make clean
```

`make clean` removes generated result subdirectories. It does not remove source code, configs, tests, or the PARSEC checkout.

## Generated Files

The following are intentionally ignored:

- `results/` outputs
- `third_party/` checkouts
- `synthetic_workloads/bin/`
- Python caches and virtual environments
- local tool/session files such as `.agents/` and `.codex`
- local key material named `key` or `key.pub`
