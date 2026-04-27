# Counter Family Phase LM

This repository keeps one pipeline:

1. Run PARSEC workloads and record hardware counter values.
2. Merge raw `perf` interval files into `merged_interval_dataset.csv`.
3. Build one LM-style sequence file per counter.
4. Run ablation to select exactly one counter per counter family.
5. Train a family-wise transformer LM on the selected counters.
6. Distill the teacher to a decision tree and lookup table.

## Main Commands

```bash
python scripts/setup_parsec.py
python -m phase_family_ml.collect --sets set1,set2,set3 --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine
python scripts/merge_runs.py --input-dir results/raw_phase_family_ml_experiments --output-dir results/processed
python -m phase_family_ml.run_pipeline --config config/phase_family_ml_defaults.json
```

Stage-by-stage commands are documented in [phase_family_ml/README.md](/home/kk6081/res/phase_family_ml/README.md).

## Surviving Layout

- `phase_family_ml/`: counter sequence creation, ablation, teacher training, student distillation, evaluation, orchestration.
- `hpc_phase_analysis/`: shared PMU discovery, collection, event aliasing, CSV/JSON utilities, PARSEC command helpers.
- `scripts/`: PARSEC setup, platform/event discovery, raw-run merge, and Slurm teacher job.
- `config/phase_family_ml_defaults.json`: the default family-LM pipeline config.
- `tests/test_phase_family_ml.py`: regression tests for the surviving flow.

Generated data belongs under `results/`; local PARSEC checkouts belong under `third_party/`.
