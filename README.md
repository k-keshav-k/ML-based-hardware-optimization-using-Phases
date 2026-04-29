# Counter Family Phase LM

This repository keeps one pipeline:

1. Run PARSEC workloads and record hardware counter values.
2. Merge raw `perf` interval files into `merged_interval_dataset.csv`.
3. Fit train-split k-means over the full safe counter vector and write clustered phase labels.
4. Run ablation to select exactly one representative counter per counter family.
5. Rebuild selected-counter streams using the same offline clustered phase labels.
6. Train shallow reduced-counter phase trees that use the last 20 intervals to predict the next 5 phases.

## Main Commands

```bash
python scripts/setup_parsec.py
python -m phase_family_ml.collect \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine \
  --reps 10 \
  --interval-ms 10 \
  --parsec-input simsmall \
  --core-collection-scope system_wide_physical_core \
  --collect-uncore
```

By default, `phase_family_ml.collect` writes each experiment set under
`/scratch/kk6081/finals_dataset/<set>/`, merges that set into its own
`processed/` directory, and runs the phase-detector pipeline into
`results/phase_family_ml/`.

Stage-by-stage commands are documented in [phase_family_ml/README.md](/home/kk6081/res/phase_family_ml/README.md).

## Paper Result Artifacts

After the pipeline finishes, use these files to update the paper:

```text
/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
```

The intended framing is: offline k-means over all safe counters defines recurring workload phases; the deployed detector is a small set of shallow decision trees trained to predict future phase labels from one selected counter per family and a bounded online history, evaluated against simple persistence and majority baselines.

## Surviving Layout

- `phase_family_ml/`: counter sequence creation, ablation, reduced-counter phase detector training, and orchestration.
- `hpc_phase_analysis/`: shared PMU discovery, collection, event aliasing, CSV/JSON utilities, PARSEC command helpers.
- `scripts/`: PARSEC setup, platform/event discovery, raw-run merge, and Slurm jobs.
- `config/phase_family_ml_defaults.json`: the default family-LM pipeline config.
- `tests/test_phase_family_ml.py`: regression tests for the surviving flow.

Generated data belongs under `results/`; local PARSEC checkouts belong under `third_party/`.
