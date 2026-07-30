# Counter Family Phase LM

This repository keeps one pipeline:

1. Run PARSEC workloads and record hardware counter values.
2. Merge raw `perf` interval files into `merged_interval_dataset.csv`.
3. Fit train-split k-means over the full safe counter vector and write clustered phase labels.
4. Run ablation to select exactly one representative counter per counter family.
5. Rebuild selected-counter streams using the same offline clustered phase labels.
6. Predict the single global teacher phase from either one-family or cross-family clustered counter histories. The family name identifies the online input, not a separate target label.

## Main Commands

```bash
python scripts/setup_parsec.py
python -m phase_family_ml.collect \
  --sets set1,set2,set3 \
  --workloads blackscholes,bodytrack,canneal,fluidanimate,freqmine,swaptions,streamcluster \
  --reps 10 \
  --interval-ms 10 \
  --parsec-input simlarge \
  --set1-threads 1,4,16 \
  --hybrid-threads 4 \
  --pairing-mode all_combinations \
  --core-collection-scope system_wide_physical_core \
  --collect-uncore
```

By default, `phase_family_ml.collect` writes each experiment set under
`/scratch/kk6081/finals_dataset/<set>/`, merges that set into its own
`processed/` directory, and runs the phase-detector pipeline into
`results/phase_family_ml/`.

For the full finals workflow across `set1`, `set2`, and `set3`, including
installing the PyTorch extra for the optional TCN/Transformer baselines, run:

```bash
bash scripts/run_full_phase_family_pipeline.sh
```

For the publication run used by the MLForSys 2026 extended abstract, use the
stricter scratch-only runner. It uses only the lightweight tree/baseline
dependencies, refuses to mix old raw runs, checks PMU permission before collection,
enumerates all workload pairs, runs grouped and workload-disjoint analyses, and
writes a validation certificate when every expected artifact is present:

```bash
bash scripts/run_mlforsys2026_real_data.sh
```

Use `PREFLIGHT_ONLY=1` for a no-collection platform and task-plan check. The default
PARSEC checkout, environment, caches, raw data, predictions, and validation report all
remain under `/scratch/kk6081`.

The script places the uv environment and all package/bytecode caches under
`/scratch/kk6081`; it does not create a repository-local `.venv`.

Stage-by-stage commands are documented in [phase_family_ml/README.md](/home/kk6081/res/phase_family_ml/README.md).

## Paper Result Artifacts

After the pipeline finishes, use these files to update the paper:

```text
/scratch/kk6081/finals_dataset/set1/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/finals_dataset/set2/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/finals_dataset/set3/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
```

The intended framing is: offline k-means over all safe counters defines recurring workload phases; the deployed detector is a small set of shallow decision trees trained to predict future phase labels from one selected counter per family and a bounded online history, evaluated against simple persistence and majority baselines.

Teacher-phase persistence, state-conditioned, Markov, and duration rows consume the
offline teacher's true current phase and are therefore oracle diagnostics. The fair
deployable history ablation is `online_current_state_tree_all_families`.

Detector-side comparison artifacts now also include:

```text
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_hardware_budget.csv
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_confusion_matrices.csv
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_workload_accuracy.csv
/scratch/kk6081/finals_dataset/<set>/results/phase_family_ml/phase_detector/config_group_holdout/global/phase_detector_per_core_accuracy.csv
```

## Surviving Layout

- `phase_family_ml/`: counter sequence creation, ablation, reduced-counter phase detector training, and orchestration.
- `hpc_phase_analysis/`: shared PMU discovery, collection, event aliasing, CSV/JSON utilities, PARSEC command helpers.
- `scripts/`: PARSEC setup, platform/event discovery, raw-run merge, and Slurm jobs.
- `config/phase_family_ml_defaults.json`: the default family-LM pipeline config.
- `tests/test_phase_family_ml.py`: regression tests for the surviving flow.

Large generated datasets, predictions, caches, and environments belong under
`/scratch/kk6081/`, not in this repository or the quota-limited home directory. Keep
only source, compact summaries, and publication figures in Git.
