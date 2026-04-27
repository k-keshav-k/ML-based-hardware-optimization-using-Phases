# Factorized Counter-Family Phase LM

This package owns the surviving flow:

1. `phase_family_ml.collect` records PARSEC counter intervals.
2. `scripts/merge_runs.py` merges raw run folders into one interval table.
3. `phase_family_ml.build_counter_sequences` writes one LM-style value sequence per counter.
4. `phase_family_ml.run_ablation` selects one counter per family with global exhaustive search.
5. `phase_family_ml.train_teacher` trains transformer teachers from selected-counter state histories.
6. `phase_family_ml.train_students` distills the teacher to a decision tree and lookup/RLE table.
7. `phase_family_ml.evaluate` exports teacher/student comparison CSVs.

## One Command After Merge

```bash
python -m phase_family_ml.run_pipeline \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml
```

`run_pipeline` builds counter sequences, runs ablation, refreshes the selected one-counter-per-family streams, trains teachers on low/moderate/high state histories, trains students, and evaluates outputs.

## Collection And Merge

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
  --output-dir results/raw_phase_family_ml_experiments

python scripts/merge_runs.py \
  --input-dir results/raw_phase_family_ml_experiments \
  --output-dir results/processed \
  --manifest-path results/raw_phase_family_ml_experiments/phase_family_ml_experiment_manifest.json
```

## Stage Commands

```bash
python -m phase_family_ml.build_counter_sequences \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --output-dir results/phase_family_ml/counter_sequences

python -m phase_family_ml.run_ablation \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --sequences-root results/phase_family_ml/counter_sequences \
  --output-dir results/phase_family_ml/ablation

python -m phase_family_ml.train_teacher \
  --config config/phase_family_ml_defaults.json \
  --input results/processed/merged_interval_dataset.csv \
  --sequences-root results/phase_family_ml/counter_sequences \
  --ablation-results results/phase_family_ml/ablation/family_ablation_results.csv \
  --require-ablation-coverage \
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

Install teacher dependencies with:

```bash
uv sync --extra teacher
```
