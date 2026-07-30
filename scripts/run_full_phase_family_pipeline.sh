#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/scratch/kk6081/finals_dataset}"
WORKLOADS="${WORKLOADS:-blackscholes,bodytrack,canneal,fluidanimate,freqmine,swaptions,streamcluster}"
INPUTS="${INPUTS:-simlarge}"
SETS="${SETS:-set1 set2 set3}"
SET1_THREADS="${SET1_THREADS:-1,4,16}"
GROUP_SIZE="${GROUP_SIZE:-2}"
HYBRID_THREADS="${HYBRID_THREADS:-4}"
REPS="${REPS:-10}"
INTERVAL_MS="${INTERVAL_MS:-10}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/phase_family_ml_defaults.json}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/kk6081/venvs/res-phase-family-ml}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/kk6081/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/kk6081/xdg-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/scratch/kk6081/python_pycache}"
mkdir -p "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"

echo "[full-run] syncing environment with PyTorch extra"
uv sync --extra teacher

for SET in $SETS; do
  echo "[full-run] collect start set=$SET"
  for INPUT in $INPUTS; do
    echo "[full-run] collect set=$SET input=$INPUT"
    uv run -m phase_family_ml.collect \
      --sets "$SET" \
      --dataset-root "$DATA_ROOT" \
      --workloads "$WORKLOADS" \
      --reps "$REPS" \
      --interval-ms "$INTERVAL_MS" \
      --parsec-input "$INPUT" \
      --set1-threads "$SET1_THREADS" \
      --group-size "$GROUP_SIZE" \
      --pairing-mode all_combinations \
      --hybrid-threads "$HYBRID_THREADS" \
      --core-collection-scope system_wide_physical_core \
      --collect-uncore \
      --skip-postprocess
  done

  echo "[full-run] merge set=$SET"
  uv run scripts/merge_runs.py \
    --input-dir "$DATA_ROOT/$SET/raw" \
    --output-dir "$DATA_ROOT/$SET/processed" \
    --manifest-path "$DATA_ROOT/$SET/raw/phase_family_ml_experiment_manifest.json" \
    --include-all-raw-runs

  echo "[full-run] pipeline set=$SET"
  uv run -m phase_family_ml.run_pipeline \
    --config "$PIPELINE_CONFIG" \
    --input "$DATA_ROOT/$SET/processed/merged_interval_dataset.csv" \
    --output-dir "$DATA_ROOT/$SET/results/phase_family_ml" \
    --experiment-mode config_group_holdout
done

echo "[full-run] complete data_root=$DATA_ROOT"
