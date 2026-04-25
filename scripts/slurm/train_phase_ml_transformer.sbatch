#!/bin/bash
#SBATCH --job-name=phase_ml_transformer
#SBATCH --account=bmsc_ga_4551-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch --nv \
  --overlay "${OVERLAY_PATH:-/scratch/kk6081/overlay-25GB-500K.ext3:rw}" \
  "${SIF_PATH:-/scratch/kk6081/ubuntu-20.04.3.sif}" \
  /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH

conda activate \${CONDA_ENV:-llmr}
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export UV_LINK_MODE=\${UV_LINK_MODE:-copy}
export UV_PROJECT_ENVIRONMENT=\${UV_PROJECT_ENVIRONMENT:-.venv}
export PHASE_ML_CONFIG=\${PHASE_ML_CONFIG:-config/phase_ml_indep_a100.json}

cd \${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
mkdir -p /scratch/kk6081/indep/phase_ml_core_uncore_large logs
echo \"PROJECT_DIR=\$(pwd)\"
echo \"PHASE_ML_CONFIG=\${PHASE_ML_CONFIG}\"

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is not available in this container/conda environment. Install uv before submitting this job.' >&2
  exit 1
fi

uv sync --extra phase-ml

uv run -m phase_ml.build_dataset \
  --config \${PHASE_ML_CONFIG}

uv run python -c \"import os; from pathlib import Path; from phase_ml.config import load_config; from hpc_phase_analysis.io_utils import read_json; c=load_config(os.environ['PHASE_ML_CONFIG']); p=Path(c['dataset']['output_dir']) / 'feature_manifest.json'; m=read_json(p); print('Dataset manifest:', m); raise SystemExit('Dataset build produced no PARSEC interval rows. Check dataset.input_csv and the suite column.') if int(m.get('interval_rows', 0)) == 0 else None; raise SystemExit('Dataset build produced no training windows. Reduce sequence_length/prediction_horizon or check run lengths.') if int(m.get('windows', 0)) == 0 else None\"

uv run -m phase_ml.label_phases \
  --config \${PHASE_ML_CONFIG}

uv run -m phase_ml.train_baselines \
  --config \${PHASE_ML_CONFIG} \
  --include-linear-svm

uv run -m phase_ml.train_transformer \
  --config \${PHASE_ML_CONFIG}

uv run -m phase_ml.train_student_tree \
  --config \${PHASE_ML_CONFIG}

uv run -m phase_ml.evaluate \
  --prediction-dir \${BASELINE_PREDICTION_DIR:-/scratch/kk6081/indep/phase_ml_core_uncore_large/baselines} \
  --transformer-predictions \${TRANSFORMER_PREDICTIONS:-/scratch/kk6081/indep/phase_ml_core_uncore_large/transformer/transformer_predictions.csv} \
  --extra-predictions \${EXTRA_PREDICTIONS:-/scratch/kk6081/indep/phase_ml_core_uncore_large/student_tree/student_decision_tree_predictions.csv} \
  --output-dir \${EVAL_OUTPUT_DIR:-/scratch/kk6081/indep/phase_ml_core_uncore_large/evaluation}
"
