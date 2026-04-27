#!/bin/bash
#SBATCH --job-name=phase_family_ml_transformer
#SBATCH --account=bmsc_ga_4551-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Single-job SLURM workflow for family-wise transformer teacher training:
# 1) optionally build/rebuild family label artifacts from merged intervals
# 2) train RoPE transformer teachers for each family (with context-mode selection)
# 3) write teacher summaries/predictions to the configured output directory

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

export PHASE_FAMILY_ML_CONFIG=\${PHASE_FAMILY_ML_CONFIG:-config/phase_family_ml_defaults.json}
export DATASET_INPUT_CSV=\${DATASET_INPUT_CSV:-}
export FAMILY_LABELS_ROOT=\${FAMILY_LABELS_ROOT:-}
export TEACHER_OUTPUT_ROOT=\${TEACHER_OUTPUT_ROOT:-}

# Label-stage controls.
export SKIP_LABEL_BUILD=\${SKIP_LABEL_BUILD:-auto}   # auto | 0 | 1
export EXPERIMENT_MODE=\${EXPERIMENT_MODE:-all}
export THRESHOLD_MODE=\${THRESHOLD_MODE:-both}

# Teacher training controls.
export FULL=\${FULL:-0}                              # 0 quick profile, 1 full profile

cd \${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
mkdir -p logs

echo \"PROJECT_DIR=\$(pwd)\"
echo \"PHASE_FAMILY_ML_CONFIG=\${PHASE_FAMILY_ML_CONFIG}\"

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is not available in this container/conda environment. Install uv before submitting this job.' >&2
  exit 1
fi

uv sync --extra phase-ml

RUN_ROOT=\$(uv run python -c \"import os; from pathlib import Path; from phase_family_ml.config import load_config; c=load_config(os.environ['PHASE_FAMILY_ML_CONFIG']); print(Path(c['dataset']['output_dir']))\")
mkdir -p \"\${RUN_ROOT}\"

if [[ -z \"\${FAMILY_LABELS_ROOT}\" ]]; then
  FAMILY_LABELS_ROOT=\"\${RUN_ROOT}/family_labels\"
fi
if [[ -z \"\${TEACHER_OUTPUT_ROOT}\" ]]; then
  TEACHER_OUTPUT_ROOT=\"\${RUN_ROOT}/teacher\"
fi

mkdir -p \"\${FAMILY_LABELS_ROOT}\" \"\${TEACHER_OUTPUT_ROOT}\"

echo \"RUN_ROOT=\${RUN_ROOT}\"
echo \"DATASET_INPUT_CSV=\${DATASET_INPUT_CSV:-<from config>}\"
echo \"FAMILY_LABELS_ROOT=\${FAMILY_LABELS_ROOT}\"
echo \"TEACHER_OUTPUT_ROOT=\${TEACHER_OUTPUT_ROOT}\"
echo \"SKIP_LABEL_BUILD=\${SKIP_LABEL_BUILD}\"
echo \"EXPERIMENT_MODE=\${EXPERIMENT_MODE}\"
echo \"THRESHOLD_MODE=\${THRESHOLD_MODE}\"
echo \"FULL=\${FULL}\"

if [[ \"\${SKIP_LABEL_BUILD}\" == \"1\" ]] || { [[ \"\${SKIP_LABEL_BUILD}\" == \"auto\" ]] && uv run python -c \"import os; from pathlib import Path; root=Path(os.environ['FAMILY_LABELS_ROOT']); manifests=list(root.glob('*/family_label_manifest.json')); raise SystemExit(0 if manifests else 1)\"; }; then
  echo 'Using existing family labels; skipping build_family_labels.'
else
  # If labels are missing, rebuild from merged interval CSV.
  label_args=(--config \${PHASE_FAMILY_ML_CONFIG})
  if [[ -n \"\${DATASET_INPUT_CSV}\" ]]; then
    label_args+=(--input \${DATASET_INPUT_CSV})
  fi
  label_args+=(--output-dir \${FAMILY_LABELS_ROOT})
  label_args+=(--experiment-mode \${EXPERIMENT_MODE})
  label_args+=(--threshold-mode \${THRESHOLD_MODE})
  uv run -m phase_family_ml.build_family_labels \"\${label_args[@]}\"
fi

# Train transformer teachers for all discovered experiment/scope directories.
teacher_args=(--config \${PHASE_FAMILY_ML_CONFIG} --labels-root \${FAMILY_LABELS_ROOT} --output-dir \${TEACHER_OUTPUT_ROOT})
if [[ \"\${FULL}\" == \"1\" ]]; then
  teacher_args+=(--full)
fi
uv run -m phase_family_ml.train_teacher \"\${teacher_args[@]}\"

# Basic completion validation: summary file must exist and be non-empty.
uv run python -c \"import csv, os, sys; from pathlib import Path; out=Path(os.environ['TEACHER_OUTPUT_ROOT']) / 'teacher_summary_all.csv'; sys.exit('teacher_summary_all.csv not found') if not out.exists() else None; rows=sum(1 for _ in csv.DictReader(out.open(newline=''))); print('teacher_summary_rows=' + str(rows)); sys.exit('teacher_summary_all.csv has zero rows') if rows == 0 else None\"
"
