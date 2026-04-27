#!/bin/bash
#SBATCH --job-name=phase_family_ml_teacher_students
#SBATCH --account=bmsc_ga_4551-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Train transformer teachers, then distill them to lookup/tree students.
#
# GPU is used by teacher training. Student distillation itself is CPU-only, but
# running it in the same job keeps the teacher/student artifacts synchronized.

mkdir -p logs

singularity exec --bind /scratch --nv \
  --overlay "${OVERLAY_PATH:-/scratch/kk6081/overlay-25GB-500K.ext3:rw}" \
  "${SIF_PATH:-/scratch/kk6081/ubuntu-20.04.3.sif}" \
  /bin/bash <<'BASH'
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

conda activate ${CONDA_ENV:-llmr}
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-.venv}

export PROJECT_DIR=${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
export PHASE_FAMILY_ML_CONFIG=${PHASE_FAMILY_ML_CONFIG:-config/phase_family_ml_defaults.json}
export DATASET_INPUT_CSV=${DATASET_INPUT_CSV:-results/processed/merged_interval_dataset.csv}
export ABLATION_RESULTS_CSV=${ABLATION_RESULTS_CSV:-results/phase_family_ml/ablation/family_ablation_results.csv}

export EXPERIMENT_MODE=${EXPERIMENT_MODE:-config_group_holdout}
export THRESHOLD_MODE=${THRESHOLD_MODE:-global}
export FULL=${FULL:-0}

# Set these to 1 to reuse existing artifacts.
export SKIP_SEQUENCE_BUILD=${SKIP_SEQUENCE_BUILD:-auto}      # auto | 0 | 1
export SKIP_SEQUENCE_REFRESH=${SKIP_SEQUENCE_REFRESH:-0}    # 0 refresh selected family streams before teacher
export SKIP_TEACHER=${SKIP_TEACHER:-0}
export SKIP_STUDENTS=${SKIP_STUDENTS:-0}

cd "${PROJECT_DIR}"
mkdir -p logs

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in this container/conda environment. Install uv before submitting this job." >&2
  exit 1
fi

uv sync --extra teacher

RUN_ROOT=$(uv run python -c "import os; from pathlib import Path; from phase_family_ml.config import load_config; c=load_config(os.environ['PHASE_FAMILY_ML_CONFIG']); print(Path(c['dataset']['output_dir']))")
export COUNTER_SEQUENCES_ROOT=${COUNTER_SEQUENCES_ROOT:-${RUN_ROOT}/counter_sequences}
export TEACHER_OUTPUT_ROOT=${TEACHER_OUTPUT_ROOT:-${RUN_ROOT}/teacher}
export STUDENTS_OUTPUT_ROOT=${STUDENTS_OUTPUT_ROOT:-${RUN_ROOT}/students}

mkdir -p "${COUNTER_SEQUENCES_ROOT}" "${TEACHER_OUTPUT_ROOT}" "${STUDENTS_OUTPUT_ROOT}"

echo "PROJECT_DIR=$(pwd)"
echo "PHASE_FAMILY_ML_CONFIG=${PHASE_FAMILY_ML_CONFIG}"
echo "DATASET_INPUT_CSV=${DATASET_INPUT_CSV}"
echo "COUNTER_SEQUENCES_ROOT=${COUNTER_SEQUENCES_ROOT}"
echo "ABLATION_RESULTS_CSV=${ABLATION_RESULTS_CSV}"
echo "TEACHER_OUTPUT_ROOT=${TEACHER_OUTPUT_ROOT}"
echo "STUDENTS_OUTPUT_ROOT=${STUDENTS_OUTPUT_ROOT}"
echo "EXPERIMENT_MODE=${EXPERIMENT_MODE}"
echo "THRESHOLD_MODE=${THRESHOLD_MODE}"
echo "FULL=${FULL}"
echo "SKIP_SEQUENCE_BUILD=${SKIP_SEQUENCE_BUILD}"
echo "SKIP_SEQUENCE_REFRESH=${SKIP_SEQUENCE_REFRESH}"
echo "SKIP_TEACHER=${SKIP_TEACHER}"
echo "SKIP_STUDENTS=${SKIP_STUDENTS}"

if [[ ! -f "${ABLATION_RESULTS_CSV}" ]]; then
  echo "Ablation CSV not found: ${ABLATION_RESULTS_CSV}" >&2
  echo "Run phase_family_ml.run_ablation first, or set ABLATION_RESULTS_CSV." >&2
  exit 1
fi

if [[ "${SKIP_SEQUENCE_BUILD}" == "1" ]] || { [[ "${SKIP_SEQUENCE_BUILD}" == "auto" ]] && uv run python -c "import os; from pathlib import Path; root=Path(os.environ['COUNTER_SEQUENCES_ROOT']); raise SystemExit(0 if list(root.glob('*/counter_sequence_manifest.json')) else 1)"; }; then
  echo "Using existing counter sequence root: ${COUNTER_SEQUENCES_ROOT}"
else
  sequence_args=(--config "${PHASE_FAMILY_ML_CONFIG}" --input "${DATASET_INPUT_CSV}" --output-dir "${COUNTER_SEQUENCES_ROOT}" --experiment-mode "${EXPERIMENT_MODE}" --threshold-mode "${THRESHOLD_MODE}")
  uv run -m phase_family_ml.build_counter_sequences "${sequence_args[@]}"
fi

if [[ "${SKIP_TEACHER}" != "1" ]]; then
  teacher_args=(--config "${PHASE_FAMILY_ML_CONFIG}" --input "${DATASET_INPUT_CSV}" --sequences-root "${COUNTER_SEQUENCES_ROOT}" --output-dir "${TEACHER_OUTPUT_ROOT}" --ablation-results "${ABLATION_RESULTS_CSV}" --experiment-mode "${EXPERIMENT_MODE}" --threshold-mode "${THRESHOLD_MODE}")
  if [[ "${SKIP_SEQUENCE_REFRESH}" == "1" ]]; then
    teacher_args+=(--skip-sequence-refresh)
  fi
  if [[ "${FULL}" == "1" ]]; then
    teacher_args+=(--full)
  fi
  uv run -m phase_family_ml.train_teacher "${teacher_args[@]}"
fi

if [[ "${SKIP_STUDENTS}" != "1" ]]; then
  uv run -m phase_family_ml.train_students \
    --config "${PHASE_FAMILY_ML_CONFIG}" \
    --sequences-root "${COUNTER_SEQUENCES_ROOT}" \
    --teacher-root "${TEACHER_OUTPUT_ROOT}" \
    --output-dir "${STUDENTS_OUTPUT_ROOT}"
fi

uv run python - <<'PY'
import csv
import os
import sys
from pathlib import Path

checks = [
    ("teacher", Path(os.environ["TEACHER_OUTPUT_ROOT"]) / "teacher_summary_all.csv"),
    ("students", Path(os.environ["STUDENTS_OUTPUT_ROOT"]) / "student_summary_all.csv"),
]
for label, path in checks:
    if not path.exists():
        sys.exit(f"{label} summary not found: {path}")
    rows = sum(1 for _ in csv.DictReader(path.open(newline="")))
    print(f"{label}_summary_rows={rows}")
    if rows == 0:
        sys.exit(f"{label} summary has zero rows: {path}")
PY
BASH
