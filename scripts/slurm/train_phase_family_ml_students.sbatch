#!/bin/bash
#SBATCH --job-name=phase_family_ml_students
#SBATCH --account=bmsc_ga_4551-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue

# Student-only distillation job. Submit after the teacher job succeeds, e.g.:
# sbatch --dependency=afterok:<teacher_job_id> scripts/slurm/train_phase_family_ml_students.sbatch

mkdir -p logs

singularity exec --bind /scratch \
  --overlay "${OVERLAY_PATH:-/scratch/kk6081/overlay-25GB-500K.ext3:rw}" \
  "${SIF_PATH:-/scratch/kk6081/ubuntu-20.04.3.sif}" \
  /bin/bash <<'BASH'
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

conda activate ${CONDA_ENV:-llmr}
set -euo pipefail

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-.venv}

export PROJECT_DIR=${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
export PHASE_FAMILY_ML_CONFIG=${PHASE_FAMILY_ML_CONFIG:-config/phase_family_ml_defaults.json}
export COUNTER_SEQUENCES_ROOT=${COUNTER_SEQUENCES_ROOT:-/scratch/kk6081/results/phase_family_ml/counter_sequences}
export TEACHER_OUTPUT_ROOT=${TEACHER_OUTPUT_ROOT:-/scratch/kk6081/results/phase_family_ml/teacher}
export STUDENTS_OUTPUT_ROOT=${STUDENTS_OUTPUT_ROOT:-/scratch/kk6081/results/phase_family_ml/students}

cd "${PROJECT_DIR}"
mkdir -p logs "${STUDENTS_OUTPUT_ROOT}"

echo "PROJECT_DIR=$(pwd)"
echo "PHASE_FAMILY_ML_CONFIG=${PHASE_FAMILY_ML_CONFIG}"
echo "COUNTER_SEQUENCES_ROOT=${COUNTER_SEQUENCES_ROOT}"
echo "TEACHER_OUTPUT_ROOT=${TEACHER_OUTPUT_ROOT}"
echo "STUDENTS_OUTPUT_ROOT=${STUDENTS_OUTPUT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in this container/conda environment. Install uv before submitting this job." >&2
  exit 1
fi

uv sync --extra teacher

uv run python - <<'PY'
import os
import sys
from pathlib import Path

from phase_family_ml.orchestration import experiment_dirs, scopes_for_experiment

sequences_root = Path(os.environ["COUNTER_SEQUENCES_ROOT"])
teacher_root = Path(os.environ["TEACHER_OUTPUT_ROOT"])

if not sequences_root.exists():
    sys.exit(f"Counter sequence root not found: {sequences_root}")
if not teacher_root.exists():
    sys.exit(f"Teacher root not found: {teacher_root}")

missing = []
found = 0
for exp_dir in experiment_dirs(sequences_root):
    for scope in scopes_for_experiment(exp_dir):
        teacher_predictions = teacher_root / exp_dir.name / scope / "teacher_predictions.csv"
        if teacher_predictions.exists():
            found += 1
        else:
            missing.append(str(teacher_predictions))

if found == 0:
    sys.exit(f"No teacher_predictions.csv files found under {teacher_root} for experiments in {sequences_root}")

print(f"student_training_inputs={found}")
if missing:
    print("[warn] missing teacher predictions for some sequence scopes:")
    for path in missing[:20]:
        print(f"[warn] {path}")
    if len(missing) > 20:
        print(f"[warn] ... {len(missing) - 20} more")
PY

uv run -m phase_family_ml.train_students \
  --config "${PHASE_FAMILY_ML_CONFIG}" \
  --sequences-root "${COUNTER_SEQUENCES_ROOT}" \
  --teacher-root "${TEACHER_OUTPUT_ROOT}" \
  --output-dir "${STUDENTS_OUTPUT_ROOT}"

uv run python - <<'PY'
import csv
import os
import sys
from pathlib import Path

summary = Path(os.environ["STUDENTS_OUTPUT_ROOT"]) / "student_summary_all.csv"
if not summary.exists():
    sys.exit(f"student_summary_all.csv not found: {summary}")

rows = sum(1 for _ in csv.DictReader(summary.open(newline="")))
print(f"student_summary_rows={rows}")
if rows == 0:
    sys.exit(f"student_summary_all.csv has zero rows: {summary}")
PY
BASH
