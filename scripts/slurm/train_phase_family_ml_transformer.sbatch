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
# 1) optionally build/rebuild counter sequence artifacts from merged intervals
# 2) train RoPE transformer teachers from ablation-selected counter streams
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
export PYTHONUNBUFFERED=\${PYTHONUNBUFFERED:-1}
export UV_LINK_MODE=\${UV_LINK_MODE:-copy}
export UV_PROJECT_ENVIRONMENT=\${UV_PROJECT_ENVIRONMENT:-.venv}

export PHASE_FAMILY_ML_CONFIG=\${PHASE_FAMILY_ML_CONFIG:-config/phase_family_ml_defaults.json}
export DATASET_INPUT_CSV=\${DATASET_INPUT_CSV:-}
export COUNTER_SEQUENCES_ROOT=\${COUNTER_SEQUENCES_ROOT:-}
export TEACHER_OUTPUT_ROOT=\${TEACHER_OUTPUT_ROOT:-}
export ABLATION_RESULTS_CSV=\${ABLATION_RESULTS_CSV:-}

# Sequence-stage controls.
export SKIP_SEQUENCE_BUILD=\${SKIP_SEQUENCE_BUILD:-auto}   # auto | 0 | 1
export EXPERIMENT_MODE=\${EXPERIMENT_MODE:-pooled_run_group}
export THRESHOLD_MODE=\${THRESHOLD_MODE:-global}
export REQUIRE_ABLATION_MATCH=\${REQUIRE_ABLATION_MATCH:-0}  # retained for compatibility; missing-family coverage now warns instead of failing

# Teacher training controls.
export FULL=\${FULL:-0}                              # 0 quick profile, 1 full profile
export SKIP_SEQUENCE_REFRESH=\${SKIP_SEQUENCE_REFRESH:-0}  # 0 refresh selected streams before teacher, 1 skip

cd \${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
mkdir -p logs

echo \"PROJECT_DIR=\$(pwd)\"
echo \"PHASE_FAMILY_ML_CONFIG=\${PHASE_FAMILY_ML_CONFIG}\"

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is not available in this container/conda environment. Install uv before submitting this job.' >&2
  exit 1
fi

uv sync --extra teacher

RUN_ROOT=\$(uv run python -c \"import os; from pathlib import Path; from phase_family_ml.config import load_config; c=load_config(os.environ['PHASE_FAMILY_ML_CONFIG']); print(Path(c['dataset']['output_dir']))\")
mkdir -p \"\${RUN_ROOT}\"

if [[ -z \"\${COUNTER_SEQUENCES_ROOT}\" ]]; then
  COUNTER_SEQUENCES_ROOT=\"\${RUN_ROOT}/counter_sequences_\${EXPERIMENT_MODE}_\${THRESHOLD_MODE}\"
fi
if [[ -z \"\${TEACHER_OUTPUT_ROOT}\" ]]; then
  TEACHER_OUTPUT_ROOT=\"\${RUN_ROOT}/teacher_\${EXPERIMENT_MODE}_\${THRESHOLD_MODE}\"
fi
if [[ -z \"\${ABLATION_RESULTS_CSV}\" ]]; then
  ABLATION_RESULTS_CSV=\"\${RUN_ROOT}/ablation/family_ablation_results.csv\"
fi

mkdir -p \"\${COUNTER_SEQUENCES_ROOT}\" \"\${TEACHER_OUTPUT_ROOT}\"

echo \"RUN_ROOT=\${RUN_ROOT}\"
echo \"DATASET_INPUT_CSV=\${DATASET_INPUT_CSV:-<from config>}\"
echo \"COUNTER_SEQUENCES_ROOT=\${COUNTER_SEQUENCES_ROOT}\"
echo \"TEACHER_OUTPUT_ROOT=\${TEACHER_OUTPUT_ROOT}\"
echo \"ABLATION_RESULTS_CSV=\${ABLATION_RESULTS_CSV}\"
echo \"SKIP_SEQUENCE_BUILD=\${SKIP_SEQUENCE_BUILD}\"
echo \"EXPERIMENT_MODE=\${EXPERIMENT_MODE}\"
echo \"THRESHOLD_MODE=\${THRESHOLD_MODE}\"
echo \"FULL=\${FULL}\"
echo \"SKIP_SEQUENCE_REFRESH=\${SKIP_SEQUENCE_REFRESH}\"
echo \"REQUIRE_ABLATION_MATCH=\${REQUIRE_ABLATION_MATCH}\"
echo \"PYTHONUNBUFFERED=\${PYTHONUNBUFFERED}\"

if [[ \"\${SKIP_SEQUENCE_REFRESH}\" != \"1\" ]]; then
  if [[ ! -f \"\${ABLATION_RESULTS_CSV}\" ]]; then
    echo \"Ablation CSV not found at \${ABLATION_RESULTS_CSV}. Set ABLATION_RESULTS_CSV or run ablation first.\" >&2
    exit 1
  fi
  uv run python - <<'PY'
import csv
import math
import os

from phase_family_ml.families import FAMILY_COUNTERS

p = os.environ['ABLATION_RESULTS_CSV']
e = os.environ['EXPERIMENT_MODE']
s = os.environ['THRESHOLD_MODE']
rows = list(csv.DictReader(open(p, newline='')))
best = {}

for r in rows:
    fam = r.get('family', '')
    if fam not in FAMILY_COUNTERS:
        continue
    if r.get('experiment', '') != e or r.get('scope', '') != s:
        continue
    if r.get('candidate_type', '') == 'unavailable':
        continue
    cset = (r.get('counter_set', '') or '').strip()
    if not cset:
        continue
    try:
        score = float(r.get('validation_score', ''))
    except Exception:
        score = float('-inf')
    if math.isnan(score):
        score = float('-inf')
    sel = int(r.get('selected', '0') or 0)
    prev = best.get(fam, (float('-inf'), -1, ''))
    if (score, sel) >= (prev[0], prev[1]):
        best[fam] = (score, sel, cset)

print(f'ablation_best_family_rows_for_{e}_{s}=' + str(len(best)))
for fam in sorted(best):
    print(f'ablation_best[{fam}]=' + best[fam][2] + ' score=' + str(best[fam][0]))
missing = [fam for fam in sorted(FAMILY_COUNTERS) if fam not in best]
if missing:
    print('[warn] no counter selected from ablation for families: ' + ', '.join(missing))
PY
fi

if [[ \"\${SKIP_SEQUENCE_BUILD}\" == \"1\" ]] || { [[ \"\${SKIP_SEQUENCE_BUILD}\" == \"auto\" ]] && uv run python -c \"import os; from pathlib import Path; root=Path(os.environ['COUNTER_SEQUENCES_ROOT']); manifests=list(root.glob('*/counter_sequence_manifest.json')); raise SystemExit(0 if manifests else 1)\"; }; then
  echo 'Using existing counter sequences; skipping build_counter_sequences.'
else
  # If sequences are missing, rebuild from merged interval CSV.
  sequence_args=(--config \${PHASE_FAMILY_ML_CONFIG})
  if [[ -n \"\${DATASET_INPUT_CSV}\" ]]; then
    sequence_args+=(--input \${DATASET_INPUT_CSV})
  fi
  sequence_args+=(--output-dir \${COUNTER_SEQUENCES_ROOT})
  sequence_args+=(--experiment-mode \${EXPERIMENT_MODE})
  sequence_args+=(--threshold-mode \${THRESHOLD_MODE})
  uv run -m phase_family_ml.build_counter_sequences \"\${sequence_args[@]}\"
fi

# Train transformer teachers for all discovered experiment/scope directories.
teacher_args=(--config \${PHASE_FAMILY_ML_CONFIG} --sequences-root \${COUNTER_SEQUENCES_ROOT} --output-dir \${TEACHER_OUTPUT_ROOT})
if [[ -n \"\${DATASET_INPUT_CSV}\" ]]; then
  teacher_args+=(--input \${DATASET_INPUT_CSV})
fi
teacher_args+=(--ablation-results \${ABLATION_RESULTS_CSV})
teacher_args+=(--experiment-mode \${EXPERIMENT_MODE})
teacher_args+=(--threshold-mode \${THRESHOLD_MODE})
if [[ \"\${SKIP_SEQUENCE_REFRESH}\" == \"1\" ]]; then
  teacher_args+=(--skip-sequence-refresh)
fi
if [[ \"\${FULL}\" == \"1\" ]]; then
  teacher_args+=(--full)
fi
uv run -m phase_family_ml.train_teacher \"\${teacher_args[@]}\"

# Basic completion validation: summary file must exist and be non-empty.
uv run python -c \"import csv, os, sys; from pathlib import Path; out=Path(os.environ['TEACHER_OUTPUT_ROOT']) / 'teacher_summary_all.csv'; sys.exit('teacher_summary_all.csv not found') if not out.exists() else None; rows=sum(1 for _ in csv.DictReader(out.open(newline=''))); print('teacher_summary_rows=' + str(rows)); sys.exit('teacher_summary_all.csv has zero rows') if rows == 0 else None\"
"
