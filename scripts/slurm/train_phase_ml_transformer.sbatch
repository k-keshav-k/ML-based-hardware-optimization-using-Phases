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
export SKIP_DATASET_BUILD=\${SKIP_DATASET_BUILD:-auto}
export SKIP_LABELING=\${SKIP_LABELING:-auto}

cd \${PROJECT_DIR:-/scratch/kk6081/Phase-pipeline/ML-based-hardware-optimization-using-Phases}
mkdir -p /scratch/kk6081/indep/phase_ml_core_uncore_large logs
echo \"PROJECT_DIR=\$(pwd)\"
echo \"PHASE_ML_CONFIG=\${PHASE_ML_CONFIG}\"

if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is not available in this container/conda environment. Install uv before submitting this job.' >&2
  exit 1
fi

uv sync --extra phase-ml

uv run python -c \"import os; from pathlib import Path; from phase_ml.config import load_config; c=load_config(os.environ['PHASE_ML_CONFIG']); print('dataset.input_csv=' + str(c['dataset']['input_csv'])); print('dataset.output_dir=' + str(c['dataset']['output_dir'])); print('label_dir=' + str(Path(c['dataset']['output_dir']).parent / 'labels'))\"

if [[ \"\${SKIP_DATASET_BUILD}\" == \"1\" ]] || { [[ \"\${SKIP_DATASET_BUILD}\" == \"auto\" ]] && uv run python -c \"import os; from pathlib import Path; from phase_ml.config import load_config; d=Path(load_config(os.environ['PHASE_ML_CONFIG'])['dataset']['output_dir']); raise SystemExit(0 if all((d / name).exists() for name in ['feature_manifest.json', 'interval_features.csv', 'window_metadata.csv', 'window_tensors.npz']) else 1)\"; }; then
  echo 'Using existing phase-ML dataset; skipping build_dataset.'
  DATASET_REUSED=1
else
  uv run python -c \"import csv, os, sys; from collections import Counter; from pathlib import Path; from phase_ml.config import load_config; c=load_config(os.environ['PHASE_ML_CONFIG']); p=Path(c['dataset']['input_csv']); print('input_exists=' + str(p.exists())); sys.exit('Input CSV is missing. Either restore merged_interval_dataset.csv or set SKIP_DATASET_BUILD=1 to reuse an existing dataset directory.') if not p.exists() else None; f=p.open(newline=''); counts=Counter(row.get('suite', '<missing>') for row in csv.DictReader(f)); f.close(); print('input_rows=' + str(sum(counts.values()))); print('suite_counts=' + repr(counts.most_common(20))); sys.exit('Input CSV has no suite=parsec rows.') if counts.get('parsec', 0) == 0 else None\"
  uv run -m phase_ml.build_dataset \
    --config \${PHASE_ML_CONFIG}
  DATASET_REUSED=0
fi

if [[ \"\${DATASET_REUSED}\" == \"1\" ]]; then
  uv run python -c \"import csv, os, sys; from pathlib import Path; import numpy as np; from phase_ml.config import load_config; c=load_config(os.environ['PHASE_ML_CONFIG']); d=Path(c['dataset']['output_dir']); x=np.load(d / 'window_tensors.npz', allow_pickle=True)['X']; window_rows=sum(1 for _ in csv.DictReader((d / 'window_metadata.csv').open(newline=''))); print('Existing dataset windows=' + str(x.shape[0])); print('Existing window_metadata rows=' + str(window_rows)); sys.exit('Existing dataset has no training windows. Restore a non-empty dataset or the merged input CSV.') if x.shape[0] == 0 or window_rows == 0 else None\"
else
  uv run python -c \"import os, sys; from pathlib import Path; from phase_ml.config import load_config; from hpc_phase_analysis.io_utils import read_json; c=load_config(os.environ['PHASE_ML_CONFIG']); p=Path(c['dataset']['output_dir']) / 'feature_manifest.json'; m=read_json(p); print('Dataset manifest:', m); sys.exit('Dataset has no PARSEC interval rows.') if int(m.get('interval_rows', 0)) == 0 else None; sys.exit('Dataset has no training windows.') if int(m.get('windows', 0)) == 0 else None\"
fi

if [[ \"\${SKIP_LABELING}\" == \"1\" ]] || { [[ \"\${SKIP_LABELING}\" == \"auto\" ]] && uv run python -c \"import os; from pathlib import Path; from phase_ml.config import load_config; d=Path(load_config(os.environ['PHASE_ML_CONFIG'])['dataset']['output_dir']).parent / 'labels'; raise SystemExit(0 if all((d / name).exists() for name in ['label_model.json', 'interval_labels.csv', 'window_labels.csv']) else 1)\"; }; then
  echo 'Using existing phase labels; skipping label_phases.'
else
  uv run -m phase_ml.label_phases \
    --config \${PHASE_ML_CONFIG}
fi

uv run python -c \"import csv, os, sys; from pathlib import Path; from phase_ml.config import load_config; d=Path(load_config(os.environ['PHASE_ML_CONFIG'])['dataset']['output_dir']).parent / 'labels'; rows=sum(1 for _ in csv.DictReader((d / 'window_labels.csv').open(newline=''))); print('Existing label windows=' + str(rows)); sys.exit('Existing labels have no window rows. Restore labels or rerun label_phases from a non-empty dataset.') if rows == 0 else None\"

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
