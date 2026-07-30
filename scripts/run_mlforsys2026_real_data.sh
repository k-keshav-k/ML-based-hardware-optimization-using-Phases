#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/kk6081}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_ROOT/mlforsys2026_real_data}"
PARSEC_DIR="${PARSEC_DIR:-$SCRATCH_ROOT/parsec-benchmark}"
WORKLOADS="${WORKLOADS:-blackscholes,bodytrack,canneal,fluidanimate,freqmine,swaptions,streamcluster}"
REPS="${REPS:-10}"
INTERVAL_MS="${INTERVAL_MS:-10}"
PARSEC_INPUT="${PARSEC_INPUT:-simlarge}"
SET1_THREADS="${SET1_THREADS:-1,4,16}"
GROUP_SIZE="${GROUP_SIZE:-2}"
HYBRID_THREADS="${HYBRID_THREADS:-4}"
TASK_ORDER_SEED="${TASK_ORDER_SEED:-17}"
CORE_COLLECTION_SCOPE="${CORE_COLLECTION_SCOPE:-system_wide_physical_core}"
COLLECT_UNCORE="${COLLECT_UNCORE:-off}"
PREPARE_PARSEC="${PREPARE_PARSEC:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/mlforsys2026_real_data.json}"

resolved_scratch_root="$(readlink -m "$SCRATCH_ROOT")"
case "$resolved_scratch_root" in
  /scratch/kk6081|/scratch/kk6081/*) ;;
  *)
    echo "[fatal] SCRATCH_ROOT must remain under /scratch/kk6081: $resolved_scratch_root" >&2
    exit 2
    ;;
esac
SCRATCH_ROOT="$resolved_scratch_root"

require_scratch_path() {
  local resolved
  resolved="$(readlink -m "$1")"
  case "$resolved" in
    "$SCRATCH_ROOT"/*) ;;
    *)
      echo "[fatal] Refusing non-scratch path: $resolved" >&2
      exit 2
      ;;
  esac
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fatal] Required command not found: $1" >&2
    exit 2
  fi
}

require_scratch_path "$DATA_ROOT"
require_scratch_path "$PARSEC_DIR"
for command_name in uv perf taskset git lscpu; do
  require_command "$command_name"
done

if find "$DATA_ROOT" -path '*/raw/*' -type f -print -quit 2>/dev/null | grep -q .; then
  echo "[fatal] DATA_ROOT already contains raw data: $DATA_ROOT" >&2
  echo "[fatal] Use a new DATA_ROOT; mixing publication runs invalidates repetition counts." >&2
  exit 2
fi

export PARSEC_DIR
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH_ROOT/venvs/res-mlforsys2026}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$SCRATCH_ROOT/python_pycache}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp/mlforsys2026}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH_ROOT/matplotlib-cache}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

mkdir -p \
  "$DATA_ROOT/preflight" \
  "$DATA_ROOT/provenance" \
  "$UV_PROJECT_ENVIRONMENT" \
  "$UV_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$PYTHONPYCACHEPREFIX" \
  "$TMPDIR" \
  "$MPLCONFIGDIR"

date -u +%Y-%m-%dT%H:%M:%SZ > "$DATA_ROOT/provenance/run_started_utc.txt"
git rev-parse HEAD > "$DATA_ROOT/provenance/base_git_commit.txt"
git status --short > "$DATA_ROOT/provenance/git_status_at_start.txt"
git diff > "$DATA_ROOT/provenance/working_tree_at_start.patch"
sha256sum \
  "$0" \
  "$PIPELINE_CONFIG" \
  > "$DATA_ROOT/provenance/source_sha256.txt"

echo "[setup] syncing the lightweight tree/baseline environment under $UV_PROJECT_ENVIRONMENT"
uv sync --frozen

if [[ "$PREPARE_PARSEC" == "1" ]]; then
  echo "[setup] preparing PARSEC and simlarge inputs under $PARSEC_DIR"
  uv run scripts/setup_parsec.py \
    --root "$PARSEC_DIR" \
    --benchmarks "$WORKLOADS" \
    --download-native-inputs
elif [[ ! -x "$PARSEC_DIR/bin/parsecmgmt" ]]; then
  echo "[fatal] PARSEC is missing at $PARSEC_DIR; rerun with PREPARE_PARSEC=1" >&2
  exit 2
fi

echo "[preflight] discovering the platform and requiring confident core PMU events"
uv run scripts/discover_events.py \
  --output-dir "$DATA_ROOT/preflight" \
  --strict-hardware-study \
  | tee "$DATA_ROOT/preflight/discovery.log"

if [[ "$CORE_COLLECTION_SCOPE" != "task_local" ]]; then
  echo "[preflight] probing system-wide core-counter permission"
  if ! perf stat -a -e cycles -- sleep 0.1 \
      > "$DATA_ROOT/preflight/system_wide_perf.stdout" \
      2> "$DATA_ROOT/preflight/system_wide_perf.stderr"; then
    echo "[fatal] System-wide perf collection is unavailable." >&2
    echo "[fatal] Inspect $DATA_ROOT/preflight/system_wide_perf.stderr and perf_event_paranoid." >&2
    exit 2
  fi
fi

UNCORE_ARGS=()
case "$COLLECT_UNCORE" in
  off) UNCORE_ARGS+=(--no-collect-uncore) ;;
  auto) ;;
  required) UNCORE_ARGS+=(--collect-uncore) ;;
  *)
    echo "[fatal] COLLECT_UNCORE must be off, auto, or required" >&2
    exit 2
    ;;
esac

COLLECTION_ARGS=(
  --sets set1,set2,set3
  --dataset-root "$DATA_ROOT"
  --workloads "$WORKLOADS"
  --reps "$REPS"
  --interval-ms "$INTERVAL_MS"
  --parsec-input "$PARSEC_INPUT"
  --set1-threads "$SET1_THREADS"
  --group-size "$GROUP_SIZE"
  --pairing-mode all_combinations
  --task-order-seed "$TASK_ORDER_SEED"
  --hybrid-threads "$HYBRID_THREADS"
  --core-collection-scope "$CORE_COLLECTION_SCOPE"
  --pipeline-config "$PIPELINE_CONFIG"
  --experiment-mode all
  "${UNCORE_ARGS[@]}"
)

echo "[preflight] materializing and validating the exact task plan"
uv run -m phase_family_ml.collect "${COLLECTION_ARGS[@]}" \
  --dataset-root "$DATA_ROOT/preflight/task_plan_dataset" \
  --dry-run \
  > "$DATA_ROOT/preflight/task_plan.json"
uv run python - "$DATA_ROOT/preflight/task_plan.json" "$WORKLOADS" "$REPS" "$SET1_THREADS" <<'PY'
import itertools
import json
import sys

plan_path, workload_text, reps_text, thread_text = sys.argv[1:]
with open(plan_path, encoding="utf-8") as handle:
    plan = json.load(handle)["task_plan"]
workloads = [item for item in workload_text.split(",") if item]
threads = [item for item in thread_text.split(",") if item]
reps = int(reps_text)
pairs = len(list(itertools.combinations(workloads, 2)))
expected = {
    "set1": len(workloads) * len(threads) * reps,
    "set2": pairs * 2 * reps,
    "set3": pairs * 2 * reps,
}
observed = {key: int(value["tasks"]) for key, value in plan["sets"].items()}
if observed != expected or int(plan["total_tasks"]) != sum(expected.values()):
    raise SystemExit(f"Unexpected task plan: expected={expected}, observed={observed}, total={plan['total_tasks']}")
print(f"Validated {plan['total_tasks']} tasks: {observed}; unordered workload pairs={pairs}")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[preflight] complete; collection skipped because PREFLIGHT_ONLY=1"
  exit 0
fi

echo "[collection] starting the complete randomized publication run"
uv run -m phase_family_ml.collect "${COLLECTION_ARGS[@]}"

for set_key in set1 set2 set3; do
  set_root="$DATA_ROOT/$set_key"
  merged="$set_root/processed/merged_interval_dataset.csv"
  results_root="$set_root/results/phase_family_ml"
  analysis_root="$set_root/analysis"
  mkdir -p "$analysis_root/paired"

  echo "[analysis] clustering stability set=$set_key"
  uv run scripts/analyze_phase_stability.py \
    --input "$merged" \
    --output "$analysis_root/phase_cluster_stability.csv" \
    --experiment-mode all \
    --k-values 2,3,4,5 \
    --seeds 3,7,11,17,23

  prediction_files=("$results_root"/phase_detector/*/global/phase_detector_predictions.csv)
  if [[ ! -e "${prediction_files[0]}" ]]; then
    echo "[fatal] No phase-detector prediction files for $set_key" >&2
    exit 2
  fi
  for predictions in "${prediction_files[@]}"; do
    experiment="$(basename "$(dirname "$(dirname "$predictions")")")"
    echo "[analysis] paired history-current bootstrap set=$set_key experiment=$experiment"
    uv run scripts/compare_phase_models.py \
      --predictions "$predictions" \
      --first-model online_phase_history_tree_all_families \
      --second-model online_current_state_tree_all_families \
      --output "$analysis_root/paired/$experiment.json" \
      --resamples 10000
  done
done

echo "[validate] requiring all planned runs and paper-facing outputs"
uv run scripts/validate_mlforsys_run.py \
  --data-root "$DATA_ROOT" \
  --workloads "$WORKLOADS" \
  --reps "$REPS" \
  --set1-threads "$SET1_THREADS" \
  | tee "$DATA_ROOT/publication_validation.log"

date -u +%Y-%m-%dT%H:%M:%SZ > "$DATA_ROOT/provenance/run_completed_utc.txt"
echo "[complete] validated real-data evidence is under $DATA_ROOT"
echo "[complete] certificate: $DATA_ROOT/publication_validation.json"
