#!/usr/bin/env bash
set -euo pipefail

if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required for process-placement experiments." >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-results/online_coscheduling_core_uncore_large}
PHASE_ARTIFACT=${PHASE_ARTIFACT:-results/phase_ml_core_uncore_large}
WORKLOADS=${WORKLOADS:-blackscholes,canneal,fluidanimate,freqmine,swaptions,streamcluster}
THREADS=${THREADS:-1,2}
REPS=${REPS:-5}
PARSEC_INPUT=${PARSEC_INPUT:-simsmall}
POLICIES=${POLICIES:-linux_default,random_placement,spread_physical,static_workload,phase_aware_oracle,phase_aware_student,phase_aware_transformer}
GROUP_MODE=${GROUP_MODE:-all_pairs}
GROUP_SIZE=${GROUP_SIZE:-2}

python -m experiments.run_online_coscheduling \
  --phase-artifact "${PHASE_ARTIFACT}" \
  --output-dir "${OUTPUT_DIR}" \
  --workloads "${WORKLOADS}" \
  --threads "${THREADS}" \
  --group-size "${GROUP_SIZE}" \
  --group-mode "${GROUP_MODE}" \
  --policies "${POLICIES}" \
  --reps "${REPS}" \
  --parsec-input "${PARSEC_INPUT}"

python -m analysis.online_coscheduling \
  --input-csv "${OUTPUT_DIR}/online_runs.csv" \
  --output-dir "${OUTPUT_DIR}/analysis"

echo "Online co-scheduling results:"
echo "  ${OUTPUT_DIR}/online_runs.csv"
echo "  ${OUTPUT_DIR}/analysis/online_policy_summary.csv"
