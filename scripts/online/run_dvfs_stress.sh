#!/usr/bin/env bash
set -euo pipefail

# Wrapper for fixed-frequency data collection and DVFS-stability analysis.
FREQUENCIES_KHZ=${FREQUENCIES_KHZ:?Set FREQUENCIES_KHZ, for example FREQUENCIES_KHZ=1200000,2200000}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/dvfs_stress_core_uncore_large}
WORKLOADS=${WORKLOADS:-blackscholes,canneal,fluidanimate,freqmine,swaptions,streamcluster}
SETS=${SETS:-set1,set2,set3}
REPS=${REPS:-3}
INTERVAL_MS=${INTERVAL_MS:-10}
PARSEC_INPUT=${PARSEC_INPUT:-simsmall}
SET1_THREADS=${SET1_THREADS:-2,4,8}
GROUP_SIZE=${GROUP_SIZE:-2}
HYBRID_THREADS=${HYBRID_THREADS:-2}
CORE_COLLECTION_SCOPE=${CORE_COLLECTION_SCOPE:-system_wide_physical_core}
COLLECT_UNCORE=${COLLECT_UNCORE:-0}
PHASE_ML_CONFIG=${PHASE_ML_CONFIG:-config/phase_ml_indep_a100.json}

args=(
  --frequencies-khz "${FREQUENCIES_KHZ}"
  --output-root "${OUTPUT_ROOT}"
  --workloads "${WORKLOADS}"
  --sets "${SETS}"
  --reps "${REPS}"
  --interval-ms "${INTERVAL_MS}"
  --parsec-input "${PARSEC_INPUT}"
  --set1-threads "${SET1_THREADS}"
  --group-size "${GROUP_SIZE}"
  --hybrid-threads "${HYBRID_THREADS}"
  --core-collection-scope "${CORE_COLLECTION_SCOPE}"
  --phase-ml-config "${PHASE_ML_CONFIG}"
)

if [[ "${COLLECT_UNCORE}" == "1" ]]; then
  args+=(--collect-uncore)
fi

python -m experiments.run_dvfs_stress "${args[@]}"

artifact_args=()
IFS=',' read -r -a freq_items <<< "${FREQUENCIES_KHZ}"
for freq in "${freq_items[@]}"; do
  artifact_args+=(--artifact "${freq}khz:${OUTPUT_ROOT}/phase_ml_${freq}khz")
done

python -m analysis.dvfs_stress "${artifact_args[@]}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

echo "DVFS outputs are under ${OUTPUT_ROOT}."
echo "DVFS analysis summary: ${OUTPUT_ROOT}/analysis/dvfs_stress_summary.json"
