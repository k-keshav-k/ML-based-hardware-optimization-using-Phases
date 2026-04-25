# A* Gap Analysis: Timing-Independent PMU Phase Signatures

## Current Strengths

- The artifact contains a real PARSEC multicore dataset: 300 merged runs, 110,785 intervals, 58,777 windows, 15 timing-independent features, physical-core streams, and uncore IMC context.
- The current split is grouped by run ID. Inspection found 296 window-producing runs and zero runs appearing in more than one split.
- The feature pipeline explicitly excludes cycles, IPC, CPI, elapsed time, per-time rates, and stall-derived fields.
- The full GPU artifact includes baselines, transformer teacher predictions, student decision-tree predictions, per-workload metrics, confusion matrices, and summaries.
- Existing results are directionally useful: exact next-phase prediction is weak, but phase-change prediction is much stronger.

## Fatal Weaknesses

- The current paper does not yet demonstrate a real resource-management benefit. That is fatal for ASPLOS/MICRO/HPCA/ISCA if the paper claims to be a systems mechanism.
- The current PDF still reads like a report: phrases such as “this report,” “semester,” “imaginative,” and “future work should” weaken credibility.
- The previous draft overclaims online/hardware readiness. The implemented artifact is software analysis and trace replay, not a hardware implementation.
- The old draft contains correctness bugs: physical-core data described as task-local, “all five repetitions” despite 10 reps, duplicated text, duplicated table row, and cramped plots.
- The closed-loop policy results must be trace-driven unless live Burst experiments are actually run.

## Missing Experiments

- Live phase-aware co-scheduling or placement with measured speedup/fairness.
- Timing-independence stress under DVFS/frequency/turbo changes.
- Local versus global versus hybrid detector comparison.
- Leave-one-workload-out and cross-setting transfer.
- Feature-group, distance-metric, persistence, centroid-count, fixed-point, and PMU-slot ablations.
- PMU multiplexing overhead with live perf collection. Offline limited-counter-slot sensitivity is implemented.
- Additional benchmark suites such as SPEC CPU2017, NAS, or OpenMP workloads.

## Missing Baselines

- Linux/default scheduling and random placement for live runs.
- Static workload-level classification.
- Last-value phase predictor as policy input.
- Global aggregate phase detector.
- Local-only detector.
- Hybrid detector with shared context.
- Oracle/offline FGMM labels.
- Reduced-counter policies constrained to typical programmable PMU slots.

## Novelty Risks

- SimPoint and BBV work already established phase clustering.
- Sherwood runtime phase tracking already showed small online phase tables.
- Perelman and Chang already argued for per-thread/per-core treatment in parallel workloads.
- P4 and phase-aware forecasting already use PMCs and learning for phase-aware prediction.
- The novelty must therefore be narrowed to: timing-independent multicore PMU signatures, phase-change detection as the control signal, behavioral equivalence classes, and a low-cost resource-policy path.

## Artifact and Reproducibility Risks

- Full raw collection may not be available in-repo; some artifacts are generated under ignored `results/`.
- Transformer training requires GPU.
- Live online policy evaluation requires machine-specific privileges and CPU affinity control.
- CAT/resctrl is not locally available unless the target node exposes `/sys/fs/resctrl`.
- DVFS/frequency experiments require cpufreq permissions and careful thermal controls.
- Current generated paper artifacts are not enough; a clean `paper/`, `artifact/`, `analysis/`, `experiments/`, and `pipeline/` structure is needed.

## Exact Action Plan

1. Keep the current measured detection results, but demote broad claims.
2. Add trace-driven policy replay and label all proxy metrics as trace-driven, not measured speedup.
3. Run the implemented online Burst co-scheduling driver and import the measured summary into the paper pipeline.
4. Add validation scripts for leakage, manifest consistency, feature policy, label consistency, and metrics consistency.
5. Add uniqueness, generalization, ablation, and hardware-cost analyses.
6. Generate `paper/main.tex` as a conference-style draft with evidence-gated claims.
7. Add reviewer-response documentation and artifact reproduction instructions.
8. Run the pipeline and tests before treating the draft as usable.
