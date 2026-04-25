# Experiment Plan

## Group A: Closed-Loop Resource Policy

Implemented now:

- Trace-driven co-scheduling replay from existing windows.
- Resource classes: memory-heavy, branch/control-heavy, mixed, and balanced.
- Policies: random, static workload, last-value, global aggregate, local-only, hybrid shared context, oracle FGMM, logistic regression, student decision tree, and transformer.
- Metrics: conflict rate, migration count, proxy weighted speedup, proxy harmonic speedup, fairness proxy, and replay overhead.

Implemented runner, results still need to be collected on Burst:

- Live Burst process-placement experiments with measured runtime, weighted speedup, harmonic speedup, and slowdown/fairness.
- CAT/resctrl LLC partitioning.
- Prefetch throttling.

Important limitation:

- Trace replay proxy metrics are not real online speedups.

## Group B: Timing-Independence Stress

Implemented collection/analyzer path, results still need to be collected on a node with frequency-control permission:

- Collect the same workloads at multiple CPU frequencies/turbo settings.
- Compare timing-independent features against timing-derived features.
- Metrics: centroid drift, phase assignment consistency, JS distance, cross-frequency generalization, and phase-change F1.

Requirement before claiming robustness:

- At least two controlled frequency settings and repeated runs.

## Group C: Generalization

Implemented now:

- Leave-one-workload-out with a nearest-centroid detector.
- Cross-setting transfer: train one setting/test others and train others/test one.

Pending data collection:

- Additional benchmark suites: SPEC CPU2017, NAS/OpenMP, or larger PARSEC inputs.

## Group D: Ablations

Implemented now:

- Feature groups: branch, L1, LLC/offcore, shared, local, all.
- Distance metrics: Manhattan, Euclidean, cosine.
- Persistence thresholds: 1, 2, 3, 5.
- Fixed-point precision: 8, 10, 12, 16 bits.
- Centroid table sizes: 4, 8, 12, 16.

Implemented offline now; live PMU multiplexing overhead is measured through the online co-scheduling run by comparing policy overhead and runtime with the selected reduced feature sets:

- Correlation-pruned feature removal.
- PMU-slot constrained schedules.
- Local/global/hybrid replay baselines.

## Group E: Online and Hardware Cost

Implemented now:

- Storage and operation estimates for centroid tables.
- Software replay overhead per decision.
- Explicit implemented-versus-estimated distinction.

Pending live collection:

- Live software overhead during workload execution.
- PMU multiplexing sensitivity.
- Vendor-specific simultaneous-event feasibility table validated on target hardware.

## Group F: Uniqueness and Equivalence Classes

Implemented now:

- Phase sharing count across workloads.
- Dominant workload share per phase.
- Workload entropy per phase.
- Resource class per phase.

Paper interpretation:

- Phase IDs are behavior classes, not program names.
