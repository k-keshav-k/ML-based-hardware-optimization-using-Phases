# Reviewer Attack Checklist

## 1. What is new versus prior phase work?

The current novelty is not “phase detection.” The defensible novelty is the combination of timing-independent PMU ratios, multicore per-core signatures, phase-change detection as the control signal, phase sharing as behavioral equivalence classes, and a resource-policy path. Online resource-management claims remain gated on the implemented live co-scheduling run producing measured results.

## 2. Why care about timing-independent counters?

Cycles, IPC, CPI, elapsed time, and per-time rates can change with DVFS, thermal throttling, scheduling, and measurement interval artifacts. Timing-independent ratios aim to preserve behavioral signatures under those changes. This claim needs DVFS stress results before it can be stated strongly.

## 3. Why not use IPC/CPI/cycles?

They are useful performance symptoms but poor phase-signature inputs when the goal is robustness to timing instability. The artifact excludes them by design. The pending stress test must quantify the tradeoff.

## 4. Why is weak next-phase accuracy acceptable?

Only because the paper reframes the goal as phase-change detection. Exact next-phase accuracy is weak and must be reported as a negative result. A resource controller needs stable change signals more than exact unsupervised phase IDs.

## 5. Does phase-change F1 translate into better resource decisions?

Not proven online unless `online_policy_summary.csv` is present. The trace replay shows a plausible path and proxy conflict reduction; real speedup/fairness claims require the implemented live co-scheduling runner or a platform-specific CAT/prefetch/bandwidth-control experiment.

## 6. Why is local/per-core better than global?

The literature motivates local views for asynchronous multicore behavior. The artifact backs this with shared-only, local-only, all-feature, and local/global/hybrid replay ablations; final claims should use whichever variant wins after the online run.

## 7. What does shared context add?

Shared context captures LLC/offcore/uncore pressure that local counters can miss. The ablation script includes shared-only and local-only feature groups; the paper should report whether shared context helps or hurts.

## 8. Is this robust across workloads, runs, and settings?

Grouped-by-run split avoids run leakage. Leave-one-workload-out and cross-setting transfer are implemented as stress tests. Any weak transfer results must be reported.

## 9. Is the hardware cost realistic?

The cost table estimates centroid storage and operations per window. It is not a hardware implementation. The paper must separate software prototype, trace replay, and possible hardware/firmware implementation.

## 10. Can this run online without perturbing workloads?

Not yet proven. The replay is offline. Live overhead must be measured on Burst or another controlled host.

## 11. What about PMU multiplexing or limited counters?

The full feature set likely exceeds common programmable counter slots. Offline PMU-slot constrained feature schedules are implemented; live multiplexing overhead still needs a hardware run.

## 12. Is it learning benchmark names?

The uniqueness analysis reports phase sharing, workload entropy, and dominant workload share. Shared phases across workloads support the behavior-class interpretation; workload-specific phases remain a limitation.

## 13. Are selected counters portable?

The selected families are broadly available, but exact event names differ. The paper should speak in counter families and provide vendor examples, not claim exact portability without validation.

## 14. What are the negative results?

- Exact next-phase accuracy is weak.
- Online speedup is not measured yet.
- DVFS robustness is not measured yet.
- Offline PMU-slot feasibility is implemented; live multiplexing overhead is pending.
- Workload/setting transfer may fail for some held-out cases.
- CAT/resctrl and prefetch controls are outside the current claim set unless a target node exposes and logs those controls.
