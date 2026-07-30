# ISPASS 2027 submission plan and readiness audit

## Decision

Target **ISPASS 2027 main track**. The paper is a closer match to ISPASS than to an
archival IISWC poster because its strongest defensible contribution is a performance-
analysis method: reducing a multiplexed PMU interface, defining train-only global
pressure phases, and quantifying the accuracy/transition/cost trade-off of an online
predictor. ISPASS explicitly solicits performance-counter methods, statistical
performance analysis, multicore systems, and power/efficiency evaluation.

ISPASS 2027 has not published its call or dates as of 2026-07-29. The most recent
rules (ISPASS 2026) allowed nine IEEE conference pages plus unlimited references and
required double-blind review:
https://ispass.org/ispass2026/submission.php. The 2026 HotCRP page reports 54 accepted
papers from 138 submissions (39.1%): https://ispass2026.hotcrp.com/. That is a useful
historical reference, not a guaranteed or predicted 2027 acceptance rate.

Do **not** submit the present result as the IISWC 2026 archival short paper merely to
meet an earlier deadline. The short-paper track is only four pages including
references and appears in the proceedings, which can complicate a later full paper:
https://iiswc.org/iiswc2026/cfposter.html. A two-page non-archival IISWC WiP poster is
compatible with later development if author feedback is worth the August 3 deadline.

## Submission claim

Use this claim:

> A train-only full-counter teacher can be approximated online with one selected
> counter per resource family and a small history tree. The reduced interface improves
> PMU scheduling quality and preserves global-pressure prediction, while history
> materially improves transition sensitivity over an equally deployable current-state
> predictor.

Do not claim an end-to-end power or cache saving until it has been measured with a
real controller or simulator. Remove the existing L1 “power factor” proxy from the
main result; its factors are not calibrated to cache capacity, energy, or performance.

## Current go/no-go status

The code and paper are **not yet submission-ready**, for evidence reasons rather than
formatting reasons. No conference decision can be guaranteed. The following are hard
submission gates.

| Gate | Current state | Required evidence |
|---|---|---|
| Final artifacts accessible | Blocked in this environment | `/scratch/kk6081/finals_dataset/set{1,2,3}` or equivalent must be visible and checksummed. |
| Leakage-free split | Implemented in code; final rerun pending | All processes in one `concurrent_group_id`, all repetitions of a configuration, and workload holdouts must remain in one split. |
| Pair coverage | Collector fixed; recollection pending | Use all 21 unordered pairs for seven workloads, not four fixed chunked pairs. Report pair count and repetitions. |
| PMU quality | Audit implemented; final audit pending | Report mean, p05, minimum enabled percentage, and fraction below 90% for every used event. Reduce event groups if p05 is poor. |
| Statistical uncertainty | Implemented; final rerun pending | Execution-group bootstrap CIs and paired CIs, at least 10,000 resamples for final tables. |
| Honest baselines | Implemented | Label teacher-phase persistence/Markov/duration models as oracle diagnostics. Compare primarily against the deployable current-state tree. |
| Robustness | Implemented; final rerun pending | `k=2..5`, at least five seeds, history `H={1,5,10,20,40}`, tree depth, and interval sensitivity. |
| Generalization | Pending | Configuration-disjoint, workload-disjoint, and unseen co-runner-pair results. Prefer a second processor generation. |
| Utility or collection benefit | Pending | Measure full-profile versus selected-profile enabled time, wall-clock overhead, and label fidelity. A real controller is optional only if this measurement contribution is strong. |
| Hardware cost | Corrected in code | Include the tree, 2-bit history buffer, and 16-bit discretizer thresholds; do not report tree bytes alone. |
| Reproducibility | In progress | Anonymous artifact, one-command scripts, software/platform manifest, raw-data checksums, and deterministic seeds. |

Every author must verify every number, citation, and interpretation. Recheck the ISPASS
2027 policy on generative-AI assistance when it is published and include any required
disclosure in a form that preserves double-blind review; do not copy a 2026 policy
blindly.

## Required final experiment matrix

1. **Multicore regimes.** Retain single-process multithreaded execution and collect all
   unordered two-workload pairs for the one-thread-per-process and four-thread-per-
   process regimes. Randomize execution order and use at least five independent
   repetitions; ten is preferable.
2. **PMU profiles.** For a balanced subset, run (a) no `perf`, (b) the full safe-counter
   teacher profile, and (c) the validation-frozen one-counter-per-family profile.
   Report runtime overhead and enabled percentage. Freeze event selection before the
   test run.
3. **Splits.** Report configuration-group holdout as the primary result. Add
   leave-one-workload-out and leave-one-pair-out results. The unit of resampling and
   splitting is the complete concurrent execution group, never a 10 ms row.
4. **Sensitivity.** Evaluate `k={2,3,4,5}`, five or more k-means initializations,
   `H={1,5,10,20,40}`, interval lengths such as 5/10/20 ms if recollection cost permits,
   and tree depths 3--8. Selection decisions use train/validation only.
5. **Metrics.** Report accuracy, macro F1, balanced accuracy, high-pressure recall,
   transition precision/recall/F1, transition accuracy, detection delay, false alarms,
   and execution-group bootstrap 95% CIs. Include phase shares and transition counts so
   persistence cannot inflate the headline.
6. **Cost.** Report PMU event slots, enabled percentage, bytes, comparisons, measured
   software inference time, and collection overhead. If RTL synthesis is unavailable,
   call the byte/operation numbers analytical estimates and avoid “hardware
   implementation” language.

## Final-run commands

Keep generated data and predictions on scratch:

```bash
scratch_root=/scratch/kk6081/ispass2027/final
mkdir -p "$scratch_root"

python scripts/merge_runs.py \
  --input-dir /scratch/kk6081/finals_dataset/set1/raw \
  --output-dir "$scratch_root/set1/processed" \
  --manifest-path /scratch/kk6081/finals_dataset/set1/raw/phase_family_ml_experiment_manifest.json

python -m phase_family_ml.run_pipeline \
  --config config/ispass2027_validation.json \
  --input "$scratch_root/set1/processed/merged_interval_dataset.csv" \
  --output-dir "$scratch_root/set1/results" \
  --experiment-mode config_group_holdout

python scripts/analyze_phase_stability.py \
  --input "$scratch_root/set1/processed/merged_interval_dataset.csv" \
  --output "$scratch_root/set1/phase_cluster_stability.csv" \
  --experiment-mode config_group_holdout \
  --k-values 2,3,4,5 \
  --seeds 3,7,11,17,23

python scripts/compare_phase_models.py \
  --predictions "$scratch_root/set1/results/phase_detector/config_group_holdout/global/phase_detector_predictions.csv" \
  --first-model online_phase_history_tree_all_families \
  --second-model online_current_state_tree_all_families \
  --output "$scratch_root/set1/history_vs_current_paired_bootstrap.json" \
  --resamples 10000
```

Repeat merging and analysis for Sets 2 and 3. Run `leave_one_workload_out` as a
separate output directory rather than overwriting the primary run. Never place raw
datasets, prediction CSVs, caches, or virtual environments in the repository/home
filesystem.

## Paper structure for nine pages

1. Introduction and three precise contributions: 1 page.
2. Motivation and related work: 0.75 page.
3. Resource-family interface and leakage-free method: 1.5 pages.
4. Experimental methodology and validity controls: 1.25 pages.
5. Main accuracy/transition/generalization results: 2 pages.
6. PMU quality, overhead, and analytical cost: 1 page.
7. Sensitivity, limitations, and implications: 1 page.
8. Conclusion: 0.25 page. References are outside the nine-page limit under the 2026
   rules; recheck the ISPASS 2027 CFP when it appears.

## Recommended schedule

- August 2026: restore/checksum final data; recollect all workload pairs and reduced-
  profile overhead runs.
- September: complete split, seed, `k`, history, interval, and second-platform studies.
- October: freeze all numbers and figures; internal adversarial review.
- November: anonymous artifact dry run on a clean account/machine and paper revision.
- When ISPASS 2027 announces dates: revalidate page limit, anonymity, artifact, and
  conflict rules immediately. Do not assume the 2026 deadline or format is unchanged.
