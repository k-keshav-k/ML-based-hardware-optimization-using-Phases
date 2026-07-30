# MLForSys 2026 submission readiness

## Target and format

- Target: Machine Learning for Systems workshop at NeurIPS 2026.
- Deadline checked on 2026-07-29: August 29, 2026, 23:59 AoE.
- Limit: four content pages, excluding references and optional appendix.
- Review: non-anonymous submissions are permitted.
- Publication: no formal proceedings; accepted papers may be submitted later to an archival venue.
- Template: official NeurIPS 2026 `sglblindworkshop` style.

The workshop manuscript is `paper/mlforsys2026.tex`; the built submission PDF is
`paper/mlforsys2026.pdf`. The ISPASS manuscript remains a separate archival-paper
draft.

## Scientific framing

The workshop paper is an evaluation-methodology contribution for ML for systems. Its
central claims are:

1. Complete executions and co-running processes, not intervals, are the split and
   resampling unit.
2. Persistence and Markov models that receive the offline teacher's current label are
   oracle diagnostics, not deployable baselines.
3. A fair current-state tree isolates the value of counter history.
4. Transition-event metrics, PMU enabled time, clustering stability, and complete
   online state must accompany accuracy.

The manuscript does not claim energy savings, synthesized hardware area, or a
controller benefit.

## Evidence status

The PDF reports a reproducible 60-run pilot as preliminary evidence. It explicitly
reports the small test-group and transition counts, workload-holdout failure case,
clustering sensitivity, and PMU multiplexing limitations. These limitations are
appropriate for a nonarchival workshop extended abstract, but the corrected real-data
run should replace or supplement the pilot if it completes before submission.

Run the full study on a bare-metal PMU-capable host:

```bash
bash scripts/run_mlforsys2026_real_data.sh
```

All large state defaults to `/scratch/kk6081`. For a preflight without collection:

```bash
PREFLIGHT_ONLY=1 bash scripts/run_mlforsys2026_real_data.sh
```

The locked default plan contains 630 execution groups and 1,050 measured process
tasks: 210 single-process runs across seven workloads and three thread counts, plus
all 21 unordered workload pairs in both the two-single-thread and two-four-thread
co-runner regimes, each with ten repetitions. The preflight independently derives
and checks these counts before collection.

The current execution environment does not expose the required core PMUs or uncore
devices. That is expected to make preflight fail; it prevents synthetic or incomplete
data from being mistaken for a publication run.

Successful completion writes:

```text
/scratch/kk6081/mlforsys2026_real_data/publication_validation.json
/scratch/kk6081/mlforsys2026_real_data/set{1,2,3}/processed/merged_interval_dataset.csv
/scratch/kk6081/mlforsys2026_real_data/set{1,2,3}/processed/pmu_collection_quality.csv
/scratch/kk6081/mlforsys2026_real_data/set{1,2,3}/results/phase_family_ml/phase_detector/phase_detector_summary_all.csv
/scratch/kk6081/mlforsys2026_real_data/set{1,2,3}/analysis/phase_cluster_stability.csv
/scratch/kk6081/mlforsys2026_real_data/set{1,2,3}/analysis/paired/*.json
```

## Required human confirmations

- Confirm that Keshav Krishna's affiliation is New York University.
- Confirm the complete author list; Mohamed Zahran is not added automatically merely
  because the original report names him as supervisor.
- Replace the personal Gmail address with an institutional address if available.
- Confirm that every listed author approves the submitted manuscript.
- Check the workshop's current author/AI-disclosure policy immediately before upload.

No technical or editorial process can guarantee acceptance. Reviewer assignment and
judgment remain outside the authors' control.
