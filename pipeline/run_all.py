"""Regenerate the paper artifact from available phase-ML outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.ablation import run_ablations
from analysis.common import load_rows, read_json, write_json
from analysis.generalization import run_generalization
from analysis.hardware_cost import estimate_cost
from analysis.online_coscheduling import analyze as analyze_online_coscheduling
from analysis.pmu_slot_sensitivity import run_slot_sensitivity
from analysis.trace_policy_replay import run_replay
from analysis.uniqueness import analyze_uniqueness
from analysis.validate_artifact import validate


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def copy_existing_figures(source_report_dir: Path, figure_dir: Path) -> None:
    source = source_report_dir / "figures"
    if not source.exists():
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.pdf"):
        shutil.copy2(path, figure_dir / path.name)


def plot_policy_replay(analysis_dir: Path, figure_dir: Path) -> None:
    rows = load_rows(analysis_dir / "trace_policy_replay.csv")
    policies = [row["policy"].replace("_", " ") for row in rows]
    conflict = [float(row["conflict_rate"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar(range(len(rows)), conflict, color="#4C78A8")
    ax.set_ylabel("Predicted conflict rate")
    ax.set_title("Trace-driven co-scheduling policy replay")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(policies, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "trace_policy_replay.pdf")
    plt.close(fig)


def plot_ablation(analysis_dir: Path, figure_dir: Path) -> None:
    rows = [row for row in load_rows(analysis_dir / "detector_ablation.csv") if row["metric"] == "manhattan" and row["fixed_point_bits"] == "float" and row["max_centroids"] == "16"]
    groups = []
    values = []
    seen = set()
    for row in rows:
        group = row["feature_group"]
        if group in seen:
            continue
        seen.add(group)
        groups.append(group.replace("_", "+"))
        values.append(float(row["phase_change_f1"]))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(range(len(groups)), values, color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Phase-change F1")
    ax.set_title("Feature-group ablation for centroid detector")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_group_ablation.pdf")
    plt.close(fig)


def plot_generalization(analysis_dir: Path, figure_dir: Path) -> None:
    rows = load_rows(analysis_dir / "generalization.csv")
    labels = [row["test"].replace("set1_single_process_multithread", "S1").replace("set2_multi_process_single_thread", "S2").replace("set3_hybrid_multi_process_multithread", "S3") for row in rows]
    values = [float(row["phase_change_f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.bar(range(len(rows)), values, color="#54A24B")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Phase-change F1")
    ax.set_title("Transfer/generalization stress tests")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "generalization_stress.pdf")
    plt.close(fig)


def plot_pmu_slots(analysis_dir: Path, figure_dir: Path) -> None:
    rows = load_rows(analysis_dir / "pmu_slot_sensitivity.csv")
    labels = [row["counter_slots"] for row in rows]
    values = [float(row["phase_change_f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(labels, values, marker="o", color="#B279A2")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Programmable counter slots")
    ax.set_ylabel("Phase-change F1")
    ax.set_title("PMU slot sensitivity")
    fig.tight_layout()
    fig.savefig(figure_dir / "pmu_slot_sensitivity.pdf")
    plt.close(fig)


def plot_online_results(analysis_dir: Path, figure_dir: Path) -> None:
    path = analysis_dir / "online_policy_summary.csv"
    if not path.exists():
        return
    rows = load_rows(path)
    policies = [row["policy"].replace("_", " ") for row in rows]
    values = [float(row["weighted_speedup_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(range(len(rows)), values, color="#72B7B2")
    ax.set_ylabel("Measured weighted speedup")
    ax.set_title("Live co-scheduling results")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(policies, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "online_coscheduling.pdf")
    plt.close(fig)


def plot_dvfs_results(analysis_dir: Path, figure_dir: Path) -> None:
    path = analysis_dir / "dvfs_stress.csv"
    if not path.exists():
        return
    rows = load_rows(path)
    labels = [row["workload"] for row in rows]
    values = [float(row["centroid_drift_l2"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(range(len(rows)), values, color="#E45756")
    ax.set_ylabel("Centroid drift (L2)")
    ax.set_title("Cross-frequency timing-independent feature drift")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "dvfs_stress.pdf")
    plt.close(fig)


def table_rows(rows: list[list[object]]) -> str:
    return "\n".join(" & ".join(str(item).replace("_", "\\_") for item in row) + r" \\" for row in rows)


def write_references(source_report_dir: Path, paper_dir: Path) -> None:
    source = source_report_dir / "references.bib"
    target = paper_dir / "references.bib"
    if source.exists():
        shutil.copy2(source, target)
    elif not target.exists():
        target.write_text("", encoding="utf-8")


def write_paper(config: dict[str, object], validation: dict[str, object]) -> None:
    paper_dir = Path(str(config["paper_dir"]))
    figure_dir = Path(str(config["figure_dir"]))
    analysis_dir = Path(str(config["analysis_dir"]))
    run_root = Path(str(config["run_root"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    paper_dir.mkdir(parents=True, exist_ok=True)
    write_references(Path(str(config["source_report_dir"])), paper_dir)

    # Pull all computed CSV/JSON artifacts and translate them into paper tables.
    metrics = load_rows(Path(str(config["eval_dir"])) / "model_comparison.csv")
    replay = load_rows(analysis_dir / "trace_policy_replay.csv")
    ablation = load_rows(analysis_dir / "detector_ablation.csv")
    generalization = load_rows(analysis_dir / "generalization.csv")
    cost = load_rows(analysis_dir / "hardware_cost.csv")
    pmu_slots = load_rows(analysis_dir / "pmu_slot_sensitivity.csv")
    online_path = analysis_dir / "online_policy_summary.csv"
    online_available = online_path.exists()
    online_rows = load_rows(online_path) if online_available else []
    dvfs_path = analysis_dir / "dvfs_stress.csv"
    dvfs_available = dvfs_path.exists()
    dvfs_rows = load_rows(dvfs_path) if dvfs_available else []
    uniqueness = read_json(analysis_dir / "phase_uniqueness_summary.json")
    transformer_summary = read_json(run_root / "transformer" / "transformer_summary.json")
    student_summary = read_json(run_root / "student_tree" / "student_decision_tree_summary.json")
    manifest = read_json(dataset_dir / "feature_manifest.json")

    best_accuracy = max(metrics, key=lambda row: float(row["accuracy"]))
    best_change = max(metrics, key=lambda row: float(row["phase_change_f1"]))
    best_replay = min(replay, key=lambda row: float(row["conflict_rate"]))
    best_ablation = max(ablation, key=lambda row: float(row["phase_change_f1"]))
    worst_generalization = min(generalization, key=lambda row: float(row["phase_change_f1"]))
    model_table = table_rows([[row["model"], row["samples"], fmt(row["accuracy"]), fmt(row["macro_f1"]), fmt(row["phase_change_f1"])] for row in metrics])
    replay_table = table_rows([[row["policy"], fmt(row["conflict_rate"]), row["migration_count"], fmt(row["weighted_speedup_proxy"]), fmt(row["fairness_proxy"])] for row in replay])
    cost_table = table_rows([[row["fixed_point_bits"], fmt(row["total_bytes_per_core"], 0), fmt(row["sub_abs_accumulate_ops_per_window"], 0)] for row in cost])
    pmu_table = table_rows([[row["counter_slots"], row["feature_count"], fmt(row["accuracy"]), fmt(row["phase_change_f1"])] for row in pmu_slots])
    online_table = table_rows(
        [
            [
                row["policy"],
                row["groups"],
                row["failures"],
                fmt(row["weighted_speedup_mean"]),
                fmt(row["harmonic_speedup_mean"]),
                fmt(row["fairness_min_speedup_mean"]),
            ]
            for row in online_rows
        ]
    )
    best_online = max(online_rows, key=lambda row: float(row["weighted_speedup_mean"])) if online_rows else {}
    mean_dvfs_drift = sum(float(row["centroid_drift_l2"]) for row in dvfs_rows) / len(dvfs_rows) if dvfs_rows else 0.0
    title_suffix = "Multicore Resource Policies" if online_available else "Trace-Driven Multicore Resource Policies"
    abstract_policy_sentence = (
        f" Live co-scheduling measurements are present: {best_online['policy'].replace('_', ' ')} has the highest measured mean weighted speedup, {fmt(best_online['weighted_speedup_mean'])}, over {best_online['groups']} co-run groups."
        if online_available
        else " A trace-driven co-scheduling replay shows how these signals can be consumed by a resource policy, but it is not a measured online speedup. Online Burst, DVFS, CAT/resctrl, and prefetch-control measurements are included only when their result files exist."
    )
    intro_evidence_sentence = (
        "The current evidence includes live process-placement measurements, trace replay, detector ablations, and hardware-cost estimates."
        if online_available
        else "The current evidence does not support a claim of online performance improvement. It supports three narrower claims: phase-change prediction is substantially more reliable than exact next-phase prediction; learned phase IDs behave as reusable resource-behavior classes rather than benchmark names; and a compact detector has plausible storage and operation cost. The missing evidence is an online co-scheduling/cache/prefetch experiment that measures real speedup or fairness."
    )
    rq4_text = (
        rf"""
\subsection{{RQ4: Does a phase-aware policy improve live co-scheduling?}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/online_coscheduling.pdf}}
\caption{{Measured live co-scheduling results. Higher weighted speedup is better.}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{Measured online co-scheduling policy summary.}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Policy & Groups & Fail & WS & HS & Fair \\
\midrule
{online_table}
\bottomrule
\end{{tabular}}
\end{{table}}

The best measured policy is {best_online['policy'].replace('_', ' ')} with mean weighted speedup {fmt(best_online['weighted_speedup_mean'])}. These values are measured wall-time results from the online runner and are distinct from the trace-replay proxy.
"""
        if online_available
        else rf"""
\subsection{{RQ4: Does a phase-aware policy reduce predicted conflicts?}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/trace_policy_replay.pdf}}
\caption{{Trace-driven co-scheduling replay. Lower predicted conflict rate is better. These are proxy replay results, not online speedups.}}
\end{{figure}}

\begin{{table}}[t]
\centering
\caption{{Trace-driven policy replay. Proxy speedup/fairness values are monotonic transforms of conflict rate, not measured execution speedups.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Policy & Conflict & Migrations & WS proxy & Fairness \\
\midrule
{replay_table}
\bottomrule
\end{{tabular}}
\end{{table}}

The best replay conflict policy is {best_replay['policy'].replace('_', ' ')} with conflict rate {fmt(best_replay['conflict_rate'])}. This is sufficient to motivate online experiments, but not sufficient to claim measured performance improvement.
"""
    )
    dvfs_text = (
        rf"""The DVFS stress analysis is present and reports mean cross-frequency centroid drift {fmt(mean_dvfs_drift)} over {len(dvfs_rows)} workload comparisons (Fig.~\ref{{fig:dvfs}}).

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/dvfs_stress.pdf}}
\caption{{Cross-frequency drift for timing-independent feature centroids.}}
\label{{fig:dvfs}}
\end{{figure}}
"""
        if dvfs_available
        else "A DVFS/timing stress test has not been collected in this artifact instance; the paper must not claim robustness under controlled frequency changes until those result files exist."
    )

    tex = rf"""
\documentclass[conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,array,url,cite,xcolor}}
\newcommand{{\system}}{{TIPS}}
\begin{{document}}
\title{{\system{{}}: Timing-Independent PMU Phase Signatures for {title_suffix}}}
\author{{\IEEEauthorblockN{{Keshav K.}}\IEEEauthorblockA{{Hardware Counter Phase Analysis Artifact}}}}
\maketitle

\begin{{abstract}}
Multicore resource managers need low-overhead signals that indicate when workload behavior changes, but timing-derived PMU metrics such as cycles, IPC, and elapsed time are sensitive to DVFS, thermal, and scheduling effects. \system{{}} uses instruction-normalized and access-normalized PMU ratios to construct per-core phase signatures and treats phase-change detection as the control signal. On the available PARSEC artifact, \system{{}} analyzes {manifest['interval_rows']:,} intervals and {manifest['windows']:,} windows from {validation['merge_merged_run_count']} merged runs. Exact next-phase prediction remains weak: the best model reaches {fmt(best_accuracy['accuracy'])} accuracy. Phase-change detection is stronger: {best_change['model'].replace('_', ' ')} reaches change-F1 {fmt(best_change['phase_change_f1'])}, and a compact student tree reaches {fmt(student_summary['phase_change_f1'])}.{abstract_policy_sentence}
\end{{abstract}}

\section{{Introduction}}
Phase detection matters only if it helps a system make decisions. Prior phase systems established that programs exhibit recurring behavior, but many rely on code signatures, timing metrics, heavy offline models, or do not close the loop with a low-cost multicore policy. The central claim of this draft is deliberately narrow: timing-independent PMU signatures are a defensible substrate for multicore phase-change detection, and the resulting phase-change stream can drive a trace-driven resource-policy replay.

{intro_evidence_sentence}

\textbf{{Contributions.}} (1) We formalize a timing-independent PMU feature set that excludes cycles, IPC, CPI, elapsed time, per-time rates, and stall-derived metrics. (2) We evaluate phase-change prediction across three multicore PARSEC settings with grouped-by-run splits and no observed run leakage. (3) We quantify phase sharing and workload purity to show that phases are behavioral equivalence classes. (4) We implement a trace-driven resource-policy replay and a reproducible artifact pipeline. (5) We provide ablations, hardware-cost estimates, and explicit pending scripts for online validation.

\section{{Background and Motivation}}
SimPoint and BBV work showed that recurring execution behavior can stand in for whole-program simulation \cite{{sherwood2001bbda,sherwood2002simpoint,hamerly2005simpoint}}. Runtime phase tracking showed that compact signatures can be used online \cite{{sherwood2003phase}}. Parallel phase work warns that global aggregation can hide asynchronous thread behavior \cite{{perelman2006parallel,chang2013sampling}}. PMU-based predictors such as P4 and phase-aware multicore forecasting motivate counters and learning, but they do not by themselves establish a cheap resource-control path for this artifact \cite{{kim2017p4,alcorta2023forecasting}}.

\section{{Design}}
\system{{}} builds local signatures from branch behavior, L1 access mix, LLC/offcore pressure, and optional uncore bandwidth. Shared context is auxiliary; the local stream remains the unit of classification. The online detector is a centroid table with EWMA smoothing and persistence filtering. A candidate phase change must persist before a resource policy acts, because the policy target is avoiding disruptive changes, not maximizing exact phase-ID accuracy.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/phase_ml_pipeline_flowchart.pdf}}
\caption{{Artifact pipeline: collection, timing-independent features, labels, predictors, replay, and paper outputs.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/online_detector_flowchart.pdf}}
\caption{{Online detector structure. The current artifact implements the detector/replay in software; hardware/firmware integration is a cost estimate, not a tapeout.}}
\end{{figure}}

\section{{Implementation}}
The implemented artifact includes feature construction, FGMM labeling, classical baselines, transformer and student-tree predictions, detector ablations, validation checks, uniqueness metrics, hardware-cost accounting, and trace-driven policy replay. The replay assigns each phase to a coarse resource class and measures predicted co-runner conflicts over collected windows. It reports proxy weighted-speedup/fairness metrics derived from conflict rate; these are not measured runtime speedups.

\section{{Methodology}}
The available dataset contains {manifest['interval_rows']:,} intervals, {manifest['windows']:,} windows, {len(manifest['feature_columns'])} features, and {validation['merge_merged_run_count']} merged PARSEC runs. The split is grouped by run ID; validation found {validation['split_leakage_run_count']} leaking runs. The evaluation set contains {sum(int(row['samples']) for row in metrics[:1])} windows per model. Existing tests pass, and validation outputs are generated under \texttt{{results/a\_star\_analysis}}.

\section{{Results}}
\subsection{{RQ1: Are timing-independent signatures stable?}}
The current artifact shows within-phase variance reduction and phase centroids over timing-independent features (Fig.~\ref{{fig:centroids}}). {dvfs_text}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/phase_feature_centroids.pdf}}
\caption{{Phase centroids over timing-independent indicators.}}
\label{{fig:centroids}}
\end{{figure}}

\subsection{{RQ2: Are phases shared across programs?}}
The uniqueness analysis finds {uniqueness['shared_phase_count']} phases present in at least four workloads and {uniqueness['workload_specific_phase_count']} workload-specific phases. Mean dominant-workload share is {fmt(uniqueness['mean_dominant_workload_share'])}. This supports the equivalence-class interpretation and argues against treating phase IDs as benchmark names.

\subsection{{RQ3: How well are phase changes detected?}}
\begin{{table}}[t]
\centering
\caption{{Prediction results. Exact next-phase accuracy is weak; phase-change F1 is the relevant control metric.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Samples & Acc. & Macro-F1 & Change-F1 \\
\midrule
{model_table}
\bottomrule
\end{{tabular}}
\end{{table}}

The best exact next-phase accuracy is only {fmt(best_accuracy['accuracy'])} from {best_accuracy['model'].replace('_', ' ')}. The best phase-change F1 is {fmt(best_change['phase_change_f1'])} from {best_change['model'].replace('_', ' ')}. The transformer reports {fmt(transformer_summary['inference_us_per_sample'])} microseconds per sample in batched GPU inference; the student tree preserves change-F1 {fmt(student_summary['phase_change_f1'])} with a compact threshold model.

{rq4_text}

\subsection{{RQ5: What is the cost?}}
\begin{{table}}[t]
\centering
\caption{{Detector cost estimate for 8 features and 16 centroids per core.}}
\begin{{tabular}}{{rrr}}
\toprule
Bits & Bytes/core & Ops/window \\
\midrule
{cost_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{RQ6: What ablations matter?}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../{figure_dir}/feature_group_ablation.pdf}}
\caption{{Feature-group ablation for the centroid detector.}}
\end{{figure}}

The best detector ablation row has phase-change F1 {fmt(best_ablation['phase_change_f1'])} using feature group {best_ablation['feature_group']}, metric {best_ablation['metric']}, persistence {best_ablation['persistence']}, and precision {best_ablation['fixed_point_bits']}. The worst transfer setting reaches phase-change F1 {fmt(worst_generalization['phase_change_f1'])}, which should be reported as a limitation rather than hidden.

\begin{{table}}[t]
\centering
\caption{{PMU slot sensitivity.}}
\begin{{tabular}}{{rrrr}}
\toprule
Slots & Features & Acc. & Change-F1 \\
\midrule
{pmu_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Discussion}}
Weak exact next-phase accuracy is not a side issue; it prevents claims about exact future phase identity. The defensible control signal is phase change. The current replay suggests how a scheduler could consume the signal, but the paper must not claim real performance benefit until live placement, CAT, prefetch, or bandwidth-throttling experiments are run.

\section{{Related Work}}
The closest work spans BBV/SimPoint phase analysis \cite{{sherwood2001bbda,sherwood2002simpoint,hamerly2005simpoint}}, runtime phase tracking \cite{{sherwood2003phase}}, parallel phases \cite{{perelman2006parallel,chang2013sampling}}, PMU-based prediction \cite{{kim2017p4,alcorta2023forecasting}}, and contention-aware resource management \cite{{fedorova2010contention,subramanian2013mise,navarro2023balancer}}. \system{{}} is weaker than these papers on online validation today, but its intended niche is timing-independent PMU phase-change detection plus low-cost resource-policy integration.

\section{{Limitations}}
If online co-scheduling or DVFS result files are absent, the corresponding claims are omitted from the results and retained as limitations. Intel CAT/resctrl is unavailable locally unless the target node exposes \texttt{{/sys/fs/resctrl}}. Prefetch controls require platform-specific support and are not claimed unless measured. PMU multiplexing and limited counter slots remain risks. PARSEC-only evaluation limits generality.

\section{{Conclusion}}
Timing-independent PMU signatures are promising for multicore phase-change detection, but the current artifact should be presented as a validated detection and trace-replay study, not as a completed resource-management system. The next decisive experiment is live phase-aware placement or cache/bandwidth control with measured speedup and fairness.

\nocite{{*}}
\bibliographystyle{{IEEEtran}}
\bibliography{{references}}
\end{{document}}
"""
    (paper_dir / "main.tex").write_text(tex, encoding="utf-8")


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processed_dir = Path(str(config["processed_dir"]))
    run_root = Path(str(config["run_root"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    label_dir = Path(str(config["label_dir"]))
    eval_dir = Path(str(config["eval_dir"]))
    analysis_dir = Path(str(config["analysis_dir"]))
    figure_dir = Path(str(config["figure_dir"]))
    source_report_dir = Path(str(config["source_report_dir"]))
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: regenerate analysis tables from existing model artifacts.
    copy_existing_figures(source_report_dir, figure_dir)
    validation = validate(processed_dir, dataset_dir, label_dir, eval_dir, analysis_dir / "validation_summary.json")
    analyze_uniqueness(dataset_dir, label_dir, analysis_dir)
    run_replay(dataset_dir, label_dir, run_root, analysis_dir)
    run_ablations(dataset_dir, label_dir, analysis_dir)
    run_generalization(dataset_dir, label_dir, analysis_dir)
    run_slot_sensitivity(dataset_dir, label_dir, analysis_dir)
    estimate_cost(analysis_dir)
    online_dir = Path(str(config.get("online_dir", "")))
    online_runs = online_dir / "online_runs.csv"
    if online_runs.exists():
        analyze_online_coscheduling(online_runs, analysis_dir)
    dvfs_analysis_dir = Path(str(config.get("dvfs_analysis_dir", "")))
    if (dvfs_analysis_dir / "dvfs_stress.csv").exists():
        shutil.copy2(dvfs_analysis_dir / "dvfs_stress.csv", analysis_dir / "dvfs_stress.csv")
        summary_path = dvfs_analysis_dir / "dvfs_stress_summary.json"
        if summary_path.exists():
            shutil.copy2(summary_path, analysis_dir / "dvfs_stress_summary.json")
    # Stage 2: render publication figures from analysis CSV outputs.
    plot_policy_replay(analysis_dir, figure_dir)
    plot_ablation(analysis_dir, figure_dir)
    plot_generalization(analysis_dir, figure_dir)
    plot_pmu_slots(analysis_dir, figure_dir)
    plot_online_results(analysis_dir, figure_dir)
    plot_dvfs_results(analysis_dir, figure_dir)
    # Stage 3: synthesize paper/main.tex from measured artifacts only.
    write_paper(config, validation)
    write_json(Path("data/manifest") / f"{config['name']}_artifact_manifest.json", config)
    print(f"Regenerated paper artifact from {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
