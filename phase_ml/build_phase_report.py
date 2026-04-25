"""Build an IEEE-style phase-signature report from measured phase-ML outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover - scipy is available in the project env.
    spearmanr = None

from hpc_phase_analysis.io_utils import ensure_dir


FORBIDDEN_PARTS = ["cycle", "elapsed", "per_ms", "ipc", "cpi", "stall", "runtime", "duration"]

FEATURE_LABELS = {
    "branch_miss_rate": "Branch miss rate",
    "branch_instructions_per_ki": "Branches/KI",
    "l1d_loads_per_ki": "L1D loads/KI",
    "l1d_stores_per_ki": "L1D stores/KI",
    "load_store_ratio": "Load/store",
    "llc_references_per_ki": "LLC refs/KI",
    "llc_misses_per_ki": "LLC misses/KI",
    "llc_miss_rate": "LLC miss rate",
    "offcore_reads_per_ki": "Offcore reads/KI",
    "shared_proxy_llc_miss_rate": "Shared LLC miss proxy",
    "shared_proxy_offcore_reads_per_ki": "Shared offcore proxy",
    "shared_imc_read_mib": "IMC read MiB",
    "shared_imc_write_mib": "IMC write MiB",
    "shared_imc_total_mib": "IMC total MiB",
    "shared_imc_write_fraction": "IMC write fraction",
}

SET_LABELS = {
    "set1_single_process_multithread": "Set 1: one multithreaded process",
    "set2_multi_process_single_thread": "Set 2: multiprocess, single thread",
    "set3_hybrid_multi_process_multithread": "Set 3: multiprocess, multithreaded",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: object, digits: int = 3) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "--"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def finite_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(columns)), np.nan)
    for row_index, row in enumerate(rows):
        for col_index, column in enumerate(columns):
            matrix[row_index, col_index] = safe_float(row.get(column, ""))
    for col_index in range(matrix.shape[1]):
        column = matrix[:, col_index]
        finite = column[np.isfinite(column)]
        fill = float(np.median(finite)) if finite.size else 0.0
        column[~np.isfinite(column)] = fill
        matrix[:, col_index] = column
    return matrix


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def save_heatmap(path: Path, matrix: np.ndarray, labels: list[str], title: str, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_correlation(interval_rows: list[dict[str, str]], feature_columns: list[str], figure_dir: Path) -> Path:
    matrix = finite_matrix(interval_rows, feature_columns)
    if spearmanr is not None:
        corr = spearmanr(matrix, axis=0).correlation
    else:
        corr = np.corrcoef(np.argsort(np.argsort(matrix, axis=0), axis=0), rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    path = figure_dir / "allowed_feature_spearman.pdf"
    labels = [FEATURE_LABELS.get(column, column) for column in feature_columns]
    save_heatmap(path, corr, labels, "Spearman correlation of allowed indicators", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    return path


def plot_phase_timeline(window_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    by_set_run: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in window_rows:
        by_set_run[row.get("experiment_set", "")][row.get("run_id", "")].append(row)
    sets = [name for name in SET_LABELS if name in by_set_run]
    fig, axes = plt.subplots(len(sets), 1, figsize=(7.0, 4.6), sharex=False)
    if len(sets) == 1:
        axes = [axes]
    for ax, set_name in zip(axes, sets):
        runs = by_set_run[set_name]
        run_id, rows = max(runs.items(), key=lambda item: len(item[1]))
        rows = sorted(rows, key=lambda row: int(row.get("window_id", "0")))
        phases = [int(row["phase_id"]) for row in rows]
        ax.step(range(len(phases)), phases, where="post", linewidth=1.0)
        ax.set_ylabel("Phase", fontsize=8)
        ax.set_title(f"{SET_LABELS[set_name]} ({run_id[:42]}...)", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Window index within representative run", fontsize=8)
    fig.tight_layout()
    path = figure_dir / "phase_timeline.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_distribution(window_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    phases = sorted({int(row["phase_id"]) for row in window_rows})
    sets = [name for name in SET_LABELS if any(row.get("experiment_set") == name for row in window_rows)]
    workloads = sorted({row.get("workload", "") for row in window_rows})
    counts: dict[tuple[str, str], Counter[int]] = {(set_name, workload): Counter() for set_name in sets for workload in workloads}
    for row in window_rows:
        set_name = row.get("experiment_set", "")
        key = (set_name, row.get("workload", ""))
        if key in counts:
            counts[key][int(row["phase_id"])] += 1
    keys = [key for key in counts if sum(counts[key].values()) > 0]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    bottom = np.zeros(len(keys))
    for phase in phases:
        values = np.asarray([counts[key][phase] for key in keys], dtype=float)
        totals = np.asarray([sum(counts[key].values()) for key in keys], dtype=float)
        frac = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
        ax.bar(range(len(keys)), frac, bottom=bottom, label=f"P{phase}")
        bottom += frac
    short_set = {
        "set1_single_process_multithread": "S1",
        "set2_multi_process_single_thread": "S2",
        "set3_hybrid_multi_process_multithread": "S3",
    }
    labels = [f"{short_set.get(set_name, set_name)}:{workload}" for set_name, workload in keys]
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of windows")
    ax.set_title("Phase distribution by workload and experiment set")
    ax.legend(ncol=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = figure_dir / "phase_distribution_by_set.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def window_feature_matrix(dataset_dir: Path, window_rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    tensors = np.load(dataset_dir / "window_tensors.npz", allow_pickle=True)
    x = np.asarray(tensors["X"], dtype=float)
    feature_columns = [str(item) for item in tensors["feature_columns"]]
    features = np.nanmean(x, axis=1)
    if features.shape[0] != len(window_rows):
        raise ValueError("window_tensors and window_labels disagree on row count")
    return features, feature_columns


def phase_centroids(features: np.ndarray, window_rows: list[dict[str, str]]) -> tuple[list[int], np.ndarray]:
    labels = np.asarray([int(row["phase_id"]) for row in window_rows], dtype=int)
    z = standardize(np.nan_to_num(features, nan=0.0))
    phases = sorted(set(labels.tolist()))
    centroids = []
    for phase in phases:
        centroids.append(z[labels == phase].mean(axis=0))
    return phases, np.vstack(centroids)


def plot_phase_distance(features: np.ndarray, window_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    phases, centroids = phase_centroids(features, window_rows)
    distance = np.zeros((len(phases), len(phases)))
    for i in range(len(phases)):
        for j in range(len(phases)):
            distance[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))
    path = figure_dir / "inter_phase_centroid_distance.pdf"
    save_heatmap(path, distance, [str(phase) for phase in phases], "Inter-phase centroid distance")
    return path


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def plot_workload_similarity(window_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    workloads = sorted({row["workload"] for row in window_rows})
    phases = sorted({int(row["phase_id"]) for row in window_rows})
    dist = np.zeros((len(workloads), len(phases)))
    phase_to_index = {phase: index for index, phase in enumerate(phases)}
    for row in window_rows:
        dist[workloads.index(row["workload"]), phase_to_index[int(row["phase_id"])]] += 1
    matrix = np.zeros((len(workloads), len(workloads)))
    for i in range(len(workloads)):
        for j in range(len(workloads)):
            matrix[i, j] = js_distance(dist[i], dist[j])
    path = figure_dir / "cross_program_signature_jsd.pdf"
    save_heatmap(path, matrix, workloads, "Cross-program phase-signature distance")
    return path


def plot_variance_reduction(features: np.ndarray, feature_columns: list[str], window_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    labels = np.asarray([int(row["phase_id"]) for row in window_rows], dtype=int)
    matrix = np.nan_to_num(features, nan=0.0)
    reductions = []
    for col_index in range(matrix.shape[1]):
        values = matrix[:, col_index]
        total_var = float(np.var(values))
        if total_var == 0.0:
            reductions.append(0.0)
            continue
        within = 0.0
        for phase in sorted(set(labels.tolist())):
            phase_values = values[labels == phase]
            within += (phase_values.size / values.size) * float(np.var(phase_values))
        reductions.append(max(0.0, 1.0 - within / total_var))
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.bar(range(len(feature_columns)), reductions, color="#4C78A8")
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels([FEATURE_LABELS.get(column, column) for column in feature_columns], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Variance reduction")
    ax.set_title("Intra-phase stability relative to whole execution")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = figure_dir / "intra_phase_variance_reduction.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_baseline_metrics(metrics_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    models = [row["model"].replace("_", " ") for row in metrics_rows]
    metrics = ["accuracy", "macro_f1", "phase_change_f1"]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for offset, metric in enumerate(metrics):
        values = [safe_float(row[metric]) for row in metrics_rows]
        ax.bar(x + (offset - 1) * width, values, width=width, label=metric.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Non-transformer prediction baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figure_dir / "baseline_metrics.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion(eval_dir: Path, metrics_rows: list[dict[str, str]], figure_dir: Path) -> tuple[Path, str]:
    best = max(metrics_rows, key=lambda row: safe_float(row.get("accuracy", "")))
    model = best["model"]
    payload = read_json(eval_dir / f"{model}_confusion_matrix.json")
    matrix = np.asarray(payload["matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    path = figure_dir / "best_baseline_confusion.pdf"
    labels = [str(index) for index in range(norm.shape[0])]
    save_heatmap(path, norm, labels, f"Normalized confusion matrix: {model}")
    return path, model


def summarize(window_rows: list[dict[str, str]], interval_rows: list[dict[str, str]]) -> dict[str, object]:
    set_windows = Counter(row.get("experiment_set", "") for row in window_rows)
    set_intervals = Counter(row.get("experiment_set", "") for row in interval_rows)
    collection_scopes = Counter(row.get("collection_scope", "") for row in window_rows)
    set_scopes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in window_rows:
        set_scopes[row.get("experiment_set", "")][row.get("collection_scope", "")] += 1
    workloads = Counter(row.get("workload", "") for row in window_rows)
    phases = Counter(int(row["phase_id"]) for row in window_rows)
    phase_workloads: dict[int, Counter[str]] = defaultdict(Counter)
    for row in window_rows:
        phase_workloads[int(row["phase_id"])][row["workload"]] += 1
    shared = []
    specific = []
    for phase, count in sorted(phases.items()):
        per_workload = phase_workloads[phase]
        present = sum(1 for value in per_workload.values() if value)
        dominant, dominant_count = per_workload.most_common(1)[0]
        item = {
            "phase": phase,
            "count": count,
            "present": present,
            "dominant": dominant,
            "dominant_share": dominant_count / count if count else 0.0,
        }
        if present >= 4:
            shared.append(item)
        if item["dominant_share"] >= 0.55 or present <= 2:
            specific.append(item)
    return {
        "set_windows": set_windows,
        "set_intervals": set_intervals,
        "collection_scopes": collection_scopes,
        "set_scopes": set_scopes,
        "workloads": workloads,
        "phase_count": len(phases),
        "shared": sorted(shared, key=lambda item: item["count"], reverse=True)[:5],
        "specific": sorted(specific, key=lambda item: item["dominant_share"], reverse=True)[:5],
    }


def table_rows(rows: Iterable[Iterable[object]]) -> str:
    return "\n".join(" & ".join(latex_escape(item) for item in row) + r" \\" for row in rows)


def write_bib(path: Path) -> None:
    path.write_text(
        r"""
@inproceedings{sherwood2001bbda,
  author={Timothy Sherwood and Erez Perelman and Brad Calder},
  title={Basic Block Distribution Analysis to Find Periodic Behavior and Simulation Points in Applications},
  booktitle={Proceedings of the International Conference on Parallel Architectures and Compilation Techniques},
  pages={3--14},
  year={2001}
}
@inproceedings{sherwood2002simpoint,
  author={Timothy Sherwood and Erez Perelman and Greg Hamerly and Brad Calder},
  title={Automatically Characterizing Large Scale Program Behavior},
  booktitle={Proceedings of the 10th International Conference on Architectural Support for Programming Languages and Operating Systems},
  pages={45--57},
  year={2002},
  doi={10.1145/605397.605403}
}
@inproceedings{sherwood2003phase,
  author={Timothy Sherwood and Suleyman Sair and Brad Calder},
  title={Phase Tracking and Prediction},
  booktitle={Proceedings of the 30th International Symposium on Computer Architecture},
  pages={336--347},
  year={2003},
  doi={10.1109/ISCA.2003.1207012}
}
@article{hamerly2005simpoint,
  author={Greg Hamerly and Erez Perelman and Jeremy Lau and Brad Calder},
  title={SimPoint 3.0: Faster and More Flexible Program Phase Analysis},
  journal={Journal of Instruction-Level Parallelism},
  volume={7},
  pages={1--28},
  year={2005}
}
@article{hamerly2006jmlr,
  author={Greg Hamerly and Erez Perelman and Jeremy Lau and Brad Calder and Timothy Sherwood},
  title={Using Machine Learning to Guide Architecture Simulation},
  journal={Journal of Machine Learning Research},
  volume={7},
  pages={343--378},
  year={2006}
}
@inproceedings{perelman2006parallel,
  author={Erez Perelman and Marzia Polito and Jean-Yves Bouguet and John Sampson and Brad Calder and Carole Dulong},
  title={Detecting Phases in Parallel Applications on Shared Memory Architectures},
  booktitle={Proceedings of the IEEE International Parallel and Distributed Processing Symposium},
  year={2006},
  doi={10.1109/IPDPS.2006.1639325}
}
@inproceedings{ding2006phase,
  author={Chen Ding and Sandhya Dwarkadas and Michael C. Huang and Kai Shen and John B. Carter},
  title={Program Phase Detection and Exploitation},
  booktitle={Proceedings of the IEEE International Parallel and Distributed Processing Symposium},
  year={2006}
}
@article{ziedan2016l2cache,
  author={Ibrahim E. Ziedan and Hazem I. Shehata and Shaymaa M. Serag},
  title={A Run-Time Program Phase Detection Technique for Optimizing Per-Phase L2 Cache Demand},
  journal={The Egyptian International Journal of Engineering Sciences and Technology},
  year={2016}
}
@inproceedings{bienia2008parsec,
  author={Christian Bienia and Sanjeev Kumar and Jaswinder Pal Singh and Kai Li},
  title={The PARSEC Benchmark Suite: Characterization and Architectural Implications},
  booktitle={Proceedings of the 17th International Conference on Parallel Architectures and Compilation Techniques},
  pages={72--81},
  year={2008},
  doi={10.1145/1454115.1454128}
}
@inproceedings{srinath2007feedback,
  author={Santhosh Srinath and Onur Mutlu and Hyesoon Kim and Yale N. Patt},
  title={Feedback Directed Prefetching: Improving the Performance and Bandwidth-Efficiency of Hardware Prefetchers},
  booktitle={Proceedings of the 13th International Symposium on High Performance Computer Architecture},
  pages={63--74},
  year={2007},
  doi={10.1109/HPCA.2007.346185}
}
@inproceedings{chang2013sampling,
  author={Chin-Hao Chang and Pangfeng Liu and Jan-Jan Wu},
  title={Sampling-Based Phase Classification and Prediction for Multi-threaded Program Execution on Multi-core Architectures},
  booktitle={Proceedings of the International Conference on Parallel Processing},
  year={2013}
}
@inproceedings{kim2017p4,
  author={Yeseong Kim and Pietro Mercati and Ankit More and Emily Shriver and Tajana Rosing},
  title={{P4}: Phase-Based Power/Performance Prediction of Heterogeneous Systems via Neural Networks},
  booktitle={Proceedings of the IEEE/ACM International Conference on Computer-Aided Design},
  pages={683--690},
  year={2017},
  doi={10.1109/ICCAD.2017.8203843}
}
@article{alcorta2023forecasting,
  author={Erika S. Alcorta and Andreas Gerstlauer},
  title={Learning-Based Phase-Aware Multi-Core CPU Workload Forecasting},
  journal={ACM Transactions on Design Automation of Electronic Systems},
  volume={28},
  number={2},
  pages={23:1--23:27},
  year={2023},
  doi={10.1145/3564929}
}
@inproceedings{carpentieri2025dvfs,
  author={Lorenzo Carpentieri and Antonio De Caro and Majid Salimi Beni and Kaijie Fan and Biagio Cosenza},
  title={Phase-Based Frequency Scaling for Energy-Efficient Heterogeneous Computing},
  booktitle={Proceedings of the IEEE International Parallel and Distributed Processing Symposium},
  year={2025}
}
@article{fedorova2010contention,
  author={Alexandra Fedorova and Sergey Blagodurov and Sergey Zhuravlev},
  title={Managing Contention for Shared Resources on Multicore Processors},
  journal={ACM Queue},
  volume={8},
  number={1},
  year={2010}
}
@inproceedings{subramanian2013mise,
  author={Lavanya Subramanian and Vivek Seshadri and Yoongu Kim and Ben Jaiyen and Onur Mutlu},
  title={{MISE}: Providing Performance Predictability and Improving Fairness in Shared Main Memory Systems},
  booktitle={Proceedings of the 19th International Symposium on High Performance Computer Architecture},
  year={2013}
}
@article{navarro2023balancer,
  author={Agustin Navarro-Torres and Jesus Alastruey-Benede and Pablo Ibanez and Victor Vinals-Yufera},
  title={BALANCER: Bandwidth Allocation and Cache Partitioning for Multicore Processors},
  journal={The Journal of Supercomputing},
  volume={79},
  pages={10252--10276},
  year={2023}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def latex_report(
    *,
    manifest: dict[str, object],
    merge_summary: dict[str, object],
    summary: dict[str, object],
    metrics_rows: list[dict[str, str]],
    best_model: str,
    feature_columns: list[str],
) -> str:
    def scope_label(scope: str) -> str:
        if scope == "system_wide_physical_core":
            return "physical-core perf"
        if scope == "system_wide_cpu":
            return "logical-CPU perf"
        return "task-local perf"

    dominant_scope = ""
    if summary["collection_scopes"]:
        dominant_scope = summary["collection_scopes"].most_common(1)[0][0]
    dominant_scope_label = scope_label(str(dominant_scope))
    if dominant_scope == "system_wide_physical_core":
        measurement_note = (
            "The present measurements use system-wide per-CPU perf collection and merge logical CPUs into physical-core streams "
            "using the host topology, while retaining uncore IMC traffic as shared context."
        )
    elif dominant_scope == "system_wide_cpu":
        measurement_note = (
            "The present measurements use system-wide per-CPU perf collection. They preserve logical-CPU streams rather than "
            "aggregating SMT siblings into physical cores."
        )
    else:
        measurement_note = (
            "The present measurements are task-local because system-wide per-core collection was not used for this dataset. "
            "Consequently, multi-threaded process rows are affinity-group task-local streams, not true independent physical-core streams."
        )
    experiment_rows = []
    for set_name in SET_LABELS:
        set_scope_counter = summary["set_scopes"].get(set_name, Counter())
        set_scope = set_scope_counter.most_common(1)[0][0] if set_scope_counter else dominant_scope
        experiment_rows.append(
            [
                SET_LABELS[set_name],
                int(summary["set_intervals"].get(set_name, 0)),
                int(summary["set_windows"].get(set_name, 0)),
                scope_label(str(set_scope)),
            ]
        )
    metrics_table = table_rows(
        [
            [
                row["model"].replace("_", " "),
                row["samples"],
                fmt(row["accuracy"]),
                fmt(row["macro_f1"]),
                fmt(row["phase_change_f1"]),
            ]
            for row in metrics_rows
        ]
    )
    experiment_table = table_rows(experiment_rows)
    shared_table = table_rows(
        [
            [item["phase"], item["count"], item["present"], item["dominant"], fmt(item["dominant_share"])]
            for item in summary["shared"]
        ]
    )
    specific_table = table_rows(
        [
            [item["phase"], item["count"], item["present"], item["dominant"], fmt(item["dominant_share"])]
            for item in summary["specific"]
        ]
    )
    feature_table = table_rows(
        [
            ["Instructions retired", "Core", "Normalizer", "Yes, denominator for per-KI ratios"],
            ["Branch instructions", "Core", "Control flow", "Yes, as branches/KI"],
            ["Branch mispredictions", "Core", "Control flow", "Yes, as branch miss rate"],
            ["L1D loads/stores", "Core", "Memory access mix", "Partly, summarized by ratios"],
            ["LLC references/misses", "Core/shared-proxy", "Shared-cache pressure", "Yes"],
            ["Offcore demand data reads", "Core/shared-proxy", "Memory pressure proxy", "Yes if available"],
            ["Cycles, IPC, CPI, elapsed time", "Core/time", "Timing", "Excluded by policy"],
            ["Resource stalls", "Core", "Stall timing proxy", "Excluded by policy"],
        ]
    )
    selected_features = ", ".join(latex_escape(FEATURE_LABELS.get(column, column)) for column in feature_columns)
    best_model_latex = latex_escape(best_model.replace("_", " "))
    return rf"""
\documentclass[conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{url}}
\usepackage{{cite}}
\usepackage{{balance}}
\usepackage{{xcolor}}
\newcommand{{\phasevec}}{{\mathbf{{s}}}}
\newcommand{{\centroid}}{{\boldsymbol{{\mu}}}}
\begin{{document}}
\title{{Lightweight Phase Signatures for Multicore CPU Workloads Using Timing-Independent Performance Counters}}
\author{{\IEEEauthorblockN{{Keshav K.}}
\IEEEauthorblockA{{Hardware Counter Correlation and Phase-ML Study}}}}
\maketitle
\begin{{abstract}}
Runtime phase behavior is a useful abstraction for multicore resource management, but practical phase detectors must avoid heavy profiling mechanisms and unstable timing-derived indicators. This report studies phase signatures for multicore CPU workloads using {dominant_scope_label} hardware performance counters collected from PARSEC runs. The study excludes cycles, IPC, CPI, elapsed time, per-millisecond rates, and stall-derived indicators, and instead uses instruction-normalized, access-normalized, and ratio-based features. We evaluate three execution settings: one multithreaded process, multiple single-threaded processes, and hybrid multiprocess multithreaded mixes. The results show that phase labels are not unique program identifiers: several phase signatures appear across different programs, while program behavior is better represented by distributions and transitions over reusable phase classes. We therefore propose a lightweight online detector based on per-core centroid signatures, minimum-persistence filtering, and a small shared-resource context vector.
\end{{abstract}}
\begin{{IEEEkeywords}}
program phases, hardware performance counters, multicore processors, PMU, PARSEC, phase signatures, lightweight hardware monitoring
\end{{IEEEkeywords}}

\section{{Introduction}}
Programs rarely exercise a processor uniformly. Their memory access patterns, branch behavior, cache locality, and sharing behavior vary over execution, creating phases that can guide cache management, prefetch control, scheduling, and resource allocation. Classical phase analysis showed that basic-block vectors can summarize recurring execution behavior and reduce simulation cost \cite{{sherwood2001bbda,sherwood2002simpoint,hamerly2005simpoint,hamerly2006jmlr}}. Runtime systems, however, cannot generally afford source-code instrumentation or binary basic-block profiling. Hardware performance counters provide a cheaper observation channel that already exists on commodity processors.

Multicore execution complicates phase detection. Threads can enter different regions at different times; independent processes can contend for shared cache and memory resources; and a globally aggregated phase can hide asynchronous behavior. Prior multithreaded phase work argues for preserving per-thread activity representations \cite{{perelman2006parallel,chang2013sampling}}, while recent phase-aware workload forecasting studies show that multicore phase models can improve prediction but often rely on heavier learning pipelines \cite{{alcorta2023forecasting}}. This report takes a more hardware-conscious position: use per-core or task-local phase signatures as the default, and add only a tiny shared-resource context vector.

The contributions are:
\begin{{itemize}}
\item A timing-independent counter-selection methodology that excludes cycles, IPC, CPI, elapsed time, per-time rates, and stall-derived indicators.
\item A measured PARSEC phase-signature study across three execution settings with {int(merge_summary["merged_run_count"])} task-local perf runs and {int(manifest["windows"]):,} sliding windows.
\item A lightweight online phase detector design based on compact centroid tables, exponentially weighted signatures, and transition filtering.
\item A balanced analysis of whether phase signatures are unique, showing that many signatures are shared across programs and are better interpreted as behavioral equivalence classes.
\end{{itemize}}

\section{{Problem Definition}}
Let core or collection unit $c$ produce a sequence of counter observations $x_{{c,t}}$ over fixed observation windows. The goal is to construct a timing-independent signature vector
\begin{{equation}}
\phasevec_{{c,t}} = f(x_{{c,t-k+1}}, \ldots, x_{{c,t}})
\end{{equation}}
using only genuine available counters and ratios such as misses per kilo-instruction or miss rates. We seek a phase assignment $p_{{c,t}}$, a phase-change signal $\Delta p_{{c,t}}$, and a compact signature representation that can support later resource adaptation. The central hypothesis is that two different programs may contain phases with similar counter signatures. Thus, a phase identifier should not be interpreted as a program identifier; it is a reusable microarchitectural behavior class.

The target setting is a multicore CPU with commodity PMCs. {measurement_note}

\section{{Literature Review}}
BBV-based work established the modern notion of program phases as recurring intervals with similar behavior rather than merely contiguous time regions \cite{{sherwood2001bbda,sherwood2002simpoint}}. SimPoint 3.0 refined clustering and interval selection for scalable architectural simulation \cite{{hamerly2005simpoint}}, and later machine-learning studies clarified preprocessing and clustering choices \cite{{hamerly2006jmlr}}. These methods are powerful offline tools but require code-structure profiles rather than low-cost online PMCs.

Runtime phase tracking moved toward hardware-feasible fingerprints. Sherwood et al. proposed a small on-chip footprint table and run-length encoded Markov prediction, showing that large-scale behavior can be tracked with very small storage \cite{{sherwood2003phase}}. Other runtime phase-detection work connects phase signatures to cache demand and adaptation opportunities \cite{{ding2006phase,ziedan2016l2cache}}. These insights are important for our design: the online mechanism should be closer to tables, thresholds, and short histories than to deep neural sequence models.

Multithreaded phase analysis introduced a key design choice: global versus local phase definitions. Perelman et al. adapted phase analysis to shared-memory parallel applications by keeping separate thread activity representations \cite{{perelman2006parallel}}. Sampling-based multicore phase classification further reinforces the need to consider per-thread or per-core phase sequences \cite{{chang2013sampling}}. Global aggregation is simpler, but it can mask asynchronous threads. Distributed approaches capture richer relationships, but raw other-core data can add noise and scale poorly.

PMU-based phase and forecasting methods demonstrate the appeal of counters for online adaptation. P4 uses phase-based neural models for heterogeneous systems \cite{{kim2017p4}}, phase-aware CPU workload forecasting combines GMM phase classification with temporal predictors for multicore workloads \cite{{alcorta2023forecasting}}, and phase-based frequency-control work shows how phase information can drive energy policies \cite{{carpentieri2025dvfs}}. These works motivate phase-aware prediction, but the present report intentionally avoids heavy online learners in the online hardware path. A transformer or neural predictor can be useful as an offline teacher, but the implementable mechanism should remain compact.

Shared-resource literature explains why local phase signatures need limited shared context. Contention-aware scheduling work identifies LLCs, memory controllers, and interconnects as shared bottlenecks \cite{{fedorova2010contention}}. MISE models memory-interference-induced slowdown in shared main memory systems \cite{{subramanian2013mise}}, while cache and bandwidth allocation mechanisms such as BALANCER show that LLC and memory pressure are actionable signals \cite{{navarro2023balancer}}. Feedback-directed prefetching is also instructive because it uses hardware feedback to control aggressiveness without a heavyweight predictor \cite{{srinath2007feedback}}.

\begin{{table*}}[t]
\centering
\caption{{Comparison of phase definitions for multicore workloads.}}
\begin{{tabular}}{{p{{0.12\linewidth}}p{{0.23\linewidth}}p{{0.18\linewidth}}p{{0.22\linewidth}}p{{0.17\linewidth}}}}
\toprule
Definition & Description & Strengths & Weaknesses & Hardware cost \\
\midrule
Global & Aggregate all cores or all threads into one program-level phase. & Simple, stable for synchronized workloads. & Masks asynchronous per-core behavior and process interference. & Low counters, low state. \\
Local/per-core & Classify each core or thread independently. & Interpretable and scalable; preserves asynchrony. & Misses shared-resource pressure. & Per-core small table. \\
Distributed & Use local and other-core state jointly. & Captures interactions and contention. & Can be noisy and communication-heavy. & State and interconnect cost grows with cores. \\
Hybrid proposed & Per-core signatures plus limited LLC/offcore context. & Keeps local interpretability while exposing shared pressure. & Shared proxies are approximate under restricted PMU access. & Small per-core table plus few shared fields. \\
\bottomrule
\end{{tabular}}
\end{{table*}}

\subsection{{Lessons Learned}}
The literature suggests four principles. First, phase definitions should be reusable behavior classes, not program names. Second, multicore phase tracking should preserve local sequences unless synchronization makes a global phase clearly valid. Third, shared-resource context should be filtered and compact. Fourth, online implementation should favor counters, tables, centroids, and hysteresis over heavyweight recurrent or transformer models.

\section{{Counter Selection and Indicator Study}}
The candidate counter families are instructions retired, branch instructions, branch mispredictions, L1 data loads and stores, LLC references and misses, offcore demand reads, and, when permitted, uncore memory bandwidth. Cycles, IPC, CPI, elapsed time, per-time rates, and stall-derived quantities are excluded. The measured feature set is:
\emph{{{selected_features}}}.

\begin{{table}}[t]
\centering
\caption{{Candidate and selected indicators.}}
\begin{{tabular}}{{p{{0.27\linewidth}}p{{0.16\linewidth}}p{{0.20\linewidth}}p{{0.27\linewidth}}}}
\toprule
Counter/indicator & Scope & Role & Decision \\
\midrule
{feature_table}
\bottomrule
\end{{tabular}}
\end{{table}}

The selection process uses rank correlation, redundancy clustering, and variance checks. Spearman correlation is preferred because nonlinear monotonic relations are common in PMU data. Redundant indicators are avoided when one interpretable ratio captures the same behavior as several raw counts. Because timing metrics are disallowed, the target notion is not direct speedup prediction. Instead, indicators are retained if they explain stable behavioral classes: intra-phase variance should be lower than whole-execution variance, and phase centroids should separate cache, control-flow, and memory-pressure regimes.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/allowed_feature_spearman.pdf}}
\caption{{Spearman correlation among timing-independent indicators. Strongly correlated groups motivate a compact selected set rather than monitoring all available events.}}
\label{{fig:corr}}
\end{{figure}}

\section{{Proposed Phase-Change Detection Algorithm}}
The proposed online detector maintains one compact phase table per core or collection unit. Each observation window computes a signature vector from instruction-normalized and ratio features. An exponentially weighted signature
\begin{{equation}}
\hat{{\phasevec}}_{{t}} = \alpha \phasevec_t + (1-\alpha)\hat{{\phasevec}}_{{t-1}}
\end{{equation}}
is compared against stored centroids $\centroid_i$ using weighted Manhattan distance. If the nearest centroid is within threshold $\tau$, the detector emits that phase ID; otherwise it allocates a new entry if table space exists, or merges into the least recently used low-confidence entry.

Noise is handled with minimum persistence. A candidate phase must be observed for $m$ consecutive windows or exceed a confidence margin before the detector raises a phase-change signal. Unstable intervals are assigned to phase zero and do not trigger resource actions. Centroids are updated by saturating fixed-point EWMA. The optional shared context contains only a few LLC/offcore pressure proxies, so the local phase remains primary.

\section{{Hardware Cost and Implementation Analysis}}
An example implementation uses six to eight feature fields per core, 12-bit fixed-point fields, 16 phase-table entries, and one confidence byte per entry. A 16-entry table with eight 12-bit centroid fields requires roughly 192 bytes per core before metadata. Distance computation is $O(Kd)$ for $K$ centroids and $d$ features; with $K=16$ and $d=8$, the detector needs 128 subtract-absolute-accumulate operations per window. Since windows are sampled at millisecond granularity, this can be implemented in firmware, a low-priority microcontroller, or simple hardware assist without adding latency to the core pipeline.

\begin{{table}}[t]
\centering
\caption{{Illustrative hardware cost for a compact detector.}}
\begin{{tabular}}{{lrr}}
\toprule
Component & Example & Storage \\
\midrule
Feature vector & 8 fields x 12 bits & 12 B \\
Phase centroids & 16 entries x 8 x 12 bits & 192 B/core \\
Counters/history & current, candidate, run length & 8--16 B/core \\
Confidence/filter state & 16 small counters & 16 B/core \\
Shared context & 2--4 socket fields & $<$16 B/socket \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Phase Discovery Experiments}}
We collected three PARSEC experiment sets using the PARSEC benchmark suite, whose workloads were designed to stress modern chip-multiprocessor behavior \cite{{bienia2008parsec}}. Set 1 runs one multithreaded process at a time. Set 2 runs multiple independent single-threaded PARSEC processes. Set 3 runs multiple PARSEC processes that are themselves multithreaded. Table~\ref{{tab:settings}} summarizes the measured v2 dataset.

\begin{{table}}[t]
\centering
\caption{{Measured experiment settings.}}
\label{{tab:settings}}
\begin{{tabular}}{{p{{0.42\linewidth}}rrp{{0.18\linewidth}}}}
\toprule
Setting & Intervals & Windows & Scope \\
\midrule
{experiment_table}
\bottomrule
\end{{tabular}}
\end{{table}}

The validated run contains {int(merge_summary["merged_run_count"])} merged runs, {int(merge_summary["interval_rows"]):,} intervals, and {int(manifest["windows"]):,} windows. The manifest and merge counts match, and all five repetitions are present. The split is grouped by run ID to avoid interval leakage.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/phase_timeline.pdf}}
\caption{{Representative phase timelines for the three experiment settings. The labels are unsupervised FGMM phases, not semantic program names.}}
\label{{fig:timeline}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/phase_distribution_by_set.pdf}}
\caption{{Phase mixture differs across workloads and execution settings, indicating that co-runner structure changes the observed signature distribution.}}
\label{{fig:dist}}
\end{{figure}}

The non-transformer baselines in Table~\ref{{tab:baseline}} are included to check whether the phase labels contain predictable temporal structure. Decision trees achieve the highest accuracy and phase-change F1 in this run, while last-value prediction is a useful but incomplete baseline because it does not predict changes. The normalized confusion matrix for the best baseline, {best_model_latex}, is shown in Fig.~\ref{{fig:confusion}}.

\begin{{table}}[t]
\centering
\caption{{Run-grouped non-transformer prediction results.}}
\label{{tab:baseline}}
\begin{{tabular}}{{lrrrr}}
\toprule
Model & Samples & Acc. & Macro-F1 & Change-F1 \\
\midrule
{metrics_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/baseline_metrics.pdf}}
\caption{{Baseline comparison for next-phase and phase-change prediction.}}
\label{{fig:baselines}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/best_baseline_confusion.pdf}}
\caption{{Normalized confusion matrix for the best non-transformer baseline under the run-grouped split.}}
\label{{fig:confusion}}
\end{{figure}}

\section{{Unique Phase Signature Analysis}}
We define a phase signature as the centroid and dispersion of timing-independent counter ratios for all windows assigned to a phase. A signature is unique only if it is both internally stable and externally separated from signatures of other phases or programs. The measured data suggests a more nuanced result: phases form behavioral equivalence classes. Some are shared across programs, while others are dominated by one workload or one workload mix.

\begin{{table}}[t]
\centering
\caption{{Examples of shared phase signatures.}}
\begin{{tabular}}{{rrrrr}}
\toprule
Phase & Windows & Workloads & Dominant & Share \\
\midrule
{shared_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Examples of more workload-specific signatures.}}
\begin{{tabular}}{{rrrrr}}
\toprule
Phase & Windows & Workloads & Dominant & Share \\
\midrule
{specific_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/inter_phase_centroid_distance.pdf}}
\caption{{Distance between phase centroids in standardized timing-independent feature space.}}
\label{{fig:distance}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/cross_program_signature_jsd.pdf}}
\caption{{Jensen-Shannon distance between workload phase distributions. Low values indicate workloads that share similar mixtures of phase signatures.}}
\label{{fig:jsd}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/intra_phase_variance_reduction.pdf}}
\caption{{Reduction in within-phase variance relative to whole-execution variance. Larger values indicate more stable phase signatures.}}
\label{{fig:variance}}
\end{{figure}}

These results support the hypothesis that different programs can exhibit similar phases. The practical implication is that online hardware should not use phase IDs as program-specific labels. Instead, it should treat phases as compact behavioral signatures such as branch-heavy, LLC-sensitive, streaming, or offcore-read-intensive classes.

\section{{Discussion}}
Global phases are attractive because they are easy to store and reason about, but they are least reliable for asynchronous multicore execution. Local phases preserve per-core behavior and work naturally for independent processes. Hybrid phases with limited shared context are a practical compromise: they retain local interpretability while exposing LLC and memory-pressure signals relevant to contention-aware scheduling.

Excluding timing and cycle-derived indicators reduces sensitivity to DVFS, thermal drift, and scheduling artifacts. The tradeoff is that the detector no longer directly observes throughput. It must infer behavior from access ratios and event composition. This is appropriate for a phase detector whose purpose is classification and resource characterization rather than direct speed prediction.

The main threat to validity is PMU policy and measurement scope. If system-wide collection is unavailable, multi-threaded streams fall back to task-local or affinity-group-local observations. If system-wide collection is available, OS activity on monitored CPUs can still appear in per-core counters. Additional threats include event-name portability, multiplexing, PARSEC input size, OS scheduling noise, SMT topology assumptions, and the fact that unsupervised labels depend on clustering settings.

\section{{Conclusion}}
This report presents a timing-independent phase-signature study for multicore CPU workloads using commodity hardware counters. The measured three-set PARSEC experiment shows that compact PMU ratios can form stable and predictable phase classes, but phase signatures are not uniquely tied to programs. The most hardware-feasible design is a per-core centroid detector with minimum-persistence filtering and a small shared-resource context vector. Future work should repeat the study with true per-physical-core and uncore collection once PMU policy permits system-wide access.

\balance
\bibliographystyle{{IEEEtran}}
\bibliography{{references}}
\end{{document}}
"""


def validate_no_forbidden(columns: Iterable[str]) -> None:
    bad = [column for column in columns if any(part in column.lower() for part in FORBIDDEN_PARTS)]
    if bad:
        raise ValueError("Forbidden timing/cycle/stall-derived columns found: " + ", ".join(bad))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="results/processed_phase_ml_experiments_v2")
    parser.add_argument("--dataset-dir", default="results/phase_ml_experiments_v2/dataset")
    parser.add_argument("--label-dir", default="results/phase_ml_experiments_v2/labels")
    parser.add_argument("--eval-dir", default="results/phase_ml_experiments_v2/evaluation")
    parser.add_argument("--output-dir", default="results/phase_ml_paper")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    dataset_dir = Path(args.dataset_dir)
    label_dir = Path(args.label_dir)
    eval_dir = Path(args.eval_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    figure_dir = ensure_dir(output_dir / "figures")

    manifest = read_json(dataset_dir / "feature_manifest.json")
    merge_summary = read_json(processed_dir / "merge_summary.json")
    interval_rows = load_rows(dataset_dir / "interval_features.csv")
    window_rows = load_rows(label_dir / "window_labels.csv")
    metrics_rows = load_rows(eval_dir / "model_comparison.csv")
    feature_columns = [str(column) for column in manifest["feature_columns"]]
    validate_no_forbidden(feature_columns)
    validate_no_forbidden(interval_rows[0].keys() if interval_rows else [])
    if merge_summary["manifest_run_count"] != merge_summary["merged_run_count"]:
        raise ValueError("Manifest and merged run counts do not match.")

    plot_correlation(interval_rows, feature_columns, figure_dir)
    plot_phase_timeline(window_rows, figure_dir)
    plot_phase_distribution(window_rows, figure_dir)
    features, tensor_feature_columns = window_feature_matrix(dataset_dir, window_rows)
    plot_phase_distance(features, window_rows, figure_dir)
    plot_workload_similarity(window_rows, figure_dir)
    plot_variance_reduction(features, tensor_feature_columns, window_rows, figure_dir)
    plot_baseline_metrics(metrics_rows, figure_dir)
    _, best_model = plot_confusion(eval_dir, metrics_rows, figure_dir)

    summary = summarize(window_rows, interval_rows)
    write_bib(output_dir / "references.bib")
    tex = latex_report(
        manifest=manifest,
        merge_summary=merge_summary,
        summary=summary,
        metrics_rows=metrics_rows,
        best_model=best_model,
        feature_columns=feature_columns,
    )
    (output_dir / "phase_signatures.tex").write_text(tex, encoding="utf-8")
    (output_dir / "build_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figures": sorted(path.name for path in figure_dir.glob("*.pdf")),
                "best_baseline_for_confusion": best_model,
                "manifest_run_count": merge_summary["manifest_run_count"],
                "merged_run_count": merge_summary["merged_run_count"],
                "interval_rows": manifest["interval_rows"],
                "windows": manifest["windows"],
                "phase_count": summary["phase_count"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote LaTeX report sources to {output_dir}")


if __name__ == "__main__":
    main()
