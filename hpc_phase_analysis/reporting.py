"""Markdown and HTML report generation."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path

import markdown

from .constants import COUNTER_FAMILIES
from .io_utils import load_csv_rows, read_json, render_markdown_table, safe_float


REPORT_CSS = """
body { font-family: Georgia, 'Times New Roman', serif; margin: 1.5rem auto; max-width: 1100px; color: #1f2933; line-height: 1.6; }
h1, h2, h3 { color: #102a43; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; table-layout: fixed; }
th, td { border: 1px solid #d9e2ec; padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; overflow-wrap: anywhere; font-size: 0.94rem; }
th { background: #f0f4f8; }
img { max-width: 100%; border: 1px solid #d9e2ec; margin: 1rem 0; }
code { background: #f0f4f8; padding: 0.1rem 0.3rem; }
"""

SYNTHETIC_DESCRIPTIONS = {
    "compute": "integer-heavy recurrence loop",
    "memory": "large streaming memory kernel",
    "cache": "random-access cache-thrashing kernel",
    "branch": "unpredictable branch-heavy kernel",
    "fp": "sqrt-heavy floating-point kernel",
    "mixed": "phased composition of compute, memory, branch, fp, and cache-thrashing behavior",
}

DERIVED_DISPLAY = {
    "metric__cpi": "CPI",
    "metric__ipc": "IPC",
    "metric__branch_mpki": "Branch MPKI",
    "metric__branch_per_ki": "Branches / KI",
    "metric__l2_mpki": "L2 MPKI",
    "metric__llc_mpki": "LLC MPKI",
    "metric__offcore_read_mpki": "Off-core read MPKI",
    "metric__fp_per_ki": "FP ops / KI",
}


def image_block(path: str, title: str, asset_prefix: str) -> str:
    prefixed = f"{asset_prefix}{path}" if path else ""
    return f"### {title}\n\n![{title}]({prefixed})\n"


def supported_note(family: str, metadata: dict[str, object]) -> str:
    if not metadata.get("supported", False):
        return "Unsupported on this host"
    if metadata.get("selection_kind") == "derived_uncore_total":
        return "Derived from system-wide Intel uncore IMC read+write"
    if metadata.get("collection_scope") == "system_wide_uncore":
        return "Preferred Intel uncore IMC event (system-wide)"
    generic_candidates = COUNTER_FAMILIES.get(family, {}).get("preferred", {}).get("generic", [])
    if str(metadata.get("selected_via", "")) in generic_candidates:
        return "Using generic perf fallback"
    return "Using preferred host event"


def compact_event_mapping_rows(alias_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for family, metadata in alias_map.items():
        if not metadata.get("analysis_confident", False):
            continue
        rows.append(
            {
                "Family": metadata.get("label", family),
                "Selected Event": metadata.get("selected_event", "") or "-",
                "Supported": "Yes" if metadata.get("supported", False) else "No",
                "Note": supported_note(family, metadata),
            }
        )
    return rows


def display_indicator(indicator: str) -> str:
    if indicator in DERIVED_DISPLAY:
        return DERIVED_DISPLAY[indicator]
    if indicator.startswith("metric__"):
        return indicator.replace("metric__", "").replace("_", " ").title()

    prefix = ""
    family_name = indicator
    if indicator.startswith("counter__"):
        prefix = "counter"
        family_name = indicator[len("counter__") :]
    elif indicator.startswith("norm__"):
        prefix = "norm"
        family_name = indicator[len("norm__") :]

    suffix = ""
    if family_name.endswith("_per_ms"):
        suffix = " / ms"
        family_name = family_name[: -len("_per_ms")]
    elif family_name.endswith("_per_ki"):
        suffix = " / KI"
        family_name = family_name[: -len("_per_ki")]

    label = COUNTER_FAMILIES.get(family_name, {}).get("label", family_name.replace("_", " ").title())
    if prefix == "norm":
        return f"{label}{suffix}"
    return str(label)


def display_indicator_list(values: str) -> str:
    indicators = [item.strip() for item in values.split(",") if item.strip()]
    return ", ".join(display_indicator(item) for item in indicators)


def recommendation_blocks(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No high-correlation groups met the recommendation threshold._"
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"### {row['group_id']}",
                    "",
                    f"- Keep: `{display_indicator(row['representative_to_keep'])}`",
                    f"- Drop: {display_indicator_list(row['indicators_to_drop'])}",
                    f"- Members: {display_indicator_list(row['candidate_indicators'])}",
                    f"- Reason: {row['reason']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_drop_reasons(filter_summary: dict[str, object]) -> str:
    reasons = filter_summary.get("dropped_reasons", {})
    if not isinstance(reasons, dict):
        return "n/a"
    parts = []
    for key, value in reasons.items():
        label = key.replace("_", " ")
        parts.append(f"{label}: {value}")
    return ", ".join(parts) if parts else "n/a"


def load_manifest_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    manifest = read_json(path)
    return manifest if isinstance(manifest, list) else []


def duration_stats(manifest_entries: list[dict[str, object]], field: str) -> str:
    durations = []
    for entry in manifest_entries:
        run_dir = Path(str(entry.get("run_dir", "")))
        results_path = run_dir / "collection_results.json"
        if results_path.exists():
            results = read_json(results_path)
            value = safe_float(results.get(field, ""))
            if value == value:
                durations.append(value)
    if not durations:
        return ""
    return f"{statistics.median(durations):.1f} ms median ({min(durations):.1f} min, {max(durations):.1f} max)"


def mixed_phase_overshoot_stats(manifest_entries: list[dict[str, object]]) -> str:
    overshoots = []
    for entry in manifest_entries:
        if entry.get("suite") != "synthetic" or entry.get("workload") != "mixed":
            continue
        requested_duration = safe_float(entry.get("requested_duration_ms", ""))
        run_dir = Path(str(entry.get("run_dir", "")))
        phase_log_path = run_dir / "phase_log.csv"
        if requested_duration != requested_duration or not phase_log_path.exists():
            continue
        rows = load_csv_rows(phase_log_path)
        if not rows:
            continue
        actual_end = safe_float(rows[-1].get("end_ms", ""))
        if actual_end == actual_end:
            overshoots.append(actual_end - requested_duration)
    if not overshoots:
        return ""
    return f"{statistics.median(overshoots):.1f} ms median ({min(overshoots):.1f} min, {max(overshoots):.1f} max)"


def workload_generation_section(manifest_entries: list[dict[str, object]], platform_info: dict[str, object]) -> str:
    if not manifest_entries:
        return "Workload-generation metadata was not available when this report was built."

    suites = sorted({str(entry.get("suite", "")) for entry in manifest_entries if entry.get("suite", "")})
    workloads = sorted({str(entry.get("workload", "")) for entry in manifest_entries if entry.get("workload", "")})
    threads = sorted({int(entry.get("threads", 0)) for entry in manifest_entries if entry.get("threads", 0)})
    combo_counts = Counter((str(entry.get("suite", "")), str(entry.get("workload", "")), int(entry.get("threads", 0))) for entry in manifest_entries)
    repetition_counts = sorted(set(combo_counts.values()))
    interval_targets = sorted({int(entry.get("collection_interval_ms", 0)) for entry in manifest_entries if entry.get("collection_interval_ms")})
    duration_targets = sorted({int(entry.get("requested_duration_ms", 0)) for entry in manifest_entries if entry.get("requested_duration_ms")})
    phase_targets = sorted({int(entry.get("requested_phase_ms", 0)) for entry in manifest_entries if entry.get("requested_phase_ms")})
    input_sizes = sorted({str(entry.get("requested_input_size", "")) for entry in manifest_entries if entry.get("requested_input_size")})
    parsec_available = platform_info.get("parsec", {}).get("available", False)
    spec_available = platform_info.get("spec2017", {}).get("available", False)
    affinity_note = "Pinned with `taskset` to the first N online CPUs for the requested thread count."
    if not any(entry.get("affinity") for entry in manifest_entries):
        affinity_note = "Affinity metadata was not recorded."

    lines = [
        f"- Suites in this generated artifact: `{', '.join(suites)}`.",
        f"- PARSEC available on this host: `{parsec_available}`. SPEC CPU2017 available on this host: `{spec_available}`.",
        f"- Workloads executed: `{', '.join(workloads)}`.",
        f"- Thread counts executed: `{', '.join(str(thread) for thread in threads)}`.",
    ]
    if len(repetition_counts) == 1:
        lines.append(f"- Repetitions per workload/thread combination: `{repetition_counts[0]}`.")
    else:
        lines.append(f"- Repetition counts observed across combinations: `{', '.join(str(count) for count in repetition_counts)}`.")
    if interval_targets:
        lines.append(f"- Target collection interval: `{', '.join(str(value) for value in interval_targets)} ms`.")
    if input_sizes:
        lines.append(f"- PARSEC input sizes used: `{', '.join(input_sizes)}`.")
    if duration_targets:
        lines.append(f"- Target synthetic run duration: `{', '.join(str(value) for value in duration_targets)} ms`.")
    if phase_targets:
        lines.append(f"- Target mixed-phase slice duration: `{', '.join(str(value) for value in phase_targets)} ms`.")

    observed_interval_duration = duration_stats(manifest_entries, "interval_actual_duration_ms")
    if observed_interval_duration:
        lines.append(f"- Observed perf-wrapped interval collection wall time: `{observed_interval_duration}`.")
    mixed_overshoot = mixed_phase_overshoot_stats(manifest_entries)
    if mixed_overshoot:
        lines.append(f"- Observed mixed-workload overshoot from phase logs: `{mixed_overshoot}`.")
    lines.append(f"- Affinity policy: {affinity_note}")

    for workload in workloads:
        if workload in SYNTHETIC_DESCRIPTIONS:
            lines.append(f"- `{workload}` workload generation: {SYNTHETIC_DESCRIPTIONS[workload]}.")
    if "parsec" in suites:
        lines.append("- `parsec` workload generation: standard PARSEC applications and kernels launched through `parsecmgmt` with taskset-based CPU affinity and the configured PARSEC input set.")
    return "\n".join(lines)


def executive_summary_text(manifest_entries: list[dict[str, object]], platform_info: dict[str, object]) -> str:
    suites = sorted({str(entry.get("suite", "")) for entry in manifest_entries if entry.get("suite", "")})
    if not suites:
        suite_note = "The current artifact did not include recorded workload-suite metadata."
    elif suites == ["synthetic"]:
        parsec_available = platform_info.get("parsec", {}).get("available", False)
        spec_available = platform_info.get("spec2017", {}).get("available", False)
        if not parsec_available and not spec_available:
            suite_note = "On this machine, the generated artifact is based on the synthetic suite only because PARSEC and SPEC CPU2017 were not detected."
        else:
            suite_note = "On this machine, the generated artifact is based on the synthetic suite only."
    elif suites == ["parsec"]:
        suite_note = "On this machine, the generated artifact is based on PARSEC workloads collected through `parsecmgmt`."
    else:
        suite_note = f"On this machine, the generated artifact includes these workload suites: `{', '.join(suites)}`."
    return (
        "This project collects Linux performance-counter measurements, aligns them with workload metadata and, when available, synthetic phase labels, "
        "and studies correlation structure to identify redundant indicators. The main recommendation workflow prefers the normalized Spearman view because "
        "raw counters often correlate simply because total activity rose across a sample, not because two indicators capture the same underlying behavior. "
        f"{suite_note}"
    )


def pmu_readiness(platform_info: dict[str, object], preprocess_summary: dict[str, object]) -> dict[str, object]:
    readiness = platform_info.get("pmu_study_readiness", {})
    if isinstance(readiness, dict) and readiness:
        return readiness
    return {
        "study_ready_core": preprocess_summary.get("study_ready_core", False),
        "study_ready_uncore": preprocess_summary.get("study_ready_uncore", False),
        "study_ready": preprocess_summary.get("study_ready", False),
        "uncore_policy_blocked": preprocess_summary.get("uncore_policy_blocked", False),
        "missing_required_confident_families": preprocess_summary.get("missing_required_confident_families", []),
        "strict_hardware_study_reason": preprocess_summary.get("strict_hardware_study_reason", ""),
        "uncore_reason": preprocess_summary.get("uncore_reason", ""),
    }


def stratified_results_text(manifest_entries: list[dict[str, object]]) -> str:
    workloads = {str(entry.get("workload", "")) for entry in manifest_entries if entry.get("workload", "")}
    if "mixed" in workloads:
        return (
            "The report also breaks correlations down by workload, thread count, and phase label whenever enough samples are available. "
            "This matters because two counters can look redundant overall but behave differently inside a specific phase."
        )
    return (
        "The report also checks correlations separately within each workload and thread count whenever enough samples are available. "
        "Because PARSEC does not provide explicit phase labels here, this workload-by-workload view is the main test of whether a redundancy pattern is stable across applications."
    )


def build_report_markdown(
    platform_info: dict[str, object],
    alias_map: dict[str, dict[str, object]],
    preprocess_summary: dict[str, object],
    analysis_summary: dict[str, object],
    merged_summary: dict[str, object],
    recommendations_rows: list[dict[str, str]],
    manifest_entries: list[dict[str, object]],
    asset_prefix: str,
) -> str:
    raw_view = analysis_summary["views"].get("raw", {})
    norm_view = analysis_summary["views"].get("normalized", {})
    derived_view = analysis_summary["views"].get("derived", {})
    readiness = pmu_readiness(platform_info, preprocess_summary)
    study_ready_core = bool(readiness.get("study_ready_core", False))
    study_ready_uncore = bool(readiness.get("study_ready_uncore", False))
    uncore_policy_blocked = bool(readiness.get("uncore_policy_blocked", False))
    missing_required = readiness.get("missing_required_confident_families", [])
    mapping_rows = compact_event_mapping_rows(alias_map)
    mapping_table = render_markdown_table(mapping_rows, ["Family", "Selected Event", "Supported", "Note"]) if mapping_rows else "_No non-generic vendor-preferred counters were available on this host._"
    dataset_rows = [
        {"Statistic": "Interval rows", "Value": merged_summary.get("interval_rows", 0)},
        {"Statistic": "Aggregate rows", "Value": merged_summary.get("aggregate_rows", 0)},
        {"Statistic": "Runs merged", "Value": merged_summary.get("merged_run_count", merged_summary.get("run_count", 0))},
        {"Statistic": "Manifest runs", "Value": merged_summary.get("manifest_run_count", 0)},
        {"Statistic": "Stale raw runs excluded", "Value": merged_summary.get("filtered_stale_run_count", 0)},
        {"Statistic": "Observed raw counter columns", "Value": len(preprocess_summary.get("raw_counter_columns_initial", []))},
        {"Statistic": "Confident counter columns kept", "Value": len(preprocess_summary.get("raw_counter_columns_kept", []))},
        {"Statistic": "Generic fallback columns excluded", "Value": len(preprocess_summary.get("excluded_generic_fallback_columns", []))},
        {"Statistic": "Columns removed as constant", "Value": len(preprocess_summary.get("removed_constant_columns", []))},
        {"Statistic": "Strict core PMU study ready", "Value": "Yes" if study_ready_core else "No"},
        {"Statistic": "Uncore IMC study ready", "Value": "Yes" if study_ready_uncore else "No"},
        {"Statistic": "Uncore blocked by host policy", "Value": "Yes" if uncore_policy_blocked else "No"},
        {"Statistic": "Observed system-wide uncore columns", "Value": len(preprocess_summary.get("observed_system_wide_uncore_columns", []))},
    ]
    dataset_table = render_markdown_table(dataset_rows, ["Statistic", "Value"])
    raw_plots = raw_view.get("plots", {})
    norm_plots = norm_view.get("plots", {})
    filter_summary = preprocess_summary.get("analysis_row_filter", {})
    confident_columns_kept = preprocess_summary.get("raw_counter_columns_kept", [])
    generic_excluded = preprocess_summary.get("excluded_generic_fallback_columns", [])
    unsupported_excluded = preprocess_summary.get("excluded_unsupported_families", [])
    analysis_ready = bool(confident_columns_kept)
    correlation_text = (
        "Raw counters show which indicators move together at the absolute-count level, but these relationships may largely reflect workload intensity. "
        "Normalized counters reduce that effect and are the better basis for redundancy decisions."
    )
    if not study_ready_core:
        correlation_text = (
            "This host is unsuitable for the strict hardware-first PMU study in its current configuration. "
            f"{readiness.get('strict_hardware_study_reason', '') or 'Required confident core families were not available.'} "
            "No correlation conclusions should be drawn from this machine until those vendor-specific counters are exposed."
        )
    elif not analysis_ready:
        correlation_text = (
            "The host exposed confident PMU families, but no confident counters were retained in the merged dataset for this particular run batch, so the analysis "
            "intentionally omits graphs and recommendations rather than plotting generic fallback counters with uncertain semantic equivalence."
        )
    elif uncore_policy_blocked:
        correlation_text += (
            " This Xeon host exposes Intel uncore IMC PMUs, but system-wide collection is currently blocked by host policy, so the present artifact reflects "
            "confident core PMU analysis only."
        )
    if not study_ready_core:
        recommendation_text = "_No recommendations were generated because this host did not satisfy the strict core PMU readiness requirements._"
    elif analysis_ready:
        recommendation_text = recommendation_blocks(recommendations_rows)
    else:
        recommendation_text = "_No recommendations were generated because no confident non-generic counters were retained in this run batch._"
    preprocessing_filter_text = (
        f"- Analysis-ready rows kept: `{filter_summary.get('kept_rows', 0)}` of `{filter_summary.get('input_rows', 0)}`.\n"
        f"- Dropped rows: `{filter_summary.get('dropped_rows', 0)}`.\n"
        f"- Drop reasons: `{format_drop_reasons(filter_summary)}`."
    )
    if not analysis_ready:
        preprocessing_filter_text = "- No analysis-ready rows were retained because no confident non-generic counters were available on this host."
    raw_indicator_summary = ", ".join(display_indicator(column) for column in preprocess_summary.get("raw_counter_columns_kept", [])) or "none"

    return f"""# Multicore Hardware Counter Correlation Study

## Executive Summary

{executive_summary_text(manifest_entries, platform_info)}

## Objective

Determine which hardware indicators are highly correlated or redundant so a later phase-classification and phase-prediction pipeline can rely on a smaller, cheaper, and more portable counter set.

## Experimental Setup

- CPU vendor/model: `{platform_info.get("cpu_vendor", "")}` / `{platform_info.get("cpu_model", "")}`
- Architecture: `{platform_info.get("architecture", "")}`
- Logical CPUs: `{platform_info.get("logical_cpus", "")}`
- Sockets: `{platform_info.get("socket_count", "")}`
- Cores per socket: `{platform_info.get("cores_per_socket", "")}`
- Perf version: `{platform_info.get("perf", {}).get("version", "")}`
- Intel PCM available: `{platform_info.get("intel_pcm", {}).get("available", False)}`
- CPU family/model id: `{platform_info.get("cpu_family", "")}` / `{platform_info.get("cpu_model_id", "")}`
- Strict core PMU study ready: `{study_ready_core}`
- Uncore IMC study ready: `{study_ready_uncore}`

## Workload Generation

{workload_generation_section(manifest_entries, platform_info)}

## Event Mapping Table

{mapping_table}

- Confident counters retained for analysis: `{", ".join(display_indicator(f"counter__{family}") for family in preprocess_summary.get("confident_families", [])) or "none"}`.
- Confident counters exposed on this host: `{", ".join(display_indicator(f"counter__{family}") for family in preprocess_summary.get("confident_families_exposed", [])) or "none"}`.
- Generic fallback counters excluded from analysis: `{", ".join(display_indicator(column) for column in generic_excluded) or "none"}`.
- Unsupported counter families excluded from analysis: `{", ".join(display_indicator(f"counter__{family}") for family in unsupported_excluded) or "none"}`.
- Missing required strict-study families: `{", ".join(str(item) for item in missing_required) or "none"}`.
- Uncore readiness note: `{readiness.get("uncore_reason", "No uncore note recorded.")}`.

## Dataset Summary

{dataset_table}

## Preprocessing and Derived Metrics

The preprocessing stage handles missing values with median imputation after logging missingness, removes constant columns, creates raw and winsorized views, and derives CPI, IPC, MPKI-style cache/branch/off-core metrics, and FP intensity metrics. Raw-counter and normalized-counter views are intentionally separated because raw counts are strongly scale-sensitive. For normalized and derived analysis views, samples with non-positive interval duration, instructions retired, or cycles are excluded to avoid startup/shutdown artifacts.

{preprocessing_filter_text}

## Global Correlation Results

{correlation_text}

{image_block(raw_plots.get("pearson", ""), "Raw Pearson Heatmap", asset_prefix) if raw_plots.get("pearson") else ""}
{image_block(raw_plots.get("spearman", ""), "Raw Spearman Heatmap", asset_prefix) if raw_plots.get("spearman") else ""}
{image_block(norm_plots.get("pearson", ""), "Normalized Pearson Heatmap", asset_prefix) if norm_plots.get("pearson") else ""}
{image_block(norm_plots.get("spearman", ""), "Normalized Spearman Heatmap", asset_prefix) if norm_plots.get("spearman") else ""}
{image_block(norm_plots.get("clustered", ""), "Clustered Heatmap", asset_prefix) if norm_plots.get("clustered") else ""}
{image_block(norm_plots.get("dendrogram", ""), "Dendrogram", asset_prefix) if norm_plots.get("dendrogram") else ""}

## Stratified Results

{stratified_results_text(manifest_entries)}

{image_block(norm_plots.get("top_correlations", ""), "Top Positive and Negative Correlations", asset_prefix) if norm_plots.get("top_correlations") else ""}

## Multicollinearity Results

Variance inflation factors answer a different question from pairwise correlation: does a counter still add information once the rest of the counter set is already known? A counter can avoid extreme pairwise correlations and still be redundant in combination with several others, so VIF should be used together with the correlation plots rather than instead of them.

{image_block(norm_plots.get("vif", ""), "VIF Bar Chart", asset_prefix) if norm_plots.get("vif") else ""}

## Feature Selection Recommendations

{recommendation_text}

The full machine-readable recommendation table remains available in `{asset_prefix}tables/recommendations.csv`.

## Limitations

- Raw counters can correlate simply because all counts grow with activity, sample duration, or thread count.
- Vendor-specific PMU names differ substantially; unsupported counters are logged and left blank.
- Intel PCM support is best-effort because deployments vary by binary name, permissions, and CLI syntax.
- Per-core task-local attribution is limited without privileged system-wide PMU access.
- Intel uncore IMC bandwidth on this host requires system-wide perf access; when `perf_event_paranoid` blocks that path, the report explicitly omits those system-wide metrics.

## Next Steps

- Validate the recommended reduced set against a downstream phase classifier or predictor.
- Repeat the study on additional Intel, AMD, and Arm systems to compare portability.
- Extend the stratified analysis with application-specific phases from PARSEC or SPEC CPU2017 when those suites are available.
- Lower `perf_event_paranoid` or grant equivalent perf capability on this Xeon host if you want IMC bandwidth collected alongside the strict core PMU set.

## Appendix

- Raw indicators analyzed: `{raw_indicator_summary}`
- Derived indicators analyzed: `{derived_view.get("indicator_count", 0)}`
- Counter families defined in the project: `{", ".join(COUNTER_FAMILIES.keys())}`

## How to interpret the results

- Two counters are likely redundant when they remain highly correlated in the normalized Spearman view and also appear in the same high-VIF neighborhood.
- Raw and normalized views can disagree because raw counts mix true behavior with overall workload activity and duration scaling.
- Use correlation to find pairwise or group redundancy, then use VIF to see whether an indicator adds information once the rest of the set is considered.
- Choose a minimal portable counter set by preferring common counters with low missingness and clear meaning, then dropping vendor-specific counters only when a more portable representative preserves the same signal.
"""


def write_report(
    platform_path: Path,
    alias_map_path: Path,
    preprocess_summary_path: Path,
    analysis_summary_path: Path,
    merged_summary_path: Path,
    recommendations_path: Path,
    manifest_path: Path,
    markdown_path: Path,
    html_path: Path,
    asset_prefix: str = "",
) -> None:
    platform_info = read_json(platform_path)
    alias_map = read_json(alias_map_path)
    preprocess_summary = read_json(preprocess_summary_path)
    analysis_summary = read_json(analysis_summary_path)
    merged_summary = read_json(merged_summary_path)
    recommendations_rows = load_csv_rows(recommendations_path)
    manifest_entries = load_manifest_entries(manifest_path)
    markdown_text = build_report_markdown(
        platform_info,
        alias_map,
        preprocess_summary,
        analysis_summary,
        merged_summary,
        recommendations_rows,
        manifest_entries,
        asset_prefix,
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html_path.write_text(f"<html><head><meta charset='utf-8'><style>{REPORT_CSS}</style></head><body>{html_body}</body></html>", encoding="utf-8")
