#!/usr/bin/env python3
"""Discover supported perf events and build the semantic alias map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import RESULTS_DIR
from hpc_phase_analysis.detect import detect_platform
from hpc_phase_analysis.events import build_alias_map, compute_study_readiness, discover_perf_list_output, extract_event_aliases, write_discovery_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--strict-hardware-study", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    platform = detect_platform()
    perf_list_output = discover_perf_list_output()
    aliases = extract_event_aliases(str(perf_list_output["stderr"]) + str(perf_list_output["stdout"]))
    alias_map = build_alias_map(platform, aliases)
    study_readiness = compute_study_readiness(platform, alias_map)
    write_discovery_artifacts(output_dir, platform, perf_list_output, alias_map, study_readiness)
    supported = sum(1 for row in alias_map.values() if row["supported"])
    print(f"Discovered {supported}/{len(alias_map)} semantic counter families.")
    if args.strict_hardware_study and not study_readiness["study_ready_core"]:
        raise SystemExit(str(study_readiness["strict_hardware_study_reason"]))


if __name__ == "__main__":
    main()
