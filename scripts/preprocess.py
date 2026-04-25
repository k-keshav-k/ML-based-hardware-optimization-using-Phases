#!/usr/bin/env python3
"""Preprocess merged datasets for correlation analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import PROCESSED_RESULTS_DIR, RESULTS_DIR
from hpc_phase_analysis.events import load_alias_map
from hpc_phase_analysis.io_utils import read_json
from hpc_phase_analysis.preprocess_lib import preprocess_interval_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROCESSED_RESULTS_DIR / "merged_interval_dataset.csv"))
    parser.add_argument("--output-dir", default=str(PROCESSED_RESULTS_DIR / "preprocessed"))
    parser.add_argument("--alias-map", default=str(RESULTS_DIR / "event_alias_map.json"))
    parser.add_argument("--platform", default=str(RESULTS_DIR / "platform_info.json"))
    args = parser.parse_args()
    alias_map_path = Path(args.alias_map)
    alias_map = load_alias_map(alias_map_path) if alias_map_path.exists() else None
    platform_path = Path(args.platform)
    platform_info = read_json(platform_path) if platform_path.exists() else None
    summary = preprocess_interval_dataset(Path(args.input), Path(args.output_dir), alias_map, platform_info)
    print(f"Preprocessed {summary['input_rows']} rows.")


if __name__ == "__main__":
    main()
