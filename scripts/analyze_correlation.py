#!/usr/bin/env python3
"""Run correlation analysis and generate figures/tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.analysis_lib import analyze_all_views
from hpc_phase_analysis.constants import PROCESSED_RESULTS_DIR, RESULTS_DIR
from hpc_phase_analysis.io_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-dir", default=str(PROCESSED_RESULTS_DIR / "preprocessed"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "tables"))
    args = parser.parse_args()
    preprocess_dir = Path(args.preprocess_dir)
    output_dir = Path(args.output_dir)
    summary = analyze_all_views(preprocess_dir, output_dir, read_json(preprocess_dir / "preprocess_summary.json"))
    print(f"Analysis complete. Recommendations: {summary['recommendations_path']}")


if __name__ == "__main__":
    main()
