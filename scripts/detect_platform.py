#!/usr/bin/env python3
"""Detect platform information for the study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import RESULTS_DIR
from hpc_phase_analysis.detect import detect_platform
from hpc_phase_analysis.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS_DIR / "platform_info.json"))
    args = parser.parse_args()
    platform = detect_platform()
    write_json(Path(args.output), platform)
    print(f"Wrote platform information to {args.output}")


if __name__ == "__main__":
    main()
