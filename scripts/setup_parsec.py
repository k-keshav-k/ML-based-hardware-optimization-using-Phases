#!/usr/bin/env python3
"""Bootstrap a local PARSEC checkout from the cirosantilli mirror."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import PROJECT_ROOT


def run_checked(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def listify_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def clone_if_needed(repo_url: str, root: Path) -> None:
    if root.exists():
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(root)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="https://github.com/cirosantilli/parsec-benchmark.git")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "third_party" / "parsec-benchmark"))
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--download-native-inputs", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    clone_if_needed(args.repo_url, root)
    run_checked(["./configure"], root)

    if args.download_native_inputs:
        run_checked(["./get-inputs", "-n"], root)

    benchmarks = listify_csv(args.benchmarks)
    if benchmarks:
        run_checked([str(root / "bin" / "parsecmgmt"), "-a", "build", "-p", *benchmarks], root)

    print(root)


if __name__ == "__main__":
    main()
