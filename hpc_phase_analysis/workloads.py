"""PARSEC command builders."""

from __future__ import annotations


def build_parsec_command(parsecmgmt: str, benchmark: str, threads: int, input_size: str = "native") -> list[str]:
    return [parsecmgmt, "-a", "run", "-p", benchmark, "-n", str(threads), "-i", input_size]
