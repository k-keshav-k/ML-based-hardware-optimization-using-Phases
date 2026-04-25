"""Small IO helpers used across the pipeline."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_").lower()


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows and fieldnames is None:
        fieldnames = []
    if fieldnames is None:
        ordered = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"<not counted>", "<not supported>"}:
        return math.nan
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if denominator in (0, 0.0) or math.isnan(denominator):
        return math.nan
    if math.isnan(numerator):
        return math.nan
    return (numerator / denominator) * scale


def quantile_clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return values.copy()
    lower = np.quantile(clean, low)
    upper = np.quantile(clean, high)
    clipped = values.copy()
    mask = ~np.isnan(clipped)
    clipped[mask] = np.clip(clipped[mask], lower, upper)
    return clipped


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def render_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider] + body)


def listify_csv_argument(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def flatten(iterable: Iterable[Iterable[object]]) -> list[object]:
    merged: list[object] = []
    for chunk in iterable:
        merged.extend(chunk)
    return merged
