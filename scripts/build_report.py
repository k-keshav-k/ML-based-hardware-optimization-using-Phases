#!/usr/bin/env python3
"""Build the Markdown and HTML report."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc_phase_analysis.constants import PROCESSED_RESULTS_DIR, RESULTS_DIR, TABLES_DIR
from hpc_phase_analysis.reporting import write_report


def build_latex(markdown_path: Path, latex_path: Path, header_path: Path) -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return False
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            pandoc,
            str(markdown_path),
            "--standalone",
            "--to=latex",
            "--resource-path",
            str(markdown_path.parent),
            "-V",
            "papersize:letter",
            "-V",
            "geometry:margin=0.35in",
            "-V",
            "fontsize=9pt",
            "-H",
            str(header_path),
            "-o",
            str(latex_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or result.stdout.strip())
        return False
    return True


def build_pdf(markdown_path: Path, pdf_path: Path, header_path: Path) -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return False
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            pandoc,
            str(markdown_path),
            "--pdf-engine=pdflatex",
            "--resource-path",
            str(markdown_path.parent),
            "-V",
            "papersize:letter",
            "-V",
            "geometry:margin=0.35in",
            "-V",
            "fontsize=9pt",
            "-H",
            str(header_path),
            "-o",
            str(pdf_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or result.stdout.strip())
        return False
    return True


def mirror_outputs(pdf_path: Path | None, mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    if pdf_path is not None and pdf_path.exists():
        shutil.copy2(pdf_path, mirror_dir / pdf_path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=str(RESULTS_DIR / "platform_info.json"))
    parser.add_argument("--alias-map", default=str(RESULTS_DIR / "event_alias_map.json"))
    parser.add_argument("--preprocess-summary", default=str(PROCESSED_RESULTS_DIR / "preprocessed" / "preprocess_summary.json"))
    parser.add_argument("--analysis-summary", default=str(TABLES_DIR / "analysis_summary.json"))
    parser.add_argument("--merge-summary", default=str(PROCESSED_RESULTS_DIR / "merge_summary.json"))
    parser.add_argument("--recommendations", default=str(TABLES_DIR / "recommendations.csv"))
    parser.add_argument("--manifest", default=str(RESULTS_DIR / "run_manifest.json"))
    parser.add_argument("--markdown-output", default=str(RESULTS_DIR / "report.md"))
    parser.add_argument("--html-output", default=str(RESULTS_DIR / "report.html"))
    parser.add_argument("--pdf-output", default=str(RESULTS_DIR / "report.pdf"))
    parser.add_argument("--latex-output", default=str(RESULTS_DIR / "report.tex"))
    parser.add_argument("--mirror-dir", default=str(RESULTS_DIR / "reports"))
    parser.add_argument("--pdf-header", default=str(Path(__file__).resolve().parents[1] / "config" / "pandoc_report_header.tex"))
    args = parser.parse_args()
    markdown_output = Path(args.markdown_output)
    html_output = Path(args.html_output)
    pdf_output = Path(args.pdf_output)
    latex_output = Path(args.latex_output)
    mirror_dir = Path(args.mirror_dir)
    pdf_header = Path(args.pdf_header)
    write_report(
        Path(args.platform),
        Path(args.alias_map),
        Path(args.preprocess_summary),
        Path(args.analysis_summary),
        Path(args.merge_summary),
        Path(args.recommendations),
        Path(args.manifest),
        markdown_output,
        html_output,
        "",
    )
    mirror_markdown = mirror_dir / markdown_output.name
    mirror_html = mirror_dir / html_output.name
    write_report(
        Path(args.platform),
        Path(args.alias_map),
        Path(args.preprocess_summary),
        Path(args.analysis_summary),
        Path(args.merge_summary),
        Path(args.recommendations),
        Path(args.manifest),
        mirror_markdown,
        mirror_html,
        "../",
    )
    latex_built = build_latex(markdown_output, latex_output, pdf_header)
    mirror_latex = mirror_dir / latex_output.name
    if latex_built:
        build_latex(mirror_markdown, mirror_latex, pdf_header)
    pdf_built = build_pdf(markdown_output, pdf_output, pdf_header)
    mirror_outputs(pdf_output if pdf_built else None, mirror_dir)
    targets = [str(markdown_output), str(html_output)]
    if latex_built:
        targets.append(str(latex_output))
    if pdf_built:
        targets.append(str(pdf_output))
    print(f"Wrote report to {', '.join(targets)}")


if __name__ == "__main__":
    main()
