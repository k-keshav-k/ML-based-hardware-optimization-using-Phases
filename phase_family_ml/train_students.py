"""CLI for distilled student model training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hpc_phase_analysis.io_utils import write_csv_rows

from .config import load_config
from .orchestration import experiment_dirs, scopes_for_experiment
from .students import train_students_for_experiment


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--sequences-root", default="")
    parser.add_argument("--teacher-root", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = load_config(args.config or None)
    dataset_cfg = config["dataset"]
    sequences_root = Path(args.sequences_root or (Path(dataset_cfg["output_dir"]) / "counter_sequences"))
    teacher_root = Path(args.teacher_root or (Path(dataset_cfg["output_dir"]) / "teacher"))
    output_root = Path(args.output_dir or (Path(dataset_cfg["output_dir"]) / "students"))
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[students] config={args.config or '<default>'}", flush=True)
    print(f"[students] sequences_root={sequences_root}", flush=True)
    print(f"[students] teacher_root={teacher_root}", flush=True)
    print(f"[students] output_root={output_root}", flush=True)
    print("[students] device=cpu", flush=True)

    all_rows: list[dict[str, object]] = []
    exp_dirs = experiment_dirs(sequences_root)
    print(f"[students] experiments={len(exp_dirs)}", flush=True)
    for exp_dir in exp_dirs:
        scopes = scopes_for_experiment(exp_dir)
        print(f"[students] experiment={exp_dir.name} scopes={','.join(scopes) or '<none>'}", flush=True)
        for scope in scopes:
            teacher_predictions = teacher_root / exp_dir.name / scope / "teacher_predictions.csv"
            if not teacher_predictions.exists():
                print(f"[students] skip missing_teacher_predictions={teacher_predictions}", flush=True)
                continue
            scope_output = output_root / exp_dir.name / scope
            print(
                f"[students] start experiment={exp_dir.name} scope={scope} teacher_predictions={teacher_predictions}",
                flush=True,
            )
            rows = train_students_for_experiment(
                experiment_dir=exp_dir,
                scope=scope,
                teacher_predictions_path=teacher_predictions,
                output_dir=scope_output,
                horizon=int(dataset_cfg["horizon"]),
                history_length=int(dataset_cfg["history_length"]),
                blend_alpha=float(config["student"]["blend_alpha"]),
                tree_max_depth=int(config["student"]["decision_tree_max_depth"]),
                tree_min_leaf=int(config["student"]["decision_tree_min_samples_leaf"]),
                run_length_buckets=[int(item) for item in config["student"]["run_length_buckets"]],
                synthetic_examples_per_family=int(config["student"].get("synthetic_examples_per_family", 0)),
                synthetic_mutation_rate=float(config["student"].get("synthetic_mutation_rate", 0.05)),
                seed=int(config["random_seed"]),
            )
            for row in rows:
                item = dict(row)
                item["experiment"] = exp_dir.name
                all_rows.append(item)
            print(f"[students] done experiment={exp_dir.name} scope={scope} summary_rows={len(rows)}", flush=True)
    write_csv_rows(output_root / "student_summary_all.csv", all_rows)
    print(f"[students] wrote_summary={output_root / 'student_summary_all.csv'} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
