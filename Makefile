PYTHON ?= python3
RAW_DIR ?= results/raw
PROCESSED_DIR ?= results/processed
PREPROCESS_DIR ?= $(PROCESSED_DIR)/preprocessed
TABLES_DIR ?= results/tables

.PHONY: synthetic setup-parsec detect discover collect-synth merge preprocess analyze report paper online-cosched dvfs-stress test example all clean

synthetic:
	mkdir -p synthetic_workloads/bin
	gcc -O2 -pthread -lm synthetic_workloads/phase_bench.c -o synthetic_workloads/bin/phase_bench

setup-parsec:
	$(PYTHON) scripts/setup_parsec.py

detect:
	$(PYTHON) scripts/detect_platform.py

discover:
	$(PYTHON) scripts/discover_events.py

collect-synth: synthetic
	$(PYTHON) scripts/run_workloads.py --suite synthetic --threads 1,2,4,8 --reps 3 --modes interval

merge:
	$(PYTHON) scripts/merge_runs.py --input-dir $(RAW_DIR) --output-dir $(PROCESSED_DIR)

preprocess:
	$(PYTHON) scripts/preprocess.py --input $(PROCESSED_DIR)/merged_interval_dataset.csv --output-dir $(PREPROCESS_DIR)

analyze:
	$(PYTHON) scripts/analyze_correlation.py --preprocess-dir $(PREPROCESS_DIR) --output-dir $(TABLES_DIR)

report:
	$(PYTHON) scripts/build_report.py

paper:
	$(PYTHON) -m pipeline.run_all --config experiments/configs/core_uncore_large.json

online-cosched:
	bash scripts/online/run_burst_coscheduling.sh

dvfs-stress:
	bash scripts/online/run_dvfs_stress.sh

test:
	$(PYTHON) -m unittest discover -s tests -p 'test*.py'

example: detect discover collect-synth merge preprocess analyze report

all: example

clean:
	rm -rf synthetic_workloads/bin
	rm -rf results/raw/* results/processed/* results/tables/* results/reports/* results/plots/* results/logs/*
	find . -path ./.venv -prune -o -path ./third_party -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
