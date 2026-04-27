PYTHON ?= python3
RAW_DIR ?= results/raw_phase_family_ml_experiments
PROCESSED_DIR ?= results/processed
FAMILY_DIR ?= results/phase_family_ml

.PHONY: setup-parsec detect discover collect merge family-pipeline test clean

setup-parsec:
	$(PYTHON) scripts/setup_parsec.py

detect:
	$(PYTHON) scripts/detect_platform.py

discover:
	$(PYTHON) scripts/discover_events.py

collect:
	$(PYTHON) -m phase_family_ml.collect --output-dir $(RAW_DIR)

merge:
	$(PYTHON) scripts/merge_runs.py --input-dir $(RAW_DIR) --output-dir $(PROCESSED_DIR)

family-pipeline:
	$(PYTHON) -m phase_family_ml.run_pipeline --config config/phase_family_ml_defaults.json --output-dir $(FAMILY_DIR)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_phase_family_ml.py'

clean:
	rm -rf $(PROCESSED_DIR)/* $(FAMILY_DIR)/*
	find . -path ./.venv -prune -o -path ./third_party -prune -o -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
