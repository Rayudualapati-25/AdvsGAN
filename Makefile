PYTHON ?= python3

.PHONY: test smoke baseline pipeline demo

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m pytest tests/smoke -q

baseline:
	$(PYTHON) -m biofake.cli train detector --config configs/experiments/screenshot_baseline.yaml

pipeline:
	$(PYTHON) -m biofake.cli pipeline run --config configs/experiments/full_report.yaml

demo:
	streamlit run src/biofake/demo/app.py -- --config configs/experiments/robust_hybrid.yaml

