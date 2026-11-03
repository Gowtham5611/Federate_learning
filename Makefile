# FedSecureAI Automation Makefile
.PHONY: eval posthoc serve test verify-hashes build-frontend all

PYTHON ?= .venv/Scripts/python
UVICORN ?= .venv/Scripts/uvicorn
PYTEST ?= .venv/Scripts/pytest

eval:
	$(PYTHON) scripts/export_test_slices.py
	$(PYTEST) tests/test_evaluation_metrics.py tests/test_serving_sequence_parity.py -v

posthoc:
	$(PYTHON) scripts/run_posthoc_analysis.py

serve:
	$(UVICORN) interfaces.api.app:create_app --factory --host 0.0.0.0 --port 8000

test:
	$(PYTEST) tests/test_evaluation_metrics.py tests/test_serving_sequence_parity.py -v

verify-hashes:
	$(PYTHON) scripts/verify_baseline_hashes.py

build-frontend:
	cd frontend && npm run build

all: verify-hashes eval posthoc test

