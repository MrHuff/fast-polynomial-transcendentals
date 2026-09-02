.PHONY: test doctor doctor-train check-downstream audit-release verify-artifacts verify-manifest plot-pretraining

PYTHON ?= python3

test:
	$(PYTHON) -m pytest

doctor:
	$(PYTHON) -m sfu_repro.doctor --profile component

doctor-train:
	$(PYTHON) -m sfu_repro.doctor --profile train

check-downstream:
	$(PYTHON) scripts/check_open_weight_environment.py --models paper

audit-release:
	$(PYTHON) scripts/audit_release_tree.py

verify-artifacts:
	$(PYTHON) scripts/verify_artifacts.py

verify-manifest:
	$(PYTHON) scripts/validate_manifest.py

plot-pretraining:
	$(PYTHON) evidence/pretraining/plot_paired_loss.py
