.PHONY: test doctor audit-release verify-artifacts verify-manifest plot-pretraining

PYTHON ?= python3

test:
	$(PYTHON) -m pytest

doctor:
	$(PYTHON) -m sfu_repro.doctor --profile component

audit-release:
	$(PYTHON) scripts/audit_release_tree.py

verify-artifacts:
	$(PYTHON) scripts/verify_artifacts.py

verify-manifest:
	$(PYTHON) scripts/validate_manifest.py

plot-pretraining:
	$(PYTHON) evidence/pretraining/plot_paired_loss.py
