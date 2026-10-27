# Shortcuts for docs/RUNBOOK.md. Stages that need a model say so; everything else runs on
# cached artifacts. Override MODEL / runs on the command line, e.g.
#   make exp1 VAL=results/<exp1 val run> TEST=results/<exp1 test run>

MODEL ?= Qwen/Qwen3.5-2B-Base
PY := uv run python

.PHONY: help setup test test-all lint check figures figures-smoke theory-synthetic \
        theory-neutral neutral exp1 exp2-h2 exp2-h3 exp2-robustness exp3-readout exp3-jspace

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup:  ## install the environment (uv.lock)
	uv sync

test:  ## fast tests (no model weights)
	uv run pytest -m "not slow" -q

test-all:  ## all tests incl. model/lens tests (downloads weights)
	uv run pytest -q

lint:  ## ruff
	uv run ruff check src scripts tests

check: lint test  ## lint + fast tests + ticket consistency
	python3 scripts/tickets.py check

figures:  ## report figures from figures/manifest.yaml (no model)
	$(PY) scripts/make_figures.py

figures-smoke:  ## plumbing check on the smoke runs -> figures/smoke/
	$(PY) scripts/make_figures.py --manifest figures/manifest_smoke.yaml

theory-synthetic:  ## T-022: theory vs Monte Carlo on synthetic manifolds (no model)
	$(PY) scripts/validate_theory_synthetic.py

theory-neutral:  ## T-023: theory vs Monte Carlo on cached neutral reps (no model)
	$(PY) scripts/validate_theory_neutral.py --model $(MODEL)

neutral:  ## T-021: neutral representations (MODEL RUN)
	$(PY) scripts/run_neutral.py --model $(MODEL)
	$(PY) scripts/neutral_sanity.py --model $(MODEL)

exp1:  ## T-024: Exp 1 analysis (needs VAL, TEST behavior runs + neutral cache)
	$(PY) scripts/run_exp1_analysis.py --model $(MODEL) --behavior $(VAL) $(TEST) \
	    --select-split val --eval-split test

exp2-h2:  ## T-032: H2 on an Exp 2 run with cached activations (RUN=...)
	$(PY) scripts/run_exp2_h2.py --run $(RUN)

exp2-h3:  ## T-033: H3 (VAL=..., TEST=...)
	$(PY) scripts/run_exp2_h3.py --val-run $(VAL) --test-run $(TEST)

exp2-robustness:  ## T-034: robustness table (VAL=..., TEST=...)
	$(PY) scripts/run_exp2_robustness.py --val-run $(VAL) --test-run $(TEST)

exp3-readout:  ## T-041: H4 J-lens + logit-lens readout (RUN=...; MODEL RUN)
	$(PY) scripts/run_exp3_readout.py --run $(RUN)
	$(PY) scripts/run_exp3_readout.py --run $(RUN) --logit-lens

exp3-jspace:  ## T-042/T-043: H5 (VAL=..., TEST=...; MODEL RUN)
	$(PY) scripts/run_exp3_jspace.py --val-run $(VAL) --test-run $(TEST)
