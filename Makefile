# alias-inertia reproducibility targets.
#   make env      install pinned dependencies
#   make smoke    ONE pair + ONE small model on CPU (~minutes): verify the pipeline end-to-end
#   make stimuli  preview the generated stimuli + depth accuracy for one pair
#   make run      full scoring run (slow; GPU + CPU; several hours)
#   make analyze  figures + dose regression + raw prior-pull + verdict (cheap; reads results/)
#   make test     unit tests
# Override the interpreter or config:  make analyze PY=python3  CONFIG=configs/full.yaml

PY ?= python
CONFIG ?= configs/full.yaml
SMOKE ?= configs/smoke.yaml

.PHONY: env smoke stimuli run analyze test clean

env:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) scripts/run.py --config $(SMOKE)
	$(PY) scripts/analyze.py --config $(SMOKE)

stimuli:
	$(PY) scripts/gen_stimuli.py --config $(CONFIG) --show 3

run:
	$(PY) scripts/run.py --config $(CONFIG)

analyze:
	$(PY) scripts/analyze.py --config $(CONFIG)
	$(PY) scripts/dose_regression.py --config $(CONFIG)
	$(PY) scripts/raw_prior_pull.py --config $(CONFIG)

test:
	$(PY) -m pytest -q

clean:
	rm -rf results/cache results/cache_smoke results/smoke* results/preview_stimuli.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} +
