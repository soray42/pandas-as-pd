# alias-inertia reproducibility targets.
#   make env      install pinned dependencies
#   make smoke    ONE pair + ONE small model on CPU (~minutes): verify the pipeline end-to-end
#   make stimuli  preview the generated stimuli + depth accuracy for one pair
#   make run      full scoring run (slow; GPU + CPU; several hours)
#   make analyze  figures + measured-frequency dose + raw prior-pull + candidate table (cheap)
#   make corpus   measure canonical alias frequencies from a public Python corpus (needs network)
#   make deepseek DeepSeek API probe: frontier model + 128k context + thinking vs non-thinking
#   make test     unit tests
# Override the interpreter or config:  make analyze PY=python3  CONFIG=configs/full.yaml

PY ?= python
CONFIG ?= configs/full.yaml
SMOKE ?= configs/smoke.yaml

.PHONY: env smoke stimuli run analyze corpus deepseek deepseek-analyze test clean

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
	$(PY) scripts/dose_measured.py --dose log_conv_count
	$(PY) scripts/raw_prior_pull.py --config $(CONFIG)
	$(PY) scripts/candidate_table.py

# Measure canonical alias frequencies from a public Python corpus (needs network; streams,
# downloads nothing large). Produces results/corpus_freq.json, the dose variable for `analyze`.
corpus:
	$(PY) scripts/corpus_freq.py --n-files 150000

# Needs DEEPSEEK_API_KEY in the environment (never committed). A few USD; responses cache under
# .cache/deepseek so re-runs are free. Core sweep + salience intervention + nonce-alias robustness.
deepseek:
	$(PY) scripts/run_deepseek.py
	$(PY) scripts/run_deepseek.py --extended --tasks forced_choice --modes nothink --conditions conventional,swapped,no_prior --depths 0 --deep-depths "" --n-runs 8 --temperature 0.7 --nonce-aliases zz,qx,vv --out results/deepseek_ext_nrun.jsonl
	$(PY) scripts/run_deepseek.py --extended --salience --tasks forced_choice --modes nothink --conditions conventional,swapped --depths 0 --deep-depths "" --n-runs 8 --temperature 0.7 --out results/deepseek_ext_nrun_salience.jsonl
	$(PY) scripts/analyze_deepseek.py
	$(PY) scripts/analyze_deepseek_ext.py

deepseek-analyze:
	$(PY) scripts/analyze_deepseek.py
	$(PY) scripts/analyze_deepseek_ext.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf results/cache results/cache_smoke results/smoke* results/preview_stimuli.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} +
