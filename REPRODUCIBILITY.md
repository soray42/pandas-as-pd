# Reproducibility

This maps the released artifacts to the ACL Responsible NLP / reproducibility checklist items.
Full machine-readable provenance is in `results/full_manifest.json` (config, config hash, per-model
fingerprints, environment, timings, cache stats).

## Compute

- GPU: one NVIDIA RTX 5070 Laptop GPU (8 GB), used in fp16 for the depth-`<=2048` bins of the
  0.5B and 1.5B models.
- CPU: 32 logical cores; llama.cpp pinned to `n_threads=8`, `n_batch=512` for determinism. Used
  for the long-distance (8192-token) bins and the 3B/7B/8B models in 4-bit GGUF.
- Host: Windows 11, Python 3.13.3, PyTorch 2.12.0.dev (CUDA 12.8 build).
- Total scoring runtime: about 6.1 hours (369 minutes) for the full run; the analysis
  (`make analyze`) takes under a minute and reads the released `results/`, so figures and
  statistics reproduce without re-scoring.

### fp16 / GGUF split

The local PyTorch build has no flash or memory-efficient SDPA kernel for this GPU (Blackwell
sm_120), so HF attention is O(L^2) in memory and is capped at depth 2048 on GPU. The long-distance
and larger-model bins therefore run as 4-bit GGUF on CPU via llama.cpp, which uses O(L) attention.
Each model uses a single backend across all its depths (no mixing within a model). This split is
recorded per model in the manifest and in `configs/full.yaml`.

## Models

GPU models (HuggingFace, fp16) with pinned commit hashes:

| Checkpoint | HF repo ID | Revision (commit) |
|---|---|---|
| Qwen2.5-0.5B (base) | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

CPU models (4-bit GGUF pulled with Ollama) with content sha256 (the `general.file_type` GGUF
quantization code is shown; 15 = Q4_K_M, 7 = Q8_0):

| Ollama tag | Quant | GGUF sha256 |
|---|---|---|
| `qwen2.5:0.5b` | Q4_K_M (15) | `c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515` |
| `llama3.2:1b` | Q8_0 (7) | `74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` |
| `llama3.2:3b` | Q4_K_M (15) | `dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff` |
| `qwen2.5-coder:7b` | Q4_K_M (15) | `60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463` |
| `llama3.1:8b` | Q4_K_M (15) | `667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29` |

The run is nine configurations over eight distinct checkpoints (Qwen2.5-0.5B-instruct is measured
in both fp16 and 4-bit GGUF). The backend reads the GGUF blob from the local Ollama store and
records its sha256 as the model fingerprint.

An OpenAI-compatible API backend (e.g. DeepSeek) was prototyped but is not part of this study:
chat-completions endpoints cannot teacher-force-score an arbitrary supplied continuation, which the
prior-pull metric requires. All released numbers use the HF and llama.cpp backends only.

## Scale of the evaluation

- 1,683 scored prefixes (one row per model x pair x condition x depth x template x rep), in
  `results/full.parquet`.
- 468 greedy generations (`results/full_generations.jsonl`), classified and checked for broken
  calls.
- 1,122 swapped and no-prior rows enter the dose regression.
- 0 stimuli skipped (no context-window overflows at the released depths).

## Statistics

- Effect sizes are log-probability differences (nats). The headline is the difference-in-differences
  gap, prior-pull(swapped) minus prior-pull(no-prior).
- Confidence intervals are 95% pair-clustered bootstrap (resample the 6 swap pairs, then rows within
  each chosen pair): 5,000 iterations for the gap and per-condition means, `bootstrap_seed=20260604`.
- Dose slope: a regression of prior-pull on `is_swapped * tier` with `C(depth)` and crossed random
  intercepts for pair and model, fit with statsmodels (`statsmodels==0.14.6`). The
  `is_swapped:tier` interaction is the gap's dose slope. The primary CI is a 2,000-iteration
  pair-clustered bootstrap of the OLS slope; the mixed-model Wald CI is reported as a secondary,
  anticonservative reference (only 6 pair-clusters). See `scripts/dose_regression.py`.
- Determinism: `seed=12345` pins Python/NumPy/PyTorch RNGs; llama.cpp scoring is bit-identical
  run-to-run with the pinned `n_threads`/`n_batch`/`seed`. The disk cache is content-addressed on
  the backend's score-affecting settings plus a hash of the scoring code, so a configuration or code
  change invalidates it automatically.

## Data

Stimuli are generated programmatically by `src/alias_inertia/stimuli.py` from the configs; they are
not sampled from any corpus. There is therefore no dataset to license and no train/test
contamination to control. The generator plus the configs reproduce every prompt deterministically;
`results/full_stimuli_meta.jsonl` records each stimulus's metadata and prompt hash.

## Licensing

- Code in this repository: MIT (see `LICENSE`).
- Model weights are not redistributed here and carry their own licenses. The Qwen2.5 and
  Qwen2.5-Coder checkpoints are released under the licenses on their model cards (Apache-2.0 for the
  sizes used here); Llama-3.1 and Llama-3.2 are under the Llama Community License. Users must obtain
  the weights from the original sources and comply with those terms.
