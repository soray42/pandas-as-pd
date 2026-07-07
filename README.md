# alias-inertia

A controlled probe of code language models. When a canonical import alias (`np`, `pd`,
`plt`, ...) is rebound to a different library, for example `import pandas as np`, does the model
track the local binding or revert to the library the alias conventionally denotes? We measure a
**prior-pull** score by continuation scoring and contrast a swapped binding against a no-prior
control, isolating prior reassertion from generic long-context degradation, and then locate the
effect mechanistically on two open models. The accompanying paper (8 pages plus appendix) is in
`paper/`.

## Headline result

Across nine model configurations from 0.5B to 8B (base and instruct), the model places more
probability on the alias's conventional methods than on the bound library's. The difference-in-
differences gap (swapped minus no-prior prior-pull) is **+6.81 nats** (95% CI [+4.56, +9.11]), with
every model's interval excluding zero. The gap scales with how common the convention is, is flat
from 0 to 8192 tokens of distance, and surfaces in generated code (74% of swapped completions access
an attribute that does not exist on the bound library).

![Dose response: the gap grows with the measured corpus frequency of the alias convention](figures/full_dose_curve.png)

| Model | Params | Tune | DiD gap [95% CI] |
|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | base | +6.70 [+4.15, +9.05] |
| Qwen2.5-0.5B | 0.5B | inst | +6.93 [+4.33, +9.19] |
| Qwen2.5-0.5B (Q4 GGUF) | 0.5B | inst | +5.51 [+2.41, +9.66] |
| Llama-3.2-1B (Q4 GGUF) | 1B | inst | +7.08 [+2.99, +10.00] |
| Qwen2.5-1.5B | 1.5B | base | +7.46 [+4.64, +10.04] |
| Qwen2.5-1.5B | 1.5B | inst | +7.56 [+4.78, +10.27] |
| Llama-3.2-3B (Q4 GGUF) | 3B | inst | +5.17 [+2.65, +8.72] |
| Qwen2.5-Coder-7B (Q4 GGUF) | 7B | inst | +6.33 [+3.31, +9.63] |
| Llama-3.1-8B (Q4 GGUF) | 8B | inst | +7.33 [+3.73, +10.28] |
| **Overall** | | | **+6.81 [+4.56, +9.11]** |

**Dose slope (measured corpus frequency):** regressing prior-pull on the swapped indicator
interacted with the log corpus frequency of each canonical `import LIB as alias` convention
(measured over 150k Python files, `scripts/corpus_freq.py`), the gap rises by **+0.94 nats per
natural-log unit of corpus frequency**; pair-clustered bootstrap 95% CI [+0.69, +1.35], 99% of
bootstrap slopes positive (`results/dose_measured.json`). Numbers are in
`results/full_analysis.json`, `results/dose_measured.json`, `results/raw_prior_pull.json`, and
`results/corpus_freq.json`; provenance in `results/full_manifest.json`.

## Frontier model and 128k context

A hosted frontier model, DeepSeek-V4-Pro (one-million-token context), cannot be teacher-forced, so
we probe it behaviorally: a two-alternative forced choice at the use site, free generation, and a
verbal "which library is this alias bound to?" question, contrasting swapped against no-prior in
both the direct and thinking modes, out to 128k tokens. The direct model picks the conventional
library's non-existent method on **65%** of swapped items against **36%** in no-prior (difference
**+0.30**, 95% CI [+0.06, +0.61]) yet names the bound library correctly **100%** of the time: it
recognizes the binding but does not act on it. The pull is flat from 0 to 128k tokens, and the
model's thinking mode drives the forced-choice rate to **0**. The same split shows up in realistic
generation: on naturalistic, task-demanding snippets across all twelve alias conventions, the direct
model's swapped broken-call rate is about three times the no-prior control (**9.2%** vs **3.1%**,
difference **+6.2 points**, 95% CI [+0.0, +12.3]) and the thinking mode removes it (**0.8%**), though
the frontier model breaks far less often than the small open checkpoints
(`results/deepseek_naturalistic_all_results.json`). Numbers in
`results/deepseek_analysis.json`; reproduce with `make deepseek` (needs `DEEPSEEK_API_KEY` in your
environment, never committed; API responses cache under `.cache/` so re-runs are free).

## Where the prior acts (mechanistic arm)

On Qwen2.5-0.5B and 1.5B (base, fp16, TransformerLens) with a token-aligned stimulus set on
numpy/pandas, a validated proxy metric (Spearman **0.87** against the full prior-pull on the same
prompts, gate in `mech/results/m0_validation.json`), and four experiments:

- **Logit lens**: the swapped trajectory tracks the conventional one layer by layer; the bound
  library's evidence surfaces mid-stack but attenuated (peak **-0.58** vs **-8.31** nats against
  no-prior on 0.5B, **-6.70** vs **-13.86** on 1.5B) and the last 4-5 blocks write the conventional
  library into the output in 100% of swapped items.
- **Activation patching** (24,960 patches): the swapped-vs-no-prior difference rides the use-site
  alias token from the embedding upward (fraction restored **0.6-1.2** at layer 0) and hands off to
  the final position exactly in the late window the lens identifies; the import line is close to
  inert and the filler is a clean null.
- **Head attribution**: the heads that fetch the binding from the import under a nonce alias lose
  that attention sharply under the canonical one; the strongest (1.5B L22H8) flips from
  binding-promoting to the model's strongest positive-attribution head.
- **Ablation (negative result)**: zero-ablating the top positive-attribution heads does not reduce
  the pull (it slightly raises it on 1.5B, in both conditions), so the late write is distributed;
  the working causal handle is the use-site alias representation. Text-level mitigations agree: a
  binding-restating comment trims only 15-25% of the gap and an obey-the-imports instruction is
  near-ineffective (`mech/results/d5_summary.json`).

Everything lives under `mech/` (results, figures, manifest) with per-experiment verdicts,
engineering notes, and the verification process in `MECH_RESULTS.md`. Reproduce with the
`scripts/mech_*.py` runners after `pip install -r mech/requirements-mech.txt` (gates first:
`mech_m0_validate.py`, then `mech_m1_logitlens.py --pilot`; one model per process; 8 GB GPU
suffices).

## Reproduce

```bash
# Anonymous review copy: download and unzip https://anonymous.4open.science/r/pandas-as-pd , then:
cd pandas-as-pd
make env        # install pinned deps (install torch for your platform from pytorch.org first)
make smoke      # ~2-4 min on CPU: verify the whole pipeline on 1 pair + 1 small model
make analyze    # ~1 min: regenerate the figures, statistics, and verdict from released results/
make run        # OPTIONAL, ~6 h: re-score everything from scratch
make deepseek   # OPTIONAL: frontier-model + 128k probe via DeepSeek API (needs DEEPSEEK_API_KEY)
```

`make analyze` reproduces every figure and statistic from the committed `results/`, so reviewers do
not need the slow scoring run. `make smoke` runs `generate -> score -> analyze` on one pair with
Qwen2.5-0.5B (downloaded on first use, about 1 GB) and prints the sanity checks: a positive swapped
prior-pull on numpy/pandas and the generation and broken-call rates.

### Runtime and the hardware split per stage

| Stage | Time | Hardware |
|---|---|---|
| `make env` | 1-2 min | any |
| `make smoke` | 2-4 min (+ 1 GB download) | CPU (set `device: cuda` in `configs/smoke.yaml` for GPU) |
| `make analyze` | < 1 min | CPU |
| `make run` | about 6 h | GPU fp16 for depths `<=2048` (0.5B, 1.5B); 4-bit GGUF on CPU via Ollama for the 8192-token bins and the 3B/7B/8B models |

`make run` needs Ollama for the CPU path (`ollama pull qwen2.5:0.5b llama3.2:1b llama3.2:3b
qwen2.5-coder:7b llama3.1:8b`); the backend reads each GGUF blob directly. The fp16/GGUF split is
forced by the local GPU lacking a flash/memory-efficient attention kernel; see `REPRODUCIBILITY.md`.

## Layout

```
src/alias_inertia/   package: lexicons, stimuli, scoring, metrics, generation, validity,
                     determinism, deepseek_probe (API behavioral arm), backends/ (base, hf, llamacpp),
                     mech/ (stimuli alignment, proxy metric, logit lens, patching, heads, ablation)
scripts/             run.py, analyze.py, dose_regression.py, dose_measured.py, raw_prior_pull.py,
                     corpus_freq.py, candidate_table.py, gen_stimuli.py, smoke_backend.py,
                     run_deepseek.py, analyze_deepseek.py, analyze_deepseek_ext.py,
                     run_naturalistic.py, run_naturalistic_all.py, run_deepseek_naturalistic.py,
                     run_deepseek_naturalistic_all.py, mech_m0_validate.py ... mech_m4_figure.py,
                     mech_d5_behavioral.py (mechanistic runners)
configs/             full.yaml (produced the results), smoke.yaml (fast pipeline check)
results/             scored prefixes (full.parquet), generations, manifest, analysis JSON;
                     deepseek_raw*.jsonl + deepseek_analysis.json (API probe)
mech/                mechanistic arm: results/ (records, summaries, manifest), figures/,
                     requirements-mech.txt; verdicts and verification in MECH_RESULTS.md
figures/             the paper figures (behavioral + mechanistic)
tests/               13 test modules: scoring math, stimuli, metrics, cache, lexicons,
                     generation, validity, deepseek probe, mech alignment/proxy/lens/patching
paper/               the paper (acl_latex.tex, custom.bib)
```

Run `make test` (or `python -m pytest -q`) for the test suite. The scoring math is checked against
an independent autoregressive reference on a tiny model that downloads on first run.

## Method in brief

Three conditions for a library pair `(A, B)` with canonical alias `a`: conventional (`import A as a`),
swapped (`import B as a`, so `A` is the prior and `B` is the binding), and no-prior (`import B as zz`,
a non-canonical alias bound to the same library). At the prefix ending `alias.` we score
discriminative continuation strings per library and take prior-pull = logsumexp over the conventional
library's continuations minus logsumexp over the bound library's. Continuation scoring (summed
teacher-forced log-prob of the whole multi-token method string) is used because method names are
multi-token. The headline gap is prior-pull(swapped) minus prior-pull(no-prior). The behavioral
claims stand on their own; the mechanistic arm (above) localizes the effect on two models and is
scoped accordingly (one pair, a validated proxy metric, no circuit-level claims).

## Anonymized release

The anonymous review copy is at https://anonymous.4open.science/r/pandas-as-pd, which hides the
repository owner. The committed files contain no author names, emails, institutions, or tracking
links. A de-anonymized public repository is for the camera-ready version only.

## License

Code is MIT (`LICENSE`). Model weights are not redistributed and carry their own licenses
(Qwen2.5/Qwen2.5-Coder per their model cards; Llama-3.1/3.2 under the Llama Community License). See
`REPRODUCIBILITY.md`.
