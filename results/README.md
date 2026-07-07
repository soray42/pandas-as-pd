# Released outputs

These files are committed so the analysis and figures reproduce with `make analyze`, without the
multi-hour scoring run.

| File | Contents |
|---|---|
| `full.parquet` | one row per (model x pair x condition x depth x template x rep): prior-pull inputs, per-continuation log-probs, metadata |
| `full_generations.jsonl` | greedy generations with their classification and broken-call check |
| `full_stimuli_meta.jsonl` | per-stimulus metadata and prompt hashes (prompts are regenerable) |
| `full_manifest.json` | config, config hash, per-model fingerprints, environment, timings |
| `full_analysis.json` | per-model gaps, distance, generation rates, verdict |
| `raw_prior_pull.json` | per-condition raw prior-pull (the no-prior baseline) |
| `corpus_freq.json` | measured `import LIB as alias` frequencies from a 150k-file Python corpus |
| `dose_measured.json` | dose slope: DiD gap vs measured log corpus frequency (mixed model + pair-clustered bootstrap) |
| `candidate_table.json` | per-continuation library, attribute, existence, token length, pinned version |
| `deepseek_raw*.jsonl` | DeepSeek-V4-Pro probe records (forced choice, generation, verbal; direct and thinking) |
| `deepseek_analysis.json` | DeepSeek forced-choice / generation / verbal rates and depth curve to 128k |
| `deepseek_ext_nrun*.jsonl` | extended 12-alias forced choice, temperature-sampled for a graded P(prior), plus salience |
| `deepseek_ext_analysis.json` | DeepSeek frontier dose-response and salience intervention |
| `deepseek_manifest.json` | DeepSeek probe provenance (model and base URL; no API key) |
| `naturalistic_scenarios.json` | model-generated pandas-demanding code contexts (validated, with ids); the naturalistic arm's stimuli |
| `naturalistic_{records.jsonl,results.json}` | local naturalistic arm, numpy/pandas case (coder-7b, 0.5b): prior-pull by condition, broken-call and wrote-correct-op rates, hashed prompts |
| `naturalistic_scenarios_more.json` | model-generated numpy / sklearn / xgboost task contexts (the other bound libraries) |
| `naturalistic_all_{records.jsonl,results.json}` | naturalistic arm across all 12 alias conventions: per-pair prior-pull DiD, swapped/no-prior/correct, broken-call rates |
| `deepseek_naturalistic_{records.jsonl,results.json}` | DeepSeek-V4-Pro on the naturalistic arm, numpy/pandas case (direct vs thinking): broken-call and wrote-correct-op rates |
| `deepseek_naturalistic_all_{records.jsonl,results.json}` | DeepSeek-V4-Pro on the naturalistic arm across all twelve alias conventions (direct vs thinking): per-pair swapped broken-call and wrote-target rates |

`dose_regression.json` is the legacy ordinal-tier version of the dose analysis, superseded by the
measured-frequency `dose_measured.json`.

Not committed (regenerable, gitignored): the score cache (`cache/`), the API response cache
(`.cache/`), smoke outputs, and run logs. Re-create with `make run`, `make corpus`, `make deepseek`,
then `make analyze`.
