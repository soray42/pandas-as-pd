# Released outputs

These files are committed so the analysis and figures reproduce with `make analyze`, without the
multi-hour scoring run.

| File | Contents |
|---|---|
| `full.parquet` | one row per (model x pair x condition x depth x template x rep): prior-pull inputs, per-continuation log-probs, metadata |
| `full_generations.jsonl` | greedy generations with their classification and broken-call check |
| `full_stimuli_meta.jsonl` | per-stimulus metadata and prompt hashes (prompts are regenerable) |
| `full_manifest.json` | config, config hash, per-model fingerprints, environment, timings |
| `full_analysis.json` | per-model gaps, dose tiers, distance, generation rates, verdict |
| `dose_regression.json` | dose slope (mixed model + pair-clustered bootstrap) |
| `raw_prior_pull.json` | per-condition raw prior-pull (the no-prior baseline) |

Not committed (regenerable, gitignored): the score cache (`cache/`), smoke outputs (`smoke*`,
`cache_smoke/`), and run logs. Re-create everything with `make run` then `make analyze`.
