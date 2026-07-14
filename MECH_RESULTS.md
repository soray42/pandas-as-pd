# PHASE M: preliminary mechanistic localization (alias-inertia)

Status: complete through M0-M4 + D5. All behavioral results, `results/`, and the paper are
untouched; every Phase M artifact lives under `src/alias_inertia/mech/`, `scripts/mech_*.py`,
`tests/test_mech_*.py`, `mech/results/`, and `mech/figures/`. Nothing here is integrated into
the paper yet; this document is the review input for that decision.

Scope discipline: two models (Qwen2.5-0.5B and Qwen2.5-1.5B, base, fp16, TransformerLens),
one primary pair (numpy<->pandas), depths {0, 512}, four experiments plus two behavioral arms.
No circuit discovery, no SAEs, no path patching. Every quantitative claim below was
independently re-derived from the raw record files in a separate verification pass; mismatches
found in drafting were corrected before this document was written (see Verification).

Scope note against the project plan: `alias-inertia_scope.md` section 8 reserves the mechanistic
layer as the main-conference extension delta. Publishing this preliminary localization in the
workshop paper spends part of that delta; the remaining reserved deltas are the real-code arm,
full circuit work (path patching, SAEs, cross-family), and frontier-scale mechanism.

## The question

The behavioral result (frozen): swapped bindings (`import pandas as np`) show prior_pull of
about +7 nats net of the no-prior control, immediately and flat with distance. The correct
continuation never appears in context, so tracking requires a two-hop composition
(np -> pandas -> pandas methods) while the failure mode is a one-hop token association
(np -> numpy methods). Central question: does the bound-library answer (a) never form across
layers, or (b) form mid-stack and get overwritten late?

## Answer in one paragraph

Both, in proportions that shift with scale, and the flip is universal: in swapped items the
bound-library evidence does surface mid-stack in 90-100% of items but is heavily attenuated
relative to no-prior (0.5B: peak bound evidence -0.58 vs -8.31 nats, 93% attenuated; 1.5B:
-6.70 vs -13.86, 52% attenuated), and the last four to five blocks then write the alias's
conventional library into the output in 100% of items. Causally (activation patching), the
swapped-vs-no-prior difference rides the use-site alias token from the embedding upward and is
written into the final position exactly in that late window; the import line is causally inert
for this difference and the filler is a clean null. At the head level, the binding-fetch heads
that attend to the import line under a nonce alias lose that attention 3-6x under the canonical
alias, and the strongest of them flips from binding-promoting to the model's strongest
positive-DLA head inside the late window. Ablating positive-DLA heads does not reduce the pull
(it slightly increases it on 1.5B), so the late-layer write is distributed rather than
concentrated in a few heads; the causal handle that does work is the use-site alias
representation itself. Text-level mitigations behave accordingly: a restatement comment trims
only 15-25% of the gap and an obey-the-imports instruction is ineffective at these scales.

## M0. Setup, alignment, proxy validation (gate: PASSED)

- Environment: transformer_lens 3.5.1 with the frozen behavioral stack unchanged
  (transformers 5.5.0, torch 2.12 nightly cu128); pins recorded in `mech/requirements-mech.txt`.
  Both models load with `from_pretrained_no_processing` (fp16, CUDA): the weight-processing
  pass OOMs the reference machine's RAM for 1.5B, and unprocessed weights match the HF forward
  exactly. Verified: the logit-lens final checkpoint reproduces the true final logits
  (argmax-exact, corr 1.0).
- Token alignment: for numpy<->pandas all three conditions tokenize to identical lengths with
  identical spans ('numpy'/'pandas' one token each; np/zz single tokens at both occurrences).
  `assert_triple_aligned` enforces per-triple equality of lengths, spans, and token_ids outside
  the library and alias positions; tests cover the misalignment paths. Stimuli: 30 bases per
  depth x 3 conditions = 180; depth-0 variety comes from a 30-name use-line variable pool
  (`{var} = {alias}.`), depth 512 adds per-rep filler. Template fixed to t1; nonce alias zz.
- Proxy metric: first tokens of the six discriminative continuations per library (no leading
  space), zero cross-set collisions; proxy_pull = LSE(numpy ids) - LSE(pandas ids) at the final
  position, invariant to the log-softmax shift.
- Proxy-vs-full validation (proxy at the TL final layer vs continuation-scored prior_pull via
  the frozen HF backend, n=180 per model), per-item records in `mech/results/m0_records.jsonl`:
  - 0.5B: Pearson 0.983, Spearman 0.867, sign agreement 0.981 (n_strong=160/180). PASS.
  - 1.5B: Pearson 0.939, Spearman 0.868, sign agreement 0.928 (n_strong=180/180). PASS.
  Sign agreement is computed on items with |full| > 1 nat. The gate run was executed twice
  end to end; all statistics reproduce exactly (deterministic pipeline).
- TL-vs-HF numerics, scoped precisely: 0.5B argmax 6/6 with mean max |logprob diff| 0.053 nats.
  1.5B argmax 4/6 with 2.65 nats mean max diff; the disagreements occur only at depth 512 and
  only among near-tied top candidates (same candidate set, gaps under 1 nat), an fp16
  eager-attention vs SDPA accumulation drift. TL is therefore NOT a faithful absolute-logit
  backend for 1.5B at 512-token contexts; what is validated is what the experiments use:
  within-TL condition contrasts and the proxy-to-behavioral ranking (the gate above).

## M1. Logit-lens trajectories (the adjudicating experiment)

Records: `mech/results/m1_records.jsonl` (360 trajectories), summary `m1_summary.json`,
figures `m1_logitlens_*.png` (checkpoint 0 omitted from plots: the embedding-only stream
carries a ~+40-nat lexical unembed artifact).

- Pilot gate (0.5B depth 0): conventional final +3.78 > 0, no_prior final -6.51 < 0. PASS.
- Swapped hugs conventional, not no-prior, at every layer, both models, both depths: mean
  |swapped - conventional| over blocks >= 2 is 0.25-0.55 nats while |swapped - no_prior| is
  3.8-4.6 nats (swapped sits 6-14x closer to conventional). The alias token identity, not the
  actual binding, determines the trajectory shape.
- Bound-side evidence in swapped: present somewhere after block 4 in 90-100% of items
  (the 90% cell is 0.5B depth 512), mean last-bound-lead layer 19.3-19.6 of 24 (0.5B) and
  23.0 of 28 (1.5B), but heavily attenuated vs no-prior:
  - 0.5B: mean per-item min over blocks >= 5 is -0.58 swapped vs -8.31 no_prior (93% attenuated)
  - 1.5B: -6.70 swapped vs -13.86 no_prior (52% attenuated)
- Late-layer resolution: in 1.5B the discrimination concentrates at blocks 22-24, where
  no_prior plunges to -14.9 (binding retrieval surge, proving the two-hop path works without a
  competing prior) while conventional AND swapped jump to +4..+6. In 0.5B the divergence is
  milder and earlier (no_prior trough -8..-9 around blocks 18-21; swapped recovers to +2.5).
- 100% of swapped items end prior-positive, both models, both depths.

Verdict: "formed-then-overwritten" describes 1.5B well; 0.5B is closer to
"barely-forms-and-is-overwritten" (its bound-side lead is marginal: mean min -0.58, and 3 of 30
depth-512 items never go negative). The two models should not be symmetrized.

## M2. Activation patching (causal localization)

Records: `mech/results/m2_*_records.jsonl` (24,960 rows: 2 models x 2 depths x 30 base pairs x
2 directions x layers x 4 position groups), CI companions `m2_ci_*.json`, heatmaps
`m2_patch_*.png`. Swapped and no_prior are paired by base (identical tokens except the alias
identity), resid_post patched per (layer x group);
fraction_restored = (pull_dst - pull_patched) / (pull_dst - pull_src).

- use_alias (the alias token in the use line): patching the earliest layers transfers most of
  the difference. At L0, noprior->swapped 0.79 (0.5B) / 0.61 (1.5B); swapped->noprior ~1.0
  (0.5B) and ~1.2 (1.5B; overshoot at L0, 1.21 at d0 / 1.19 at d512). The effect decays and
  crosses zero at L20 (0.5B) / L22 (1.5B): the alias-identity information has left the alias
  position by then. The L0 resid is essentially the token embedding, so early rows amount to
  lexical substitution; the informative content is the decay profile.
- final_pos: the complementary ramp. 0.5B climbs gradually (0.04 at L0, 0.56 at L8, 0.81 at
  L14, ~1.0 from L19). 1.5B stays near 0 through L14, then 0.23 (L15-18), 0.55 (L20),
  0.71 (L21), 0.97 (L22), ~1.0 from L23. The final-position write window (L20-23) matches the
  M1 bifurcation (blocks 22-24) layer for layer; correlational and causal timing agree.
- import_span: near zero at the use site. At depth 0 there are small L0-L1 excursions in the
  swapped->noprior direction (0.25-0.41; the import-line alias token embedding carries a small
  direct contribution); at depth 512 the residual peaks move to mid layers and shrink
  (<= 0.10 at L5 for 0.5B, <= 0.21 at L7 for 1.5B). Within a base pair the import lines differ
  only in the alias token, so this shows the alias-identity difference does not flow through
  the import line to the use site.
- filler_span: |mean fraction| <= 0.01 in the noprior->swapped direction and <= 0.022 in
  swapped->noprior at depth 512 (peak 1.5B L6); exactly 0 at depth 0 (empty group). Clean null.

Verdict: the competing prior does not act by degrading retrieval from the import line; it rides
the use-site alias token from the embedding upward and is written into the final position in
the late window M1 identified. Replacing the use-site alias representation in the first few
layers suffices to make the swapped model track the binding (fraction restored 0.6-1.2); no
import-line intervention does anything. This localizes the swapped-vs-no-prior difference; it
does not decompose the no-prior binding retrieval itself.

## M3. Import-line attention and head attribution

Records: `mech/results/m3_records.jsonl` (120,960 rows), summary + `m3_top_heads.md`, figures
`m3_heads_*.png`. Attention = per-head mass from the final position onto the import span
(pattern row sliced in-hook). DLA = per-head final-position output, computed manually as
z[final] @ W_O per head then passed through the frozen final RMSNorm (divide by cached scale,
multiply by ln_final.w; the basis verified to sum to the true logits), projected onto
d = mean(W_U[:, numpy ids]) - mean(W_U[:, pandas ids]). DLA is a linearization of the proxy
(logsumexp is nonlinear) and a direct-contribution measure, not a net causal effect (see M4).

Honesty notes on the attention metric:
- The import line occupies positions 0-3, which is also the start-of-sequence attention sink in
  Qwen models: hundreds of heads put 0.6-1.0 mass there with DLA ~ 0 and write nothing. Raw
  import-attention mass is therefore not interpretable on its own.
- Pooled over import-attending heads, attention does NOT collapse in swapped (it is slightly
  higher). The meaningful statistic is the per-head condition contrast (identical token
  positions across conditions, so a contrast cannot be a sink artifact), and the collapse is
  specific to the small set of heads that also carry binding DLA.

Findings (consistent at both depths):
- Heads selected by the largest import-attention contrast that also carry DLA show one
  signature: import attention collapses in swapped AND the binding-promoting output weakens or
  flips.
  - 0.5B L17H1: attn 0.79 -> 0.26 (d0), 0.61 -> 0.09 (d512); DLA -0.27/-0.24 -> -0.17.
  - 1.5B L21H5: attn 0.88 -> 0.42 (d0), 0.86 -> 0.29 (d512); DLA -0.16/-0.15 -> +0.03/+0.02.
    Selection note: this head is picked by the attention-contrast ranking, not by swapped-DLA
    (it ranks ~91/336 there, precisely because the hijack zeroes its swapped-side output);
    its relevance is its no-prior binding role.
  - 1.5B L22H8: attn 0.70 -> 0.23 (d0), 0.52 -> 0.11 (d512); DLA -0.06 -> +0.21, making it the
    strongest positive-DLA head in the model, inside the L20-23 write window.
- Reverse-contrast heads (more import attention in swapped: the 0.5B L4 cluster; 1.5B L8H7 at
  0.04 -> 0.90 at d512) have DLA ~ 0: same-token matching between the two alias occurrences,
  descriptive only. Per-model caveat: 1.5B L4H0 does carry +0.07 swapped DLA, so the
  "L4 cluster is inert" statement is 0.5B-specific.
- Remaining positive-DLA mass is diffuse: a late L26-27 cluster on 1.5B (+0.07..+0.12 each;
  note L26H2/L26H4 attend the import similarly in both conditions, i.e. always-on import
  readers, not prior-discriminating) and weak mid-layer heads on 0.5B (~+0.11 each).
- Noted anomaly: 0.5B L8H7 carries binding-side DLA (-0.10) with near-zero import attention,
  i.e. binding-consistent output sourced from somewhere other than the import line.

Verdict (memory-vs-context reading): the context (binding-fetch) heads exist and work under a
nonce alias; under the canonical alias the same heads lose their import-line attention and
their output stops carrying the binding, while positive-DLA output concentrates in the late
window. Combined with M2, the query side of those heads (the use-site alias token) is what
changes.

## M4. Top-k positive-DLA head ablation (negative result)

Records/summaries: `mech/results/m4_k{1,3,5}/`, figure `mech/figures/m4_ablation.png`.
Zero/scale (factors 0, 0.25, 0.5) the hook_z of the top-k positive-DLA heads (ranked on
swapped; k in {1,3,5}) at all positions; measure the proxy on swapped (effect) and conventional
(collateral). 60 items per condition per cell. 0.5B has only 4 positive-DLA heads in its top
list, so k=5 ablates 4.

- 0.5B (L15H2; +L16H10, L19H8; +L23H3): swapped +2.60 -> +2.52 / +2.60 / +2.52 at k=1/3/5
  (factor 0); conventional +3.48 -> +3.45 / +3.58 / +3.46. No swapped-specific effect at any k
  (deltas -0.08 to -0.01 against a ~0.9-nat condition gap; at k=3 conventional moves +0.10
  while swapped does not move). Not a confirmation of anything.
- 1.5B (L22H8; +L27H0, L27H1; +L26H2, L27H4): ablation INCREASES the pull: swapped
  +4.27 -> +4.74 / +4.66 / +4.68; conventional +4.89 -> +5.43 / +5.31 / +5.36. The shift is
  not selective: at k=1 the conventional shift is statistically LARGER than the swapped shift
  (item-bootstrap difference +0.07, 95% CI [+0.02, +0.13]); at k=3/5 the difference CI includes
  zero. Partial factors interpolate monotonically.

Verdict, and the required reconciliation with M3: positive DLA measures a head's direct
contribution, not its net effect under intervention; removing these heads slightly raises the
proxy pull (consistent with downstream self-repair / backup behavior reported for ablations).
The late-layer prior write is distributed, simple head ablation is not a mitigation, and no
claim that M4 "causally validates" the M3 ranking is supported. The causal handle that does
move behavior remains M2's use-site alias representation (fraction restored 0.6-1.2). Reported
as a negative result, no cherry-picking.

## D5. Restatement and instruction arms (behavioral, frozen metric)

Records: `mech/results/d5_records.jsonl` (728 rows), summary `d5_summary.json`, figure
`d5_arms.png`. Full continuation-scored prior_pull (frozen HF fp16 backend, fresh runs), arms
built by post-processing the swapped prompt: plain, restatement ('# note: {alias} is
{bound_lib} in this file' immediately before the use line), no_prior, instruction (two
obey-the-imports comment lines above the import; instruct models only). Pairs: numpy<->pandas
(very_common) and sklearn<->xgboost (rare). Models: Qwen2.5-0.5B/-Instruct/1.5B/-Instruct.
n=24 per cell at depth 512 (depth 0 has n=2 by the behavioral rep-collapse; auxiliary only).
Two pairs, so per-pair reporting, no cluster bootstrap. All numbers verified against raw
records (30/30 claims).

numpy__pandas, depth 512 (prior_pull; delta vs plain [95% CI]):

| model | plain | restatement | instruction | no_prior |
|---|---|---|---|---|
| 0.5B | +7.75 | +6.00 (-1.75 [-1.92, -1.57]) | n/a | -1.98 |
| 0.5B-Instruct | +8.61 | +7.08 (-1.53 [-1.74, -1.30]) | +8.79 (+0.18 [-0.03, +0.40]) | -1.53 |
| 1.5B | +9.12 | +6.49 (-2.63 [-2.80, -2.46]) | n/a | -1.41 |
| 1.5B-Instruct | +9.71 | +7.33 (-2.38 [-2.55, -2.20]) | +8.86 (-0.85 [-1.08, -0.62]) | -1.03 |

sklearn__xgboost (rare): plain is already negative (-1.2 to -3.3), i.e. the models track the
binding when the alias convention is rare, echoing the behavioral dose-response; intervention
deltas there are small and mixed in sign, consistent with weak instruction-following at these
scales; no conclusion drawn.

Verdict: on the strong convention a passive adjacent restatement reliably trims 1.5-2.6 nats of
a ~10-nat gap (15-25%, every CI excluding zero) but leaves the model far on the prior side; an
explicit instruction moves 0.5B-Instruct not at all and 1.5B-Instruct by under 1 nat. This
mirrors the frozen DeepSeek salience result and fits the mechanism: the pull rides the alias
token identity (M2), which neither text intervention changes.

## Figures

- `mech/figures/m0_proxy_vs_full_*.png` (proxy validation scatter, per model)
- `mech/figures/m1_logitlens_*.png` (trajectories, 3 conditions x 2 depths, per model)
- `mech/figures/m2_patch_{direction}_{model}_{depth}.png` (8 heatmaps, layer x group)
- `mech/figures/m3_heads_*.png` (top heads by |DLA| with import attention, per model)
- `mech/figures/m4_ablation.png` (ablation factor sweep, both models, effect + collateral)
- `mech/figures/d5_arms.png` (behavioral arms per model and pair)

## Caveats and scope limits (read before integrating anywhere)

1. Two sizes of ONE family (Qwen2.5 base) on one synthetic stimulus family and one pair
   (numpy<->pandas): every mechanism claim is scoped to "Qwen2.5-0.5B/1.5B on our stimuli",
   not "code LMs".
2. All M1-M4 quantities are proxy_pull (first-token lexicon logsumexp), validated against the
   full behavioral metric at Spearman ~0.87; conclusions are claims about the proxy.
3. TL absolute logits for 1.5B diverge from HF at 512-token contexts (near-tie argmax flips,
   2.65 nats mean max diff); only within-TL contrasts and proxy ranking are validated.
4. Depths {0, 512} only, and M2 shows the import-span profile changes location between them;
   no extrapolation to other depths.
5. Logit lens is correlational; M2 patching is causal at resid-position granularity; head-level
   path patching, learned interventions, and SAEs are out of scope (reserved for the
   main-conference extension).
6. "Positive-DLA heads" is a correlational label; M4 shows their ablation does not reduce the
   pull. DLA and net causal effect must not be conflated anywhere in the paper text.
7. D5's instruction arm is a lower bound from small instruct models, not a verdict on
   instruction-following at scale; depth-0 D5 cells are n=2 and not leaned on.
8. The naturalistic and DeepSeek arms are untouched by Phase M; no mech claim extends to them.

## Engineering notes (gotchas hit and handled; each caught by a validation gate)

1. torch.inference_mode() breaks TL's lazy stack computations; torch.no_grad() everywhere.
2. apply_ln paths need names_filter to include resid_pre, resid_post, and exactly
   'ln_final.hook_scale'.
3. from_pretrained weight processing OOMs 15 GB system RAM for 1.5B; both models use
   from_pretrained_no_processing, validated by the final-checkpoint identity (corr 1.0).
4. cache.stack_head_results materialises a [pos, heads, d_head, d_model] broadcast (~2.4 GB
   per layer for 1.5B at depth 512) and OOMs an 8 GB GPU; per-head contributions are computed
   manually at the final position only.
5. TL 3.5.1 stack_head_results(apply_ln=True) omits ln_final.w on no_processing models
   (mismatches true logits by ~14 nats; the with-w basis matches to 0.01). The m3 run was
   redone in the verified basis after this was caught.
6. Depth-0 stimuli need explicit variety (a 30-name use-line variable pool); the behavioral
   rep-collapse would otherwise produce one base.
7. Shell pipelines mask crash exit codes (python | grep); long runs launched unpiped with -u.
   Split per-model processes required merge-on-write in every runner, plus union-merged
   config/manifest model lists so provenance describes all data on disk (one clobber incident
   was caught and the affected 0.5B m1 shard re-run).
8. Ambient desktop VRAM pressure (8 GB card, WDDM) makes 1.5B runs intermittently fail with
   transient CUDA OOM; runs use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True plus one
   automatic retry.

## Reproduction

```bash
# environment (on top of the frozen behavioral env)
pip install -r mech/requirements-mech.txt
# tests (alignment, proxy, logit-lens LN identity, self-patch identity): 80 tests
PYTHONPATH=src python -m pytest tests/test_mech_alignment.py tests/test_mech_proxy.py \
    tests/test_mech_logitlens.py tests/test_mech_patching.py -q
# gates and experiments (one model per process; each merges into shared outputs)
PYTHONPATH=src python scripts/mech_m0_validate.py --models Qwen/Qwen2.5-0.5B   # gate
PYTHONPATH=src python scripts/mech_m1_logitlens.py --pilot                     # pilot gate
PYTHONPATH=src python scripts/mech_m1_logitlens.py --models <model>
PYTHONPATH=src python scripts/mech_m2_patching.py  --models <model>
PYTHONPATH=src python scripts/mech_m3_heads.py     --models <model>
PYTHONPATH=src python scripts/mech_m4_ablate.py    --top-k K --models <model> --out mech/results/m4_kK
PYTHONPATH=src python scripts/mech_d5_behavioral.py
```

Provenance: `mech/results/mech_manifest.json` (model revisions, TL version, seeds, per-run
configs), environment fingerprints embedded in every summary JSON, prompts hashed in every
records file (3 examples kept in `example_prompts_mech.json` / `example_prompts.json`).

## Verification

Every number above was re-derived from the raw records in an independent per-experiment
verification pass that also audited the text for overclaims and contradictions.
Corrections applied as a result: M2 overshoot location (L0, not
late-layer) and exact filler/import bounds; M3 L21H5 selection framing (attention-contrast
ranked, not swapped-DLA ranked) and the per-model L4 caveat; M4 collateral made precise
(k=1 conventional shift exceeds the swapped shift, CI [+0.02, +0.13]); TL-faithfulness
claims scoped to contrasts; config/manifest provenance union-merge fixed in all runners.
D5: 30/30 claims verified against raw records.
