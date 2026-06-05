# alias-inertia - project scope, taxonomy & TODO (v0.2)

*Working title; rename freely. Repo: `alias-inertia`. A controlled behavioral probe - not a benchmark.*

**Plan: archival workshop paper now → substantial-extension main-conf paper later.** Because the workshop paper is *archival* (a real publication), the conference version must add genuinely new content. We therefore **deliberately reserve** the mechanistic layer, the real-code arm, and the corpus-frequency scaling-law for the main conf, and the workshop paper claims the **behavioral** result only. This keeps the extension path clean (no duplicate-publication problem). See §8.

---

## 0. One-liner & target

When a canonical import alias (`np`, `pd`, `torch`, `sk`, `xgb`, `plt`…) is bound to a target **other than its conventional one**, does a model keep tracking the *local* binding, or does the **corpus prior reassert** - and how does that depend on **distance from the import**?

**Target:** **archival** paper at an EMNLP-2026-colocated interp/eval workshop - **BlackboxNLP** (primary) / GenBench / Eval4NLP, **archival track**, up to ~8 pp + refs (a short 4-pp archival paper is also acceptable at these venues). Deadline ≈ mid-August. Goes into the ACL Anthology - a real, citable (if modest-tier) publication. Review is genuine but workshops are far more forgiving than main tracks; clean controls + clean execution carry it.

---

## 1. Core claim (falsifiable)

**H1 - prior reassertion, net of long-context rot.** For a *swapped* binding (canonical alias → a different real library), the model places probability on **prior-target** methods rather than **bound-target** methods at the usage site, and this prior-pull **grows with distance** from the import. Crucially, this exceeds the generic binding-decay seen for a **non-canonical alias bound to the same library** (which has no competing prior).

**The money figure:** prior-pull score vs. distance, three curves -
- *Conventional* (np→numpy): correct & flat (positive control / ceiling).
- *Swapped* (np→pandas): prior-pull positive and **rising** with distance.
- *No-prior* (zz→pandas): the generic-rot baseline.

> **The gap between *Swapped* and *No-prior* = corpus inertia, net of long-context rot.** That gap is the result. If it's ~0, there's no story beyond ordinary rot (informative, but stop). This is the **go/no-go gate**.

---

## 2. Taxonomy

### 2.1 Terminology (lock these)
- **Canonical alias** - a near-universal import alias (np, pd, torch, …).
- **Prior target** - the library the alias conventionally denotes (np → numpy).
- **Bound target** - what the alias denotes *in this snippet* (may equal the prior target, or not).
- **Prior reassertion / corpus inertia** - model behaving as if alias → *prior* target despite bound target ≠ prior target.
- **Binding-tracking** - model correctly using the *bound* target.
- **Prior-pull score** - metric: `logP(prior-target methods) − logP(bound-target methods)` at the usage site. `>0` = inertia, `<0` = tracking. (§4)

### 2.2 Axes

| Axis | Levels (this paper) | Role |
|---|---|---|
| **A. Binding condition** | Conventional / **Swapped** / No-prior | core treatment + controls |
| **B. Alias (prior strength)** | np, pd, torch, plt, sk, xgb (ordinal: very-common → rare) | dose ordering |
| **C. Distance** | ≈0 / 512 / 2k / 8k / 32k tokens (cap at ctx limit) | long-context axis |
| **D. Template variant** | 2-3 phrasings | prompt-robustness |
| **E. Probe type** | behavioral-logprob *(this paper)* · mechanistic *(reserved → main conf)* | measurement layer |

### 2.3 Binding conditions - exact construction
For a swap **pair** of two canonical libraries (e.g. numpy↔pandas, torch↔sklearn):
- **Conventional** - `import numpy as np` → score numpy methods. Positive control; model should be correct, no conflict.
- **Swapped** - `import pandas as np` → correct = pandas, inertia = numpy. **Treatment.**
- **No-prior** - `import pandas as zz` (non-canonical alias, no prior) → correct = pandas, no competing prior. **Generic-rot baseline.**

Cheap optional arms (worth including in an archival paper if time allows - they add robustness/interest): **Cross-swap** (np↔pd swapped *together* - two strong priors colliding); **Restatement** (swapped + `# np is pandas here` near the usage - retrieval-decay vs. hard-prior diagnostic). Defer **Fictional-target** (np→`quux`).

---

## 3. Stimulus construction - DECISION

**Programmatically generated minimal pairs.** Not AI-generated, not hand-swapped GitHub.

- **Why templates:** total control over A/B/C ⇒ clean factorial, contamination-free, reproducible, scales for $0, no validity ambiguity. Standard for probing/interp (cf. IOI). The control conditions - which are what make this attack-proof - *require* this level of control.
- **Why not GitHub-now:** (1) hand-swap is slow & error-prone; (2) swapped real numpy code is usually **invalid** under the new binding; (3) GitHub code is **in training data** → can't separate tracking from memorized-snippet recognition.
- **Why not AI-gen:** contamination + must validate/clean = more work, less control. Worst option.
- **GitHub is reserved for the main conf:** a real-code arm there, programmatically filtered for validity-under-binding + deduped/contamination-checked. That's the ecological-validity layer (part of the §8 extension).

**Stimulus shape:**
```
import {lib} as {alias}
{neutral filler - parameterised length, MUST NOT use {alias}}
{alias}.            <- read next-token distribution here
```
Filler = generic Python (defs/loops on unrelated vars) so the binding is only "needed" at the usage site; no disambiguating context near `{alias}.`.

---

## 4. Metric

At the prefix ending `{alias}.`:
- **L_prior(lib)**, **L_bound(lib)** = *discriminative* method/attr token sets (exclude shared names like `sum`,`mean`,`T`). numpy: `array, arange, zeros, linspace, dot, ndarray`; pandas: `DataFrame, read_csv, concat, merge, Series, groupby`; etc.
- `prior_pull = logsumexp(logp over L_prior) - logsumexp(logp over L_bound)`.
- **Tokenization caveat:** method names may be multi-token. Prefer **continuation-scoring** - compare teacher-forced `logP("DataFrame(")` vs `logP("array(")` over a small discriminative continuation set per library (robust to multi-token names). Single-first-token sets OK only if first tokens are discriminative.
- Conventional condition: bound = prior, so prior_pull is degenerate -> use it as a **ceiling check** (is mass on the right methods at all?).

---

## 5. Models

For an archival paper the "n=1 model" critique bites harder than for a poster, so aim for **~4-5 models** - still ~$0:
- **DeepSeek API** - primary. Exposes logprobs, cheap (~$ for thousands of short prompts), strong (prior definitely present), credible. Behavioral layer only.
- **3-4 small open models on the Bocconi A100** (free) - for a **size spread**: e.g. Qwen2.5-Coder-0.5B / 1.5B / 7B, Llama-3.2-3B. Turns "few models" into a **scaling observation** (does tracking improve, or prior-grip strengthen, with scale?).
- A100 sizing: `nvidia-smi` for 40/80 GB. bf16 ≈ 2 B/param ⇒ 7-8 B easy; 13 B fine; 30 B needs 4-bit on 40 G; 70 B = 4-bit only. Long context inflates KV cache → keep ≤7 B for the long bins.
- **No frontier closed-model API** - reserved for the main conf (spend only after the workshop validates the effect).

---

## 6. Scope

**IN (archival workshop paper):**
- Synthetic minimal pairs (full factorial, not a pilot): {Conventional, Swapped, No-prior} × ~6 aliases × ~5 distance bins × 2-3 templates, with repetitions.
- ~4-5 models (DeepSeek + 3-4 small open; size spread).
- Behavioral logprob (prior-pull) only.
- Proper stats: effect sizes + CIs; the binding×distance interaction; the **Swapped−No-prior gap** with uncertainty (mixed-effects or bootstrapped).
- 1-2 main figures + a results table.
- Thorough related work (semantic override 2602.17520; variable-renaming robustness; knowledge conflict; NoLiMa / long-context rot) - positioning is graded in archival review.
- **Limitations** section (ACL requires it); optional broader-impact.
- Optional: 1-2 cheap extra arms (Cross-swap, Restatement).

**RESERVED → main conf (hard reserve; these *are* the extension delta in §8):**
- Mechanistic layer: attention-to-import-line, activation patching, circuit localization.
- Real-GitHub-code ecological-validity arm + contamination controls.
- Rigorous corpus-frequency dose-response / scaling law.
- Frontier API models.
- Execution-prediction / bug-detection task variants.

---

## 7. TODO (≈2-3 focused weeks; archival = a real short paper. Deadline ≈ mid-Aug ⇒ spread across the ~10-week runway, interleaved with SalienceDx; don't sprint.)

- [ ] **1. Lexicons (0.5 d).** Discriminative L_prior/L_bound per swap pair (numpy↔pandas, torch↔sklearn, +1-2 more). Pick continuation strings; verify discriminative under the tokenizers used.
- [ ] **2. Stimulus generator (1.5-2 d).** Templated builder: `import {lib} as {alias}` + parameterised non-alias filler + usage prefix. Params: alias, condition, depth bin, template. Emits prompt + metadata row. *(Critical path.)*
- [ ] **3. Scorer (1-2 d).** One interface, two backends: DeepSeek API logprobs + HF teacher-forced logits. Compute prior_pull via continuation-scoring. **Reuse the SalienceDx harness.**
- [ ] **4. PILOT + go/no-go (0.5-1 d).** Tiny set (np↔pd, 3 conditions × 3 depths × 2 templates) on DeepSeek. Sanity: Conventional → right methods; No-prior tracks pandas & stays flat; metric separates conditions. **Gate:** Swapped−No-prior gap that grows with depth? If flat at depth 0, pivot the framing to *distance-specific* failure. Debug tokenization here. **Do not start the write-up until this passes.**
- [ ] **5. Full run (1-1.5 d).** Full factorial × ~4-5 models (DeepSeek + open on A100). Collect, with reruns. (+ optional arms.)
- [ ] **6. Analysis + figures + stats (2 d).** prior-pull vs distance, faceted by condition (+ alias tier / size). Interaction + gap with uncertainty. 1-2 figures + table.
- [ ] **7. Write-up to archival standard (4-6 d).** ~8 pp: intro, related work (the positioning above), method, results, limitations, conclusion. Camera-ready polish. *(This is now the real bottleneck, alongside experimental completeness.)*

**Critical-path risk:** steps 1-2 + tokenization in 4. Keep the set small until the pilot separates the controls. Effort split ≈ experiment 6-9 d / write-up 4-6 d.

---

## 8. Expansion → main conf - the substantial-extension rule (read this)

Because the workshop paper is **archival = a publication**, the main-conf paper **must add substantial new content** beyond it: you cite your own workshop paper and clearly delineate the new contribution. A re-skin of the same results is **duplicate publication** - not allowed. (Exact thresholds vary by venue; check the target conference's policy. Rule of thumb: significant new method/experiments/analysis, not just polish.)

This is *why* §6 reserves three deltas - they constitute the extension:
1. **Depth (the elevator):** mechanism - does attention from the usage token point back to the import line, and decay with distance? Activation-patch the binding representation to localize where it's stored / fails. "Observation → mechanism" is what turns a workshop note into a main-conf paper.
2. **Breadth / ecological validity:** real-GitHub-code arm (contamination-checked) + frontier models. Spend the API budget only after the workshop validates the effect.
3. **Generality:** recast "library aliases" as one instance of **high-prior token bindings**; tie to semantic-override / knowledge-conflict; make **prior frequency** the treatment variable → dose-response / scaling law (the rigorous corpus-frequency cut here).

**Net split:** workshop = behavioral observation + clean controls (archival, ~8 pp). Main conf = mechanism + ecological validity + dose-response (EMNLP/ACL/NeurIPS main). Keep the workshop's claims scoped to behavior-over-distance so the mechanism is genuinely new later.
