# E07 — Listwise LLM Arbitration with Abstain Option

**Motivation** (audit obs. 6): LLM arbitration is per-pair binary (Yes/No with first-token
logprobs), invoked per gated pair. For an ambiguous source, k gated candidates cost k calls,
and the LLM never sees the alternatives it is implicitly choosing between — the single
strongest use of an LLM in matching is comparative judgment.

## Research questions

- **RQ07.1**: Does listwise arbitration improve decisions on ambiguous sources compared with
  independent binary pair judgements?
- **RQ07.2**: Does grouping candidates reduce calls and total tokens after accounting for the
  longer prompt and any order-averaging calls?
- **RQ07.3**: How sensitive are accuracy, abstention, and calibration to candidate ordering,
  list length, and sibling-heavy pools?
- **RQ07.4**: Does listwise `none` probability improve E04 NIL handling in supervised and
  target-label-free regimes?
- **RQ07.5**: Which transformation of the competing listwise probabilities yields calibrated
  pair evidence for fusion, and how does that answer change with list length?

**Hypothesis**: one listwise call per gated *source* — "here are the top-k candidates with
their evidence briefs; answer with the letter of the best match or Z for none" — (a) cuts LLM
calls/tokens ≥40% on gated sources, and (b) improves decision accuracy on those sources,
because the comparative frame resolves confusable siblings that independent binary calls both
accept.

## Change

- Gating: unchanged (U ≥ τ_LLM), but grouped by source; a source enters listwise mode when ≥2
  of its candidates gate (single-gated-pair sources keep the binary path — cheaper).
- Prompt: candidate briefs (existing pair-brief machinery, one packet per candidate,
  token-budgeted) + constrained answer alphabet {A..E, Z}; probability per candidate from
  first-token logprobs over the alphabet (same logprob plumbing as today’s A/B head). Retain
  the complete categorical vector `P(A..E,Z)`; `P(Z)` feeds E04 only through its common-scale
  ranking/calibration step.
- Probability semantics: sweep `llm.listwise_probability` over
  `raw_joint|conditional_real|pairwise_vs_none|max_normalized`:
  `raw_joint=P(letter)`; `conditional_real=P(letter)/(1−P(Z))`;
  `pairwise_vs_none=P(letter)/(P(letter)+P(Z))`; and
  `max_normalized=P(letter)/max_j P(j)`. Clamp zero denominators. `max_normalized` is a ranking
  diagnostic, not a probability, and cannot support a calibration/promotion claim. The other
  modes enter the unchanged fusion formula as `p_llm`, but each mode gets its own selector and
  score-calibration refit on permitted training data—coefficients/thresholds are never reused
  across these different score distributions.
- Overflow: `llm.listwise_max_candidates=5`. If more than five candidates gate, select the five
  highest `S_base` values (deterministic target-IRI tie-break) before randomizing prompt order;
  overflow candidates retain their non-LLM score. Record overflow-source rate and omitted-
  candidate ranks. Use a binary control restricted to the same five candidates for the quality
  comparison, while the shipped uncapped binary path remains the operational cost comparator.
  Report cost against both binary controls so savings from grouping and savings from truncation
  are distinguishable. Chunking is excluded from the primary arm because reconciling multiple
  chunk winners requires another judgement and changes the hypothesis.
- Fusion: unchanged formula, `S_final = (1−w_i)·S_base + w_i·p_llm` per candidate — the
  decomposition invariant holds.
- Optional self-consistency arm: 3 samples at temperature 0.7, majority letter, probability =
  vote share blended with logprobs (tests whether sampling beats greedy logprobs).
- Config `llm.decision_mode: binary|listwise|listwise_sc` plus the probability mode above.

Touched: decision path in the scorer + prompt builders; router unchanged.

## Arms & validation

Development: binary controls plus listwise × all probability modes; freeze one promotable
probability mode, then run listwise/listwise_sc on the full reporting matrix with 3 seeds.
Primary: macro F1 + local MRR **and** LLM cost (calls, tokens — this experiment's promotion
explicitly trades both). Report categorical NLL/Brier/top-choice ECE for `P(A..E,Z)`, pairwise
ECE/Brier after fusion/calibration, and reliability sliced by list length. Also analyze gated,
overflow/non-overflow, and sibling-confusion sources.

All paired arms pin the complete E00 LLM fingerprint; a resolved-model change aborts the
matrix. Replicate binary versus the frozen listwise mode on one development task with a second,
separately pinned logprob-capable model. This is a model-specificity diagnostic only: if the
effect reverses, scope RQ07.1 to the primary model and record cross-model generalization as
inconclusive.

**Promotion**: standard quality criteria, plus cost: listwise must be ≤ the binary same-five
token spend.

For the declared cost-reduction arm, quality is non-inferior only when the 95% CI lower bound
is above −0.5 macro F1 points versus binary same-five, and the ≥40% token reduction versus the
uncapped operational baseline has a CI excluding zero. This is the arm's pre-registered
cost-primary promotion endpoint.

E07 changes the LLM's frame while keeping every arm zero-shot and keeping the `U ≥ τ_LLM` gate.
E21 supervises the remaining choices — which exemplars the prompt shows, which pairs are worth a
call at all, and whether the judgement can be distilled — and consumes whichever decision mode
freezes here. The promoted arm of this experiment is the `label_free` resolution of the LLM
component.

**Effort**: M. **Risks**: letter-position bias in LLMs — randomize candidate order per call and
average over 2 permutations in the validation arm (measure bias explicitly); long prompts on
k=5 briefs — enforce the existing per-brief token budget; hosted models without multi-token
alphabet logprobs — capability probe falls back to binary (same pattern as today).
