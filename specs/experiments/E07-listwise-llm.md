# E07 — Listwise LLM Arbitration with Abstain Option

**Motivation** (audit obs. 6): LLM arbitration is per-pair binary (Yes/No with first-token
logprobs), invoked per gated pair. For an ambiguous source, k gated candidates cost k calls,
and the LLM never sees the alternatives it is implicitly choosing between — the single
strongest use of an LLM in matching is comparative judgment.

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
  first-token logprobs over the alphabet (same logprob plumbing as today’s A/B head);
  `p_llm(pair) = P(letter)`, `p_none = P(Z)` (feeds E04's NIL machinery — run E07 before or
  jointly with E04's accept_model arm).
- Fusion: unchanged formula, `S_final = (1−w_i)·S_base + w_i·p_llm` per candidate — the
  decomposition invariant holds.
- Optional self-consistency arm: 3 samples at temperature 0.7, majority letter, probability =
  vote share blended with logprobs (tests whether sampling beats greedy logprobs).
- Config `llm.decision_mode: binary|listwise|listwise_sc`.

Touched: decision path in the scorer + prompt builders; router unchanged.

## Arms & validation

binary (baseline) / listwise / listwise_sc. Full matrix with LLM enabled, 3 seeds. Primary:
macro F1 + local MRR **and** LLM cost (calls, tokens — this experiment's promotion explicitly
trades both). Slice analysis: accuracy on gated sources only; sibling-confusion cases
(same-parent candidates) before/after.

**Promotion**: standard quality criteria, plus cost: listwise must be ≤ binary's token spend.
If quality is neutral but cost drops ≥40%, that alone promotes (cost-reduction goal per
protocol).

**Effort**: M. **Risks**: letter-position bias in LLMs — randomize candidate order per call and
average over 2 permutations in the validation arm (measure bias explicitly); long prompts on
k=5 briefs — enforce the existing per-brief token budget; hosted models without multi-token
alphabet logprobs — capability probe falls back to binary (same pattern as today).
