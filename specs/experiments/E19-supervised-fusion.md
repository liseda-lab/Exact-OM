# E19 — Supervised Evidence Fusion (learned σ-mixing weights)

**Motivation** (audit obs. 19; builds on obs. 8 and finding F3): channel authority is decided by
one fixed analytic form — `σ_c = q_c · |s_c − τ|^γ` (`pair_adaptive_scorer.py:572`), normalized
across channels, with the lexical/structural blend
`w_struct = σ_struct / (σ_lex + σ_struct)` (`pair_adaptive_scorer.py:650`). The form asserts that
a channel's authority is its quality times its margin from τ raised to a power that is identical
for every channel, every domain, and every entity kind. γ, τ, and β are the constants whose
provenance the audit could not reconstruct. E10 sweeps them on a grid; nothing in the programme
fits them, and nothing tests the functional form itself against one learned from labels.

This is distinct from E18. E18 learns a head **after** the scorer and leaves the published score
untouched. E19 changes the score itself, so `S_base`, the τ-relative deviations, the uncertainty
`U`, and therefore the LLM gating rate all move with it. The two can both promote; their
interaction is a mandatory 2×2 contrast in E17.

## Research questions

- **RQ19.1**: Does replacing analytic σ-mixing with learned per-channel weights improve macro F1
  and local MRR at fixed evidence and fixed retrieval?
- **RQ19.2**: Is a single global weighting sufficient, or are the weights genuinely
  pair-adaptive — conditioned on evidence coverage, quality, and missingness?
- **RQ19.3**: Are the shipped constants recoverable as approximate optima of the fitted form?
  That is, was the folklore right, and was it right everywhere or only on biomedical tracks?
- **RQ19.4**: How does learned fusion change `U`, the LLM gating rate, and total LLM cost, and is
  that change beneficial independently of the quality delta?
- **RQ19.5**: Do learned weights transfer across ontology pairs, domains, and entity kinds, or is
  fusion pair-specific enough to require per-track fitting?

## Hypotheses

Learned fusion improves macro F1 by 0.5–1.5 points, with the gain concentrated on tracks where
one channel is systematically unreliable — the out-of-domain lexical channel on Conference and
OAEI-KG is the clearest case, since SapBERT cosine is trusted there by exactly the same rule as
on Bio-ML. Pair-adaptive gating is expected to beat a single global weighting, because evidence
coverage varies far more within a track than between tracks. The fitted γ is expected to sit
near the shipped value on biomedical tracks and away from it elsewhere, which would explain the
documented constant drift as two different regimes recorded at different times. LLM gating rate
should fall, because better-fused scores are less uncertain; this is a cost result that holds or
fails independently of RQ19.1.

## Change

`matching.fusion: analytic_shipped|analytic_fitted|learned_global|learned_adaptive`, plus
`matching.fusion_scope: global|per_kind|per_profile`.

1. `analytic_shipped`: today's fixed `q_c·|s_c−τ|^γ` form and constants, reproduced bit for bit.
2. `analytic_fitted`: the same parametric family, but fit `τ`, `γ`, and non-negative channel
   multipliers on the training split under bounded, pre-registered ranges. Fix the mean channel
   multiplier to 1 to remove scale non-identifiability. This is the only arm from which fitted
   `τ` and `γ` are read directly.
3. `learned_global`: per-channel non-negative weights `w_c`, fitted from labels, entering as
   `S = Σ_c (w_c q_c / Σ_j w_j q_j) · s_c`. Normalization is retained, so the importance
   decomposition remains exact by construction.
4. `learned_adaptive`: `w_c` is produced by a small non-negative gating function of observable
   channel state — quality, evidence coverage, missingness indicators, and entity kind — whose
   outputs are normalized across channels. The decomposition invariant still holds algebraically;
   only the weights' origin changes.
5. All fitted arms use the same source-disjoint trainer and reference-completeness/
   positive-unlabelled rules as E18 on the training split only, with validation for early stopping
   and model selection. Their artifacts carry parameterization, negative-label policy, feature
   schema, pool fingerprint, dataset lock, and seed.
6. **Deliverable regardless of promotion**: a fitted-versus-shipped constants table reporting
   the directly estimated `analytic_fitted` `γ`, `τ`, and channel multipliers per track and entity
   kind. For learned-global/adaptive models, an optional post-hoc projection onto the analytic
   family may be reported only with its projection loss and residual; projected values are
   labelled approximations, never constants uniquely implied by the learned model. This closes
   the folklore question alongside E10's swept provenance table.

Implementation boundary: the mixing computation in `pair_adaptive_scorer.py` gains a weight
provider; `analytic_shipped` returns today's `q · |s − τ|^γ` and is the default. Channel
implementations in `pair_adaptive_channels.py` are untouched — this experiment changes how
channels are combined, never what they measure.

## Arms & validation

Stage 1 (development-only screening, 1 seed):
{analytic_shipped, analytic_fitted, learned_global, learned_adaptive} × applicable
{global, per_kind} scopes on Bio-ML train/validation. Freeze exactly one promotion candidate on
development data; retain the other fitted arms as diagnostics for RQ19.2/RQ19.3.

Stage 2 (reporting, 3 seeds): the frozen candidate versus `analytic_shipped` on Bio-ML test and
every eligible track with a training split, with the label-free comparator column mandatory as in
E18. A 2×2 against E10's accept model runs here, because a better-fused score and a better accept
head plausibly compete for the same headroom.

Primary: macro F1. Co-primary declared before results open: LLM gating rate and token spend, as
RQ19.4 is a cost claim that must not be answered post hoc from a quality run. Secondary: local
MRR, P/R, ECE/Brier, per-channel weight distributions and their entropy, coverage/abstention, and
the fitted-constants table.

Weight entropy is reported for every arm. A fused score that collapses onto a single channel may
still score well while destroying the multi-channel explanation the product delivers; that
outcome must be visible in the results note rather than discovered after promotion.

## Promotion

Standard criteria, with three additions:

- Exact explanation reconstruction is a hard constraint here, not a trade-off. An arm that
  breaks it cannot promote in any form.
- Mean per-pair weight entropy must not fall below a bound pre-registered on development data,
  so a promoted fusion still explains decisions through multiple channels.
- The learned weighting ships as the **`supervised` resolution** of the fusion component under
  `supervision.mode`; `analytic_shipped` remains the `label_free` resolution and the default for
  tracks without training references.

**Paper contribution**: the analytic quality×margin fusion rule is the methodological core of the
system and is currently justified by construction rather than by evidence. Fitting the same form
from labels and reporting where the fitted optimum agrees with the shipped constants — and where
it does not — is a substantive result either way, and RQ19.4 supplies a rarely reported finding:
better fusion reduces the number of LLM calls a matcher needs.

**Effort**: M. **Risks**: learned weights can collapse onto one channel (entropy guard above);
changing `U` moves LLM cost and NIL behavior together, so E04 and E07 comparisons must pin the
fusion arm; per-kind fitting fragments already small property and instance references — report
per-kind sample counts and fall back to the global scope when a kind is underpowered.
