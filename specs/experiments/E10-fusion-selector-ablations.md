# E10 — Fusion & Selector Ablations (γ/τ_LLM sweep, GBDT accept, pairwise accept training)

**Motivation** (audit obs. 8, 10 + finding F3): core constants have drifted through iterations
(docs said γ=0.73, shipped config says 2.0; τ_LLM has three different values across doc/config/
class default) — evidence they were once swept but the record is lost. And the acceptance
classifier trains on exactly one sample per source (its winner), discarding runner-up and
pairwise structure; both stages are linear models chosen for interpretability but never
benchmarked against stronger tabular learners on the same features.

**Hypothesis**: (a) the fusion surface around (γ, τ_LLM, β) is flat near the shipped values —
confirming them — OR reveals a better region (either outcome is valuable; the doc/config
mismatch suggests uncertainty); (b) a monotonic-constrained GBDT accept model beats logistic
by a small margin on F1 at some interpretability cost; (c) pairwise/augmented accept training
(winner + runner-up as explicit negative) improves abstention quality (AUROC of p_match) even
if F1 moves little.

## Change

All config-only or selector-internal, gated:
1. Sweep harness configs for γ ∈ {0.5, 0.73, 1, 2, 3}, τ_LLM ∈ {0.35, 0.5, 0.6}, β ∈
   {0.6, 0.8, 1.0} (3-D grid pruned by early kills on one dev task; needs WP-J's exposure of
   these as sweepable config — already true — and the tuner).
2. `selector.accept_model: logistic|gbdt_monotonic` (lightgbm with monotone constraints on
   evidence features; keep the linear model's feature set identical).
3. `selector.accept_training: winner_only|winner_plus_runnerup`.
4. Deliverable regardless of promotion: a committed **constants provenance table** — final
   swept values + dev-set evidence — replacing folklore defaults (closes finding F3's origin
   problem for good).

## Arms & validation

Stage 1 (cheap): γ/τ_LLM/β sweep on two dev tasks, 1 seed, prune to top-3 configs.
Stage 2: pruned fusion configs × {logistic, gbdt} × {winner_only, +runnerup}, full matrix,
3 seeds. Primary: macro F1 + local MRR; secondary: p_match AUROC, LLM invocation rate (τ_LLM
moves it directly — cost column mandatory), explanation-impact note (GBDT loses exact linear
attributions for the *accept* step; the pairwise-score decomposition is untouched, but the
promotion decision must weigh this explicitly against the paper's interpretability claims).

**Promotion**: standard criteria; GBDT additionally requires the interpretability trade-off to
be explicitly accepted by the owner (it changes the "fully inspectable decision process"
story — flag, don't decide, in the results note).

**Effort**: M–L (mostly compute). **Risks**: sweep overfitting to dev tasks — reporting matrix
is untouched until stage 2; lightgbm is a new dependency — optional extra, experiment-only
until promoted.
