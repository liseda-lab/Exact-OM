# E18 — Supervised Candidate Reranking (learning-to-rank over the pool)

**Motivation** (audit obs. 18): Exact-OM already has a supervised candidate ranker. When selector
calibration is enabled, `_rank_training_groups` groups candidate rows by source and
`_fit_rank_model` fits a linear listwise softmax model over `RANK_FEATURE_NAMES`
(`selector/calibration.py:171-185,834-919`); the resulting utility is applied to every candidate
before the ten-feature top-1 acceptance head runs (`selector.py:644-741`). The missing experiment
is therefore not “supervised ranking versus no supervised ranking.” It is a controlled audit of
the existing supervised ranker: whether its listwise objective is preferable to pointwise and
pairwise alternatives, whether a more expressive model helps, how it behaves with incomplete
references and gold-absent pools, and what accuracy is lost by preserving Exact-OM's score
decomposition.

`S_base` remains the mandatory label-free comparator, but the current listwise-linear selector is
the supervised control. Optional LLM probability calibration is another existing use of training
labels and is pinned identically across all E18 arms.

## Research questions

- **RQ18.1**: At fixed retrieval and evidence, how much does the existing supervised
  listwise-linear ranker improve local ranking and global macro F1 over analytic `S_base`?
- **RQ18.2**: Do pointwise, pairwise, or NIL-aware listwise objectives improve on the existing
  listwise objective when trained on identical features and effective training sources?
- **RQ18.3**: Does a channel-gating reranker whose output exactly reconstructs from channel
  contributions match feature-additive and unconstrained tabular ceilings, and what is the
  measured cost of the product invariant?
- **RQ18.4**: After supervised reranking, does the E10 acceptance head still contribute, or has
  the ranking head absorbed its signal?
- **RQ18.5**: Which slices does reranking help — lexically weak pairs, dense pools, ambiguous
  sibling sets — and which does it leave unchanged?

## Hypotheses

The current listwise-linear ranker is expected to beat analytic `S_base` most on pools containing
confusable siblings. Pairwise training may improve hard-pool MRR but should not dominate the
current listwise objective globally. A NIL-aware listwise objective should help only when the
inventory declares complete references and a material gold-absent rate. The channel-gating model
is expected to land within 0.5 F1 points of the best feature-additive model; a larger gap is the
measured cost of the existing explanation invariant, not permission to conceal it.

## Change

No new evidence extraction and no scorer change:

1. **Frozen controls**:
   `selector.rerank: analytic|current_listwise|pointwise|pairwise|listwise_nil`.
   `analytic` bypasses the fitted ranker and orders by the current analytic score.
   `current_listwise` is a bit-for-bit control using today's `_fit_rank_model`; it is the primary
   supervised baseline, not a new arm.
2. **Features**: the objective comparison uses exactly today's `RANK_FEATURE_NAMES` vector
   (`selector.py:80-98`) for every arm. A separate development-only
   `features: current|extended` factor adds raw per-channel qualities and missingness indicators;
   it is never changed at the same time as the objective in a causal contrast. Channel-gating
   consumes the raw channel scores/qualities required by its constrained form. No feature may
   read a reference file; labels enter only the objective and split assembly.
3. **Objectives**: `pointwise` is per-row logistic; `pairwise` is RankNet over confirmed
   positive/negative pairs within a source; `current_listwise` retains today's source-group
   softmax; `listwise_nil` adds an explicit no-match item to that softmax. Source groups never
   cross folds.
4. **Reference semantics**: an unlisted candidate is a confirmed negative only when the
   inventory declares the training reference complete for that task×kind×relation, or when the
   dataset supplies an explicit negative. With `known_incomplete` references, pointwise and
   pairwise arms use a pre-registered positive-unlabelled loss or confirmed semantic
   incompatibilities; with `unknown` completeness, negative-dependent arms are descriptive.
   The manifest records which rule supplied each negative.
5. **Gold-absent groups**: `current_listwise`, pointwise, and pairwise ranking retain today's
   behavior of excluding groups with no confirmed positive from their ranking loss. Only
   `listwise_nil` may train on them, by assigning the explicit no-match item as gold, and only
   under complete references. The same groups are passed separately to E04's acceptance/NIL
   trainer; they are not silently treated as ordinary negative rows.
6. **Model family**:
   `selector.rerank_model: current_linear|channel_gating|additive_gam|gbdt_monotonic`.
   `channel_gating` predicts normalized non-negative channel weights and emits
   `Σ_c I_c·s_c`, preserving the current channel decomposition exactly.
   `additive_gam` emits an exact feature-level additive decomposition but not the current channel
   decomposition, so it is a reportable ceiling unless a separate explanation-schema product
   decision is approved. `gbdt_monotonic` is an unconstrained scientific ceiling and cannot
   promote under the current explanation contract.
7. Fitting is out-of-fold by source on the **training split only**, with the validation split
   used for early stopping and model selection. Fitted artifacts record objective, negative-label
   policy, feature/explanation schema, candidate-pool fingerprint, dataset lock, and seed.

Implementation boundary: refactor the existing
`selector/calibration.py::_fit_rank_model` behind objective/model providers under
`exact/impl/models/selector/`, reusing `_rank_training_groups` and
`features.py::_score_group`. `current_listwise` must remain bit-identical to today's path; the
scorer, channels, and retrieval are untouched.

## Arms & validation

Stage 1 (development-only screening carve-out, 1 seed, Bio-ML train/validation):
`analytic`, `current_listwise`, the objective matrix on current features, and then the
{surviving objective × eligible model family × current/extended features} matrix. Freeze
exactly one decomposition-preserving promotion candidate on development data; retain at most one
feature-additive and one unconstrained diagnostic ceiling. No reporting or promotion claim is
made here.

Stage 2 (reporting, 3 seeds): the frozen promotion candidate versus `current_listwise` as the
primary supervised comparison, with `analytic` and the frozen ceilings reported separately, on
Bio-ML test splits and every eligible task whose inventory declares both usable training labels
and reference completeness. Tracks without training references are E15's regime and appear only
through the mandatory label-free comparator column.

Every results table reports, side by side: `label_free` (analytic `S_base` plus the promoted E15
acceptance path), `current_in_pair_supervised` (today's listwise-linear ranker),
`in_pair_supervised_variant` (the frozen E18 candidate), and — once E16 has run —
`cross_pair_transfer` for that same candidate.

Primary: macro F1 over eligible task×kind cells, with local MRR as the co-primary ranking
endpoint declared before results are opened. Secondary: Hits@1, P/R, candidate recall (unchanged
by construction — report it to prove retrieval was held fixed), abstention/coverage, ECE/Brier,
channel-importance distributions before and after, and the pre-registered slices from RQ18.5.
Also report the accept-head ablation for RQ18.4 (rerank alone / accept alone / both).

## Promotion

Standard criteria on the primary endpoint. Additionally:

- The promoted arm must preserve the active explanation schema exactly. Under the current schema
  this limits promotion to `channel_gating`; feature-additive and GBDT arms remain diagnostic.
- The supervised reranker ships as the **`supervised` resolution** of the ranking component
  under `supervision.mode` (README §Supervision as a configured mode). Analytic `S_base` remains
  the `label_free` resolution and is not removed, regardless of this experiment's outcome.
- Because the head is fitted per pair and per pool, its promotion evidence is bound to the
  recorded candidate-pool fingerprint. An E05 or E20 retrieval promotion requires a refit and an
  E17 reconfirmation before the flag ships.

**Paper contribution**: the existing system already provides a rarely documented listwise-linear
supervised ranker. The publishable claim is a controlled objective/model ablation at fixed
retrieval and evidence, including incomplete-reference and NIL-aware behavior, plus the measured
cost of constraining the learned ranker to Exact-OM's channel decomposition.

**Effort**: M. **Risks**: incomplete references can turn valid alternatives into false
negatives; a no-match objective is invalid unless absence from the reference really means
absence; small training splits invite overfitting; a reranker fitted on one pool silently
degrades on another; and feature leakage would invalidate the experiment. The completeness,
artifact, split, and feature-builder assertions above are therefore promotion gates.
