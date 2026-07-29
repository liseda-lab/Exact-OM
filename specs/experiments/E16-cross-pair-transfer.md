# E16 — Cross-Pair Transfer and Domain Generalization

**Motivation** (audit obs. 16): supervised selector/calibration and learned relation typing are
useful only when compatible training mappings exist. Rather than treating each ontology pair as
an isolated task, Exact-OM should test whether a model trained on one or more pairs transfers to
a new pair with no target labels, and whether unlabelled target recalibration closes
distribution shift. This is distinct from E15: E15 learns no head from mapping labels for
deployment; E16 deliberately transfers a head learned from other pairs.

## Research questions

- **RQ16.1**: How transferable are the global accept classifier, score calibrator, and decision
  threshold from one ontology pair to another of the same domain and entity kind?
- **RQ16.2**: Does pooled leave-one-pair-out training generalize better than the best single
  source pair?
- **RQ16.3**: Can unlabelled target score/feature normalization improve transfer without target
  reference labels?
- **RQ16.4**: How does transfer degrade across biomedical→general-domain and class→property/
  instance shifts, and which feature/channel shifts predict failure?
- **RQ16.5**: For typed tracks, does the E14 learned relation head transfer, or is semantic
  entailment a stronger no-target-label solution?

## Hypotheses

Same-domain class-equivalence transfer should recover at least 70% of the gain that in-pair
supervision has over E15's label-free baseline. Pooled leave-one-pair-out training should be
more stable than single-pair transfer. Robust feature normalization using unlabelled target
data should reduce calibration error, but cross-kind transfer is expected to fail unless
missing-channel indicators and kind-balanced training are used.

## Transfer objects and arms

Transfer these components separately before testing their combination: (a) accept-model
coefficients, (b) score calibrator, (c) acceptance threshold, and, on typed tasks, (d) the E14
relation head. The base encoders/scorer remain frozen; transferring all components at once
would not identify the source of failure.

Any head promoted by E18–E23 joins this list as a separately transferred component: the E18
ranking head, the E19 fusion weights, the E20 fine-tuned encoder, and the E21 LLM gate and
distilled student, plus E23's inductive structural head. E23's transductive graph artifact is
pair/hash-bound by definition and is reported as non-transferable rather than illicitly reused.
Each transferable object fails differently — the fine-tuned encoder is expected to be the most
vocabulary-bound, and the LLM gate is additionally bound to a resolved model identity, so its
transfer arm is only meaningful under a pinned deployment.
Transfer results for these heads are reported per component, never as one pooled "supervised
stack transfers" claim.

For every ordered source→target pair:

1. `label_free`: strongest E15 arm, no mapping-trained head.
2. `source_only`: train/model-select on source train/validation; apply coefficients and source
   threshold unchanged to target.
3. `source_model_target_unsup_cal`: source coefficients plus label-free target normalization/
   thresholding (robust z/quantile alignment or the winning E15 calibration rule).
4. `pooled_lopo`: train on all eligible pairs except the target; balance task and entity kind;
   select hyperparameters without target labels.
5. `in_pair_supervised`: target train/validation, upper comparator.
6. `oracle_target`: post-hoc target choice, diagnostic only.

All model artifacts record their source pairs, entity kinds, relation vocabulary, feature
schema, dataset hashes, and training seed. Missing feature channels are explicit indicators,
not silently zeroed as if observed evidence were negative.

Implementation boundary: use versioned selector/relation-head artifacts with strict feature-
schema validation in the existing trainer/checkpoint layer. Transfer matrices and unlabelled
drift diagnostics belong to the E00 harness/analysis modules, not to runtime scoring logic.

## Validation matrix

1. **Within biomedical/class**: the full ordered matrix among NCIT–DOID, SNOMED–FMA, and
   SNOMED–NCIT, using only each source pair's train/validation and each target pair's untouched
   test.
2. **Cross-domain/class**: pooled biomedical sources → eligible Anatomy, Conference, OAEI-KG,
   DISO, and BioKG/KG-Align class tasks with no target refit.
3. **Cross-kind exploratory**: class-trained and kind-balanced pooled heads → E11 property and
   E12 instance tasks. Claims are separate by kind.
4. **Typed relations**: leave-one-pair-out on E14-eligible typed pairs, comparing transferred
   learned/hybrid heads with target-label-free semantic entailment.

Pre-register the complete matrix before opening target test results. Three seeds, paired across
arms. A target test may be evaluated once per frozen transfer family; it is never used to pick
the source pair or adaptation method.

Use only the development/deployable portion of E03's donor×recipient threshold matrix to
pre-register plausible source groups and normalization choices. E03's symmetric
`oracle_diagnostic` matrix is compared post hoc for explanation only and cannot choose an E16
source model.

## Metrics and analysis

Primary: target task×kind macro F1 (relation-macro F1 for the typed head). Secondary: P/R,
coverage, ECE/Brier, threshold shift, direction accuracy, and candidate recall. Report absolute
regret to `in_pair_supervised`. Also report transfer retention
`(M_transfer - M_label_free) / (M_in_pair - M_label_free)` when the denominator is positive;
otherwise report only absolute deltas.

Relate transfer failures to observable drift: feature population stability/KS statistics,
channel missingness, score quantiles, candidate-pool size, entity kind, domain, and relation
prevalence. Any proposed automatic source-model selector must use only these unlabelled
statistics and be evaluated leave-one-target-out.

## Promotion

A transferred or pooled head becomes a no-target-label option only if it beats E15 label-free
selection with CI excluding zero, improves at least two target pairs, and no target×kind or
rare-relation slice regresses by more than 1 F1 point. Package it with strict feature-schema
compatibility and training-provenance checks; otherwise fall back to E15. Cross-kind and typed
promotion decisions are separate from class-equivalence transfer.

**Pre-registered criterion-2 override**: target×kind and rare-relation slices use a 1-point
regression bound because transfer reporting subdivides each target's reference. The target
macro CI, improvement on at least two pairs, and the default 0.5-point-gate outcome are still
reported; no loss on a rare relation may be hidden by class-equivalence gains.

**Effort**: M–L. **Risks**: repeated target-test inspection can become implicit tuning; shared
ontologies (for example SNOMED in two Bio-ML pairs) can inflate apparent transfer; feature
schemas may differ by kind/format; target refs may be incomplete. Report both shared-ontology
and disjoint-ontology transfer slices, freeze the matrix, and keep oracle arms non-deployable.
