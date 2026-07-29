# XR-WP4 — Repair corpus, utility learning, and bounded proposals

**Status:** proposed<br>
**Depends on:** XR-WP2; XR-WP3 for rich proposal tasks; XR-E00 frozen data/split schemas<br>
**Unlocks:** XR-E04, XR-E05 learned arm, XR-E06<br>
**Research boundary:** implements L1 state utility and L2 candidate proposal for RQ5–RQ7. Learning
never establishes or participates in the safety acceptance condition.

## Objective

Build a leakage-controlled repair corpus and reproducible progression from hand-coded utility to
pointwise, pairwise, and decision-focused state models. Add bounded guard/replacement retrieval
models only after deterministic generators provide a stable baseline.

Search prioritization (proposal task L3) is optional and belongs to XR-WP5/XR-E07 because it affects
efficiency, not semantic preference.

## Training corpus

Combine, with source labels retained:

1. curated positive/relation mappings from frozen public references;
2. mechanism-aware synthetic corruptions and their traces;
3. exact safe optima and regret tables for small components;
4. expert pairwise/setwise preferences collected outside test splits;
5. weak candidate/repair outputs from established systems, never ground truth by default.

Corpus records contain core/alignment/policy/inventory hashes, split group, reference status
(positive/confirmed negative/unknown), acceptable assignment set when known, external evaluation
utility, per-state regret, and licensing/provenance. Set-valued optima are preserved rather than
arbitrarily selecting one label.

Synthetic generation uses the operators in `02-experimental-protocol.md`. A corruption is retained
only after the oracle verifies the intended effect. Corruptions of one clean conflict component
stay in one split.

## Feature groups

Implement auditable feature schemas for:

- mapping-local evidence: Exact-OM scores/margins, relation evidence, provenance, replacement rank,
  guard evidence and complexity;
- ontology structure: depth/density, parents/children, restrictions, domain/range, mapped-neighbor
  agreement, guard distinctiveness/contradiction;
- conflict context: explanation participation by direction, conflict sizes/counts, affected classes,
  centrality, counterfactual safe states;
- global context: pair/domain, OWL profile, matcher/calibration regime, alignment density, and
  candidate budget.

Feature availability flags are separate from numeric values. Text/embedding features are cached by
content hash and split-audited. Conflict features used for a candidate must be available at the
declared inference stage; do not train on conflicts discovered only after the final decision unless
the production loop also exposes them through a versioned rescoring step.

## Model progression

### L1 utility

1. deterministic hand-coded integer utility;
2. tabular pointwise classifier/regressor;
3. pairwise state ranker;
4. decision-focused/structured model;
5. optional conflict-graph model only after simpler baselines are frozen.

Every model exports a versioned inference bundle containing feature schema, normalization,
checkpoint hash, training/split manifest, output scaling, tie behavior, and supported state/compiler
versions. The inference seam returns integer utilities and uncertainty; safety fields are absent.

### L2 proposal

Train/reuse retrieval and ranking over a finite legal inventory:

- source/target endpoint alternatives;
- source/target named or constructed guards;
- qualification components.

Primary training targets are top-k coverage/listwise rank. Logical syntax/profile validity is a
deterministic filter, not a predicted permission. A constrained language model may score closed
inventory candidates but cannot emit unrestricted final OWL text.

## Objectives

- Pointwise pretraining: cross-entropy/regression or within-object pairwise ranking.
- Set-valued structured hinge: loss-augmented and acceptable-set maximizations use the same frozen
  reasoner-constrained solver.
- Decision regret: optimize a documented surrogate for regret under an external gold utility.
- Proposal retrieval: contrastive/listwise objectives with recall at the production k values.

Training jobs may cache exact safe feasible sets for small components. Any approximation in
loss-augmented inference is recorded and cannot be described as exact decision-focused training.

## Split and leakage enforcement

The data loader refuses a training/evaluation manifest with overlapping ontology-pair or conflict
group hashes. It audits:

- ontology/entity/text/embedding overlap according to the experiment's split level;
- corruption siblings across splits;
- exact-solver labels and query-basis derivation from test data;
- calibration/tuning access to held-out matcher/domain outputs;
- expert labels collected after viewing confirmatory outcomes.

Transductive pair adaptation is a separately named model/config and is never reported as offline
generalization. Reference absence is unknown; negative sampling uses only confirmed negatives or a
documented positive-unlabeled objective.

## Reproducibility and artifacts

Add `tools/repair/` commands for corpus construction, exact-label generation, training, calibration,
and inference export. Each job writes a manifest with input/output hashes, config, versions, seeds,
split group, and resource use. Checkpoints are immutable and never selected on confirmatory test
metrics.

The deterministic hand-coded utility remains available whenever a model/dependency/checkpoint is
missing. Falling back changes a quality provenance field, not policy or safety behavior.

## Tests

- feature schema and missingness round-trip;
- integer scaling/order preservation and overflow detection;
- deterministic inference for a frozen checkpoint/config;
- learned utilities cannot construct or mutate `SafetyReport`/certificate verification fields;
- split-manifest overlap/leakage rejection;
- set-valued labels preserve multiple optima;
- structured loss uses only safe assignments from the recorded solver/certificate cache;
- proposal outputs contain only inventory symbols and pass the same deterministic filters;
- model-unavailable fallback is byte-stable and recorded;
- tiny synthetic task where decision regret and local accuracy intentionally disagree.

## Acceptance criteria

XR-WP4 is done when:

1. corpus/data-card schemas distinguish positive, negative, unknown, weak, and set-valued labels;
2. split/leakage audits fail closed;
3. hand-coded, pointwise, pairwise, and decision-focused training/inference run reproducibly on a
   hermetic small corpus;
4. model bundles fully identify features, data, compiler, policy assumptions, and integer scaling;
5. L2 outputs stay within the bounded grammar/entity inventory;
6. all model failures fall back without changing the safety path;
7. XR-E04 can compare utilities over identical inventories and XR-E05 can compare proposal arms;
8. no confirmatory test data was used to select architecture or thresholds.

## Experiment handoff

- XR-E04 tests RQ5/H4; local accuracy alone cannot pass it.
- XR-E05 tests RQ6/H6 and separates retrieval, pruning, oracle, and utility failure.
- XR-E06 tests RQ7/H8 on frozen pair/domain/matcher holdouts.

Landing a learned checkpoint does not make it a default. Promotion requires G5b and a
model-specific decision record.
