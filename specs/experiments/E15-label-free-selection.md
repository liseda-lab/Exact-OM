# E15 — Target-Label-Free Selection and Calibration

**Motivation** (audit obs. 15; builds on obs. 8/10): the strongest global acceptance path fits a
classifier and threshold from reference mappings, but many tracks provide no training
alignment. A fixed threshold or an implicit fallback is not a satisfactory unsupervised
method. E03 tests individual threshold/calibration choices; this experiment tests complete,
deployable acceptance strategies when the target ontology pair has no labels.

## Research questions

- **RQ15.1**: Which target-label-free selector best balances precision, recall, and abstention
  across ontology pairs, entity kinds, and candidate-pool densities?
- **RQ15.2**: How much performance is lost relative to an in-pair supervised selector, and is
  that gap mostly calibration, ranking, or missing candidate recall?
- **RQ15.3**: Do score-distribution methods or structural/reciprocal consensus generalize more
  reliably under domain shift?
- **RQ15.4**: Can high-confidence pseudo-label self-training improve over a rule-based selector
  without collapsing on sparse or low-exact-match tasks?
- **RQ15.5**: Are label-free confidence estimates calibrated enough for NIL/abstention and
  typed-relation pipelines?

## Hypotheses

A reciprocal, margin-normalized consensus selector should beat the current no-training
fallback by at least 1 macro F1 point and remain within 1.5 points of in-pair supervision on at
least half of eligible class-equivalence tasks. Distribution-only thresholds are expected to
be brittle when match prevalence changes. One conservative self-training iteration may improve
recall, while repeated iterations are expected to propagate errors.

## Methods and arms

Every deployable arm is forbidden from reading target-pair reference labels:

1. `current_fallback`: shipped heuristic/fixed-threshold behavior.
2. `score_partition`: best pre-registered E03 distribution rule (Otsu/knee) and a two-component
   beta mixture over bounded `S_final`; select the high-score component using posterior
   confidence, with a declared degenerate-distribution fallback.
3. `reciprocal_consensus`: accept reciprocal top-1 candidates whose robust within-source margin
   exceeds a median/MAD rule and whose independent channel ranks agree; collision and
   kind/semantic conflicts cause abstention.
4. `pseudo_label_accept`: pseudo-positives are exact or reciprocal/high-margin multi-channel
   agreements; pseudo-negatives are low-tail candidates and structurally incompatible winners.
   Fit the existing interpretable accept head once, with confidence weights. No iterative arm
   beyond one update may promote in this experiment.
5. `in_pair_supervised`: current OOF-trained selector, reported as an upper comparator where
   train refs exist. Once E18–E23 report, the strongest applicable promoted supervised configuration
   replaces it as this comparator, so the label-free gap is measured against what the system
   actually does with labels rather than against the weakest supervised path.
6. `oracle_threshold`: post-hoc best target threshold, diagnostic ceiling only.

The target-label-free methods may inspect unlabelled target scores, graph structure, and
candidate sets. Method hyperparameters and fallback rules are frozen before reporting. Results
record whether those constants were analytical defaults or chosen on named, disjoint
development pairs; neither case permits target-pair labels.

Implementation boundary: add `selector.label_free_mode` and method parameters in
`exact/impl/models/selector/`; reuse its feature and acceptance code for pseudo-label fitting.
E03 owns reusable calibration primitives. The default remains the current fallback until this
experiment promotes a replacement.

## Validation

First hide training references on Bio-ML and other label-rich development pairs to measure the
gap against supervision under controlled conditions. Then report on truly no-training-label
tracks: eligible Anatomy, Conference, OAEI-KG, DISO, and BioKG/KG-Align tasks. Use the same
candidate pools and scorer across selector arms. Run class, property, and instance slices when
their evidence configuration has been frozen by E11/E12. Three seeds.

Primary: task×kind macro F1 against `current_fallback`. Secondary: P/R, candidate recall,
coverage/abstention, threshold distribution, ECE/Brier, NIL AUROC where available, and regret
to `in_pair_supervised` and `oracle_threshold`. Report prevalence, candidate-pool size, exact-
anchor coverage, and score-separation diagnostics for every task. Pseudo-label precision/recall
may be measured against gold **after** the run for analysis only.

## Promotion

Choose one label-free default only if it improves macro F1 over `current_fallback` with CI
excluding zero, no task×kind slice regresses by more than 1 point, coverage does not collapse by
more than 5 absolute percentage points from `current_fallback` on any task×kind slice, and it
has deterministic degenerate-case behavior. The supervised and oracle arms cannot become
no-label defaults. If no method dominates, select via an observable,
pre-registered task characteristic (for example score-mixture separation), validated
leave-one-task-out rather than chosen on each task's gold.

**Pre-registered criterion-2 override**: target-label-free task×kind slices use a 1-point
regression bound because several no-training tracks have small or incomplete references. The
task×kind macro CI, the absolute coverage guard above, and the default 0.5-point-gate outcome
remain mandatory.

The promoted method here is the `label_free` resolution of the acceptance component and remains
reachable by explicit config on every track, including tracks that do have training labels. E22
decides whether it also becomes the `auto` default; a label-free selector that matches
supervision at achievable label budgets is a promotable outcome, not a fallback.

**Effort**: M. **Risks**: pseudo-labels may encode the same errors as the scorer; exact anchors
are sparse in some domains; incomplete gold distorts apparent precision; unsupervised
thresholds can infer the wrong match prevalence. Keep methods conservative, expose coverage,
and treat target-label-free as a supervision claim—not as a claim that no labelled data was
ever used during method development.
