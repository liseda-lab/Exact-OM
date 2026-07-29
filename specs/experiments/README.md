# Exact-OM Methodology Experiments Plan

**This plan is deliberately separate from the engineering overhaul (`specs/WP-*`).** Everything
here **changes results** and therefore requires empirical validation before it can touch the
product defaults. It runs **after** the overhaul completes, because it depends on: WP-J
(scoring constants exposed as config — audit finding F5), WP-C (honest cumulative timing),
WP-I (revision-pinned datasets), WP-B (the frozen parity baseline), and WP-E (extended eval).

Motivating observations 1–23 are in `specs/04-methodology-audit.md` §"Methodological
observations"; the `Finding` column below links each result-changing experiment back to its
observation. The current methodology is *sound*; these are hypotheses for improving or
extending it, not fixes.

## Research programme

The programme separates five axes that must not be conflated in one result:

1. **Decision quality** — retrieval, pair scoring, extraction, calibration, and abstention
   (E01–E10).
2. **Entity kind** — class, object/data property, and instance equivalence (E11–E12).
3. **Knowledge representation** — OWL/RDF, CSV-only, and CSV plus pre-materialized Datalog
   facts (E13).
4. **Mapping semantics** — whether an accepted pair is equivalent, source-subsumed-by-target,
   or source-subsumes-target (E14).
5. **Supervision regime** — in-pair labels, no target-pair labels, or labels transferred from
   other ontology pairs (E15–E16), and how much each pipeline stage gains from labels when they
   are available (E18–E23).

Axis 5 runs in both directions, and neither direction supersedes the other. E15/E16 ask how well
the system works when the target pair has no labels. E18–E23 ask what the system should do with
the labels that many tracks *do* ship: Bio-ML and several other pinned tracks provide substantial
train/validation splits. Today those labels already train the selector's listwise-linear ranker
and acceptance head and can fit optional LLM probability calibration; they do not train retrieval,
fusion, LLM gating/distillation, or graph structure, and the existing ranker's objective/model
family has not been ablated. Every supervised experiment therefore names both the current
supervised control and its label-free counterpart, and every results table carries the applicable
regimes.

Axis 5 also crosses axes 2 and 3 rather than running parallel to them. E23 is the clearest case:
which supervised method is appropriate depends on how much TBox structure the input actually has,
so its result is a threshold on a measured structural profile rather than a single winner.

E17 is the cross-cutting integration gate: it tests the promoted configuration across these
axes and does not assume that independently positive changes are additive.

An experiment changes one of these axes and holds the others fixed unless it explicitly
declares a factorial interaction. In particular, entity discovery, pair detection, relation
typing, and pair acceptance are reported as separate stages. This prevents a relation-typing
failure from being reported as a retrieval failure, or a materialization gain from being
reported as a CSV-parser gain.

## Dual objective: shipped system and publishable result

These specs are written to produce both a product and a paper, and the two goals constrain each
other. Pre-registration, frozen matrices, power declarations, blinded adjudication, and the
oracle/deployable separation exist because a result has to survive review, not only a release
gate. Conversely, an arm that cannot ship — one that reads target labels, depends on a mutable
hosted model, or breaks the explanation invariant — is still reportable science and is kept in
the programme as a diagnostic ceiling rather than deleted.

Two consequences are binding. A negative or null result is a deliverable: several experiments
(E10's constants provenance table, E19's fitted-versus-shipped constants, E22's crossover table)
produce their scientific artifact regardless of whether any flag flips. And the supervised and
label-free paths are both first-class outputs — the interesting claim is not that supervision
wins, but where in the pipeline it wins, by how much, and how many labels it takes.

## Experiments

| ID | Title | Finding | Cost | Expected value | Priority |
|----|-------|---------|------|----------------|----------|
| E00 | Experiment harness & baseline lineage | — | M | enables everything | **first, mandatory** |
| E05 | Candidate retrieval upgrades (encoder, fusion, adaptive k) | 3 | M | recall ceiling ↑ | **1**, before pool consumers |
| E01 | Global alignment extraction (mutual-best / assignment) | 1 | S | P↑ on 1-1 tracks, cheap | **1** |
| E03 | Score calibration, threshold transfer & tuning-objective ablation | 8 | S–M | robustness across tasks | **1** |
| E06 | String-similarity ensemble as a second lexical signal | 4 | S–M | R↑ on non-biomedical tracks | **1** |
| E07 | Listwise LLM arbitration with abstain option | 6 | M | accuracy ↑ AND LLM cost ↓ | **2** |
| E04 | NIL / abstention as a first-class output (DISO 2026) | 9 | M | required for DISO track | **2** (deadline-driven) |
| E02 | Anchor-guided structural rescoring (second pass) | 2 | M–L | biggest headroom on sparse-lexical tracks | **3** |
| E08 | Attribute-channel polarity & evidence double-counting | 5 | M | precision ↑, cleaner explanations | **3** |
| E09 | Hierarchy semantics: IC-weighted ancestor overlap + siblings | 7 | M | structural channel strength ↑ | **3** |
| E10 | Fusion & selector ablations (γ/τ_LLM sweep, GBDT accept, pairwise accept) | 8, 10 | M–L | validates/updates core constants | **3** |
| E11 | Property-equivalence matching | 11 | M | reliable object/data-property alignments | **1** |
| E12 | Instance-equivalence matching | 12 | M–L | reliable ABox/entity alignments | **1** |
| E13 | Representation robustness: OWL/RDF vs CSV vs CSV+Datalog | 13 | M | validates pure-KG coverage and evidence parity | **1** |
| E14 | Typed Correspondence Semantics: Equivalence vs Subsumption | 14 | L | relation-correct KG alignments | **2** |
| E15 | Target-label-free selection and calibration | 15 | M | deployable matching without train references | **1** |
| E16 | Cross-pair transfer and domain generalization | 16 | M–L | quantifies reuse of supervised models | **2**, after E15 |
| E18 | Supervised candidate reranking (learning-to-rank over the pool) | 18 | M | ranking ↑ from already-available labels | **1** |
| E19 | Supervised evidence fusion (learned σ-mixing weights) | 19 | M | fitted constants, LLM gating ↓ | **1** |
| E21 | Supervised LLM: exemplars, learned gating, distillation | 21 | M–L | LLM cost ↓↓ at equal quality | **2** |
| E20 | Supervised retrieval: contrastive fine-tuning + cross-encoder | 20 | L | recall ceiling ↑ | **1**, after E05 and before pool consumers |
| E23 | Supervised graph-structure matching for TBox-poor inputs | 23 | L | structural evidence where hierarchy fails | **3**, after E12/E13 |
| E22 | Label efficiency, active labelling & supervision-mode resolution | 22 | M | ships the versioned auto-resolution policy | **3**, after E18–E21, E23 |
| E24 | Contrastive-channel degeneracy (why the only penalising signal never fires) | RR-1 | M | restores precision mechanism, or removes a dead channel | **1** |
| E25 | LLM gate viability (does the branch fire, does it help) | RR-2 | M | removes a dependency or makes gating measurable | **1**, before E07/E21 |
| E26 | Quality-proxy validity ($q_k$ vs correctness; σ decomposition) | RR-3 | M | explains what the core mechanism actually does | **2** |
| E17 | Promoted-stack integration audit | 17 | M | validates interactions and release candidate | **final release gate** |

Priorities 1 → 3 = run order. E00 blocks all. E05 runs next through its end-to-end promotion
decision; candidate-consuming experiments freeze their pools only afterward. If E05 promotes,
append a rolling snapshot and regenerate candidate-pool fingerprints, capability baselines, and
power estimates before downstream experiments lock. E03's distribution-threshold stage
precedes E15.
The evidence-only stages of E11/E12 can run in parallel with class-only E15; freeze both sides
before the pre-registered property/instance selector cross. E16 consumes the strongest E15
method and frozen E11/E12 evidence bundles; its typed-head arm consumes E14. E13's
serialization-parity stage can run immediately, while its Datalog-information arm requires the
same fixed matcher configuration on both sides. E14 consumes E13's normalized hierarchy
semantics but not its quality result. E04's deadline-critical NIL/heuristic arms may run before
E07, but the E04 `accept_model` × listwise-`p_none` arm and RQ07.4 run only after or jointly
with E07; if scheduling prevents that joint arm, RQ07.4 is explicitly `inconclusive`, not
silently answered from the binary path.

The supervised experiments slot in as follows. **E20 is a retrieval change** and therefore shares
E05's slot discipline: it must reach its promotion decision before candidate-consuming
experiments freeze their pools, never after. E18 and E19 freeze only once retrieval has settled,
because both fit heads whose evidence is bound to a candidate-pool fingerprint. E19 additionally
changes `U`, so any E04, E07, or E21 comparison running alongside it pins its fusion arm rather
than mixing fusion modes across paired arms. E21 consumes the frozen E07 decision mode and needs
a one-time training-split LLM sweep budgeted before scheduling. E23 consumes E13's normalized
graph, E12's shuffle control, and E02's anchor machinery, so it follows all three; its structural
profile extends E00's inventory and must be pre-registered before its own results open. E22 runs
last among these: it consumes the promoted E15 label-free method and the fitted heads from
E18–E21 and E23, and its output supplies the versioned auto-resolution policy and
component-specific effective-training-unit thresholds behind which those arms are deployed.

E24–E26 derive from the review-response measurements in
`Paper/specs/review-response/` (findings labelled `RR-n` above) rather than from the methodology
audit, and they differ from the rest of the programme in one respect: each has a well-defined
outcome in which the correct action is to **remove or simplify** a component rather than improve
it. E25 is a prerequisite for E07 and E21, both of which assume a non-empty gated population that
has never been observed; running them first would measure a mechanism that does not engage. E24
and E25 both change $U$, so their reporting runs must not be in flight against different fusion
arms, and E26's Stage 2 decomposition should precede E19, which otherwise fits weights over a
quality term whose contribution is unquantified.

E17 runs after a batch of promotion-eligible arms has been selected and before a release flips
multiple defaults. A retrieval promotion after any downstream experiment was frozen marks that
experiment's product-promotion evidence stale: its candidate-pool fingerprint must be
reconfirmed under E17 (or the experiment rerun) before its flag can ship.

## Validation protocol (binding for every experiment)

**Dataset capability inventory (stage 0, before any run)**:

- Generate `dataset_inventory.parquet` from the pinned track layouts: task/pair, source format,
  entity counts by kind, reference counts by kind and relation, train/validation/test
  availability, candidate-pool coverage, NIL count, hierarchy predicates, and Datalog fact
  count, plus declared reference completeness (`complete|known_incomplete|unknown`). A track is
  eligible for a research claim only when the relevant cell is non-zero.
- **Development**: Bio-ML tasks with provided train/validation splits for class-equivalence
  development. Property, instance, and typed-relation development use only the explicitly
  declared training split of a suitable track; they never inherit class labels as if they
  were labels for another kind or relation.
- **Reporting**: Bio-ML test splits, Anatomy, Conference (all eligible pairs), DISO, OAEI-KG,
  and the published BioKG/KG-Align pairs when available. The mini-BioKG fixture validates I/O
  and metric plumbing only; it is too small for a performance claim.
- Test splits are touched once per pre-registered experiment family, at the end. Never tune on
  anything the evaluator subtracts or scores. The F1/F2 provenance guards from the methodology
  audit must be active.

**Supervision labels (mandatory on every results table)**:

- `in_pair_supervised`: train labels from the same ontology pair and entity kind.
- `target_label_free`: no labels from the target ontology pair; target scores/features may be
  used only by a declared unsupervised method.
- `cross_pair_transfer`: labels come from named source pairs; no target-pair refit or threshold
  selection.
- `oracle_diagnostic`: target reference used post hoc to show a ceiling. This arm is never
  deployable and can never be promoted.

Train/validation/test refers to **reference labels**, not merely to candidate files. An arm is
not called unsupervised if a threshold, feature normalizer, early-stop rule, or model choice saw
target-pair gold labels. In these specs, `target_label_free` is the operational deployment
definition of **unsupervised on the target pair**; results must still disclose any labelled
source pairs used to develop or choose the method.

Absence from a reference is not automatically a negative label. Candidate rows may be used as
confirmed negatives only when the inventory declares the relevant training reference complete,
or when an explicit negative/semantic incompatibility is available. Experiments using
`known_incomplete` references must specify a positive-unlabelled or confirmed-negative method;
negative-dependent arms on `unknown` completeness are descriptive unless separately justified.
The run manifest records the negative-label policy.

**Supervision as a configured mode**:

The labels above describe what a *result* means. The product exposes the corresponding choice as
configuration, per component, generalizing the resolution the selector already performs
(`core/actions/alignment.py:323` falls back to `refs["train"]`; `calibration.enabled: "auto"`):

- `supervision.mode: auto|supervised|label_free`, with per-component overrides under
  `supervision.components.*` for retrieval, fusion, rerank, llm, accept, calibration, structure,
  and relation typing.
- `supervision.auto_policy.kind: current|threshold|profile_rule`, an optional immutable policy
  artifact, and `min_effective_training_units` for the simple threshold policy. Effective units
  are declared per component and kind (for example usable positive source groups, safely labelled
  retrieval negatives, counterfactual LLM examples, typed-relation support, or graph seeds plus a
  structural profile); raw mapping-row counts are not interchangeable.
- **Passing training data makes supervised behavior eligible.** Under `auto`, the versioned policy
  resolves each component from its effective-unit count and other declared observable features
  such as pool coverage/density, score separation, entity kind, relation support, and
  TBox/relational profile. It never reads reporting gold.
- **Not passing training data selects label-free behavior.** Under `auto`, a component with no
  resolved training reference resolves to `label_free` and never errors.
- `label_free` as an explicit override ignores training data even when present, which is what
  makes a reproducible `target_label_free` measurement expressible in config alone. `supervised`
  as an explicit override fails loudly rather than silently degrading when required labels, safe
  negative semantics, or compatible fitted artifacts are missing.

E22 fits the effective-unit thresholds and optional profile rule that decide which resolution
`auto` should prefer; until it reports, `auto_policy.kind: current` reproduces today's behavior.
A label-free component that matches supervision at achievable label budgets becomes that
component's default resolution even where labels exist — the cheaper path wins on evidence, and
the supervised path stays reachable by explicit config.

Every run records its resolved mode per component in the run manifest, and the results schema
carries it as the supervision label together with the auto-policy hash, observed policy features,
effective-unit definition/count, and resolution reason. A results row whose supervision label
disagrees with its resolved config is a harness error, not a reporting detail.

**Runs & statistics**:
- ≥3 seeds per confirmatory/reporting configuration (baseline and variant), same seeds across
  arms. A deterministic retrieval/parity stage or development-only screening/pruning stage may
  use one seed only when it makes no reporting-set or promotion claim; every surviving arm is
  rerun with ≥3 seeds before confirmation.
- Report candidate recall before end-to-end results so retrieval misses are visible.
- Report per-task × entity-kind P/R/F1 (global) and MRR/Hits@1 (local), plus macro averages
  over tasks. Do not micro-average classes, properties, and instances into a number dominated
  by the largest kind.
- For E14, additionally report typed P/R/F1 per relation, relation-macro F1, and direction
  accuracy conditional on the entity pair being correct. Report untyped pair F1 beside typed
  F1 to isolate pair detection from relation classification.
- Significance: paired bootstrap over per-source decisions (10k resamples) on the primary
  metric; report the CI, not just the point delta. For instances, cluster by source entity (and
  by connected component as a sensitivity check) rather than treating candidate rows as
  independent.
- Cost: LLM call counts + tokens, and wall-time from the WP-C ledger (`cumulative compute`),
  reported next to quality metrics. E12/E13 also report peak memory and source-load/index time.
  A quality win that doubles cost is a finding, not a win.
- Report coverage and abstention rate next to accuracy. A method must not improve F1 merely by
  silently dropping a hard entity kind or a rare relation.
- Join E00's power table before freezing an experiment matrix. For every task×kind×relation
  primary slice, record the hypothesized effect, MDE at 80% power, and
  `powered|underpowered|descriptive`. This declaration is made before result inspection.
  Underpowered cells remain visible but yield `inconclusive`, not a post-hoc merged endpoint;
  any composite slice must be defined from development data before reporting runs.
- Before an experiment's first execution, append a generated **Pre-run power declaration**
  table to its spec (task, kind, relation, independent units, hypothesized delta, MDE80, status,
  and any pre-defined composite) and commit it with the experiment YAML. The harness refuses to
  generate or change this section after a reporting result exists.

**Known-incomplete reference protocol** (mandatory for precision-led claims):

- Official raw metrics remain primary and are never rewritten. For a task marked
  `known_incomplete` or `unknown`, pool the union of arm-disagreement mappings absent from the
  reference, deduplicate them, and sample with pre-registered strata for arm-only/both,
  task×kind, and score band.
- Hide arm identity, score, and hypothesis from at least two domain-competent annotators; a
  third adjudicates disagreements. Render identical evidence packets, record
  correct/incorrect/uncertain, and report inter-annotator agreement.
- Choose the sample size before opening predictions using E00's binomial-precision calculation
  (target CI width recorded; minimum 50 when available). Report inverse-probability-weighted
  adjusted precision with a bootstrap CI beside raw precision. Adjudication is a sensitivity
  analysis, cannot tune the arm, and uncertain cases are reported rather than forced.

**Discipline**:
- One change per causal experiment arm; if two mechanisms interact (e.g. E01×E03), run the
  2×2. E17 is the explicit exception: its purpose is to combine already-eligible changes, with
  leave-one-out and dependency-derived factorial controls.
- Each spec pre-registers numbered research questions. Every results note answers each one with
  `supported`, `not supported`, or `inconclusive`, cites the relevant table/CI, and records the
  important failure slices. A result can be useful without promoting its arm.
- Every experiment lands as: config-flagged code (default = current behavior), a runner config
  under `exp/experiments/EXX/`, and a results note appended to the experiment's spec file
  (tables + decision). No experiment code merges without its flag defaulting off.
- `B0` is the immutable post-overhaul/WP-B historical baseline. `R_n` is the immutable rolling
  snapshot of current product defaults after promotion batch `n`. Each experiment locks an
  `R_n` ID, config/commit hash, dataset lock, and candidate-pool fingerprint before running;
  its primary baseline arm is that snapshot (or its matching capability slice), while `B0` is
  also reported for long-term comparability. At programme start `R_0` points to the matching
  `B0` default/capability snapshot. No delta is taken from paper numbers.
- A promotion never overwrites a baseline. After its integration gate passes, append `R_{n+1}`
  with parent `R_n`, the promoted flags, fitted artifacts, and results hashes. If defaults move
  after an experiment locks its baseline, the scientific result remains valid for `R_n`, but
  product promotion requires confirmation against the current rolling snapshot or E17.
- Results are conditional on the recorded candidate-pool fingerprint. A retrieval change
  invalidates downstream product-default evidence on a different pool even when the original
  scientific comparison remains reportable.

## Promotion criteria (experiment → product default)

Each experiment pre-registers exactly one primary promotion comparison and endpoint for each
explicit product-default decision before opening reporting/test results (normally one per
experiment; separately deployed supervision/entity-kind regimes may name one each). A multi-arm
experiment either selects one frozen arm on development data or treats additional comparisons
as exploratory. If more than one comparison remains confirmatory and promotion-eligible, use
Holm-adjusted tests/intervals across them within that experiment family.

An arm becomes the new default only if all applicable criteria hold. The bounds below are the
defaults. A spec may replace only the task/slice-regression or cost bound with an explicitly
labelled **pre-registered override**, justified by reference-set support or an inherent
capability cost before test results are inspected. The results note must report both the
default-gate outcome and the override-gate outcome; leakage, significance, explanation, and
delivery criteria cannot be relaxed.

1. Improvement on the pre-registered primary promotion endpoint with a 95% bootstrap CI
   excluding zero, on ≥3 seeds. The endpoint is the macro quality metric for a quality arm. A
   declared cost/robustness/explanation arm may instead use its named objective only with a
   pre-registered quality non-inferiority margin and CI.
2. No single reporting task regresses by more than 0.5 F1 points (or 0.005 MRR).
3. Cost neutral or justified (wall time within 1.2×; LLM tokens within 1.2× — unless the
   experiment's stated goal is a cost reduction).
4. Explanations remain exact: the importance decomposition must still reconstruct the final
   score algebraically (this is a product invariant, not a metric).
5. The flag flip + updated generated config/docs land as their own PR citing the results note.
6. When more than one result-changing flag would ship since the last release baseline, E17's
   combined-stack gate passes. Sequentially flipping flags does not waive this requirement.

For a new entity kind, input representation, or relation type, criterion 2 is applied within
that slice as well as globally. A class-equivalence gain cannot compensate for a property,
instance, or subsumption collapse. For E15/E16, promotion is regime-specific: a label-free or
transferred default may be promoted for tracks without target training labels even when the
in-pair supervised upper bound remains better.

Promotion is likewise regime-specific for E18–E23, in the other direction. A supervised arm
promotes as the `supervised` resolution of its component; it never removes or overrides that
component's label-free path, so a run supplying no training data is unaffected by any supervised
promotion. Each supervised experiment reports its label-free comparator in the same table, and
its promotion evidence is bound to the candidate-pool fingerprint and feature schema its head was
fitted on. E22 alone supplies the versioned policy that decides what `auto` resolves to; a
supervised component winning its own experiment does not by itself make supervision the default.

## Template for new experiment specs

`EXX-title.md`: Motivation (audit finding or capability gap) · **Numbered research questions** ·
Hypotheses (falsifiable, with expected direction and rough magnitude) · Change (implementation
sketch, config flags, touched modules) · Arms & sweep · Validation (eligible tasks, splits,
entity kinds, relations, supervision label, metrics, seeds, cost) · Promotion decision rule
(one primary comparison + endpoint per explicit default decision, non-inferiority margin where
needed, and any justified pre-registered bound override) · Effort & risks · Results note
(appended after running and answering every research question, power declaration, rolling/B0
comparison, and both default/override gates).
