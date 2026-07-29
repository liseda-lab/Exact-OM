# E22 — Label Efficiency, Active Labelling, and Supervision-Mode Resolution

**Motivation** (audit obs. 22): with E18–E21 and E23 the programme has a supervised method at
every stage — retrieval (E20), fusion (E19), ranking (E18), LLM use (E21), graph structure
(E23), acceptance (E10), calibration (E03), NIL (E04), relation typing (E14) — and a label-free
counterpart for each (E15 for acceptance, E02 for structure), plus transfer (E16). Two questions
are then left unanswered, and both are product decisions rather than curiosities.

First, **how many labels each component actually needs**. The system currently treats supervision
as a binary switch: a training reference either resolves or it does not
(`exact/core/actions/alignment.py:323` falls back to `refs["train"]`), and the one place a
quantity is consulted is the selector's `min_positive_sources: 50`
(`selector.py:64`) — a folklore constant of exactly the kind this programme exists to replace. A
user with fifty mappings and a user with five thousand are in different regimes, and nothing
currently distinguishes them.

Second, **which resolution each component should take for a given run**. That belongs in config,
under the user's control, with a default chosen from evidence.

## Research questions

- **RQ22.1**: For each supervised component, how many effective training units are required before
  it beats its label-free counterpart, and how sharp is the crossover?
- **RQ22.2**: Given one shared pool of labelled sources that every compatible component may
  reuse, which component gains most from those labels, and which combination yields the best
  marginal quality? Where relation- or kind-specific annotations have genuinely different costs,
  where should additional annotation effort go?
- **RQ22.3**: Does active selection of which sources to label reach each crossover with fewer
  labels than random selection?
- **RQ22.4**: Can the supervision mode be resolved automatically from observable statistics —
  available label count, candidate-pool density, score separation, domain, entity kind — with
  leave-one-task-out validation and without reading gold?
- **RQ22.5**: Does a mixed configuration — supervised where labels suffice, label-free elsewhere
  — beat the all-or-nothing switch the system uses today?

## Hypotheses

The crossovers differ by roughly an order of magnitude between components: acceptance and
calibration heads, which fit few parameters over aggregate features, should cross in the low
hundreds of usable source groups; the reranker, with more parameters over per-candidate features, should
need more; encoder fine-tuning and E23's transductive graph channel should need the most and may
not cross at all on the smaller tracks. E23's crossover is additionally conditioned on the
structural profile rather than on label count alone, so it is reported as a surface over both
rather than as a single threshold. Active selection is expected to reach each crossover with
roughly half the labels of random selection, because uncertain sources carry more information than
the exact-match sources that dominate a random sample. The same labelled source is reused by
retrieval, fusion, ranking, and acceptance at no extra annotation cost; “allocation” therefore
means which components to enable and, only when annotation schemas differ, which additional
source/kind/relation labels to request. Mixed mode should beat all-or-nothing precisely because
the crossovers differ.

A label-free method that matches supervision at realistic budgets is an explicitly permitted
outcome, and would make `label_free` the better default for that component. This experiment is
designed so that result is reportable rather than awkward.

## Change

This is where the supervision-mode config lands, generalizing the pattern the selector already
uses:

```yaml
supervision:
  mode: auto|supervised|label_free        # run-level default
  components:                              # per-component override
    retrieval:   auto|supervised|label_free
    fusion:      auto|supervised|label_free
    rerank:      auto|supervised|label_free
    llm:         auto|supervised|label_free
    accept:      auto|supervised|label_free
    calibration: auto|supervised|label_free
    structure:   auto|supervised|label_free
    relation:    auto|supervised|label_free
  auto_policy:
    kind: current|threshold|profile_rule
    artifact: null                         # immutable fitted policy + feature schema
    fallback: label_free
  min_effective_training_units:            # threshold-policy defaults
    <component>:
      unit: <declared-unit>
      minimum: <int>
```

Semantics, in the shape the user selects them:

- **Effective training units are component-specific**, not raw reference-row counts. Ranking,
  fusion, acceptance, and calibration count usable labelled source groups; retrieval counts
  positive sources plus safely labelled negatives; relation typing additionally records
  per-relation/direction support; LLM gating counts counterfactual call examples; and graph
  structure counts seed sources jointly with `tbox_richness`/`relational_density`. All units are
  kind-specific and candidate-pool-aware.
- **Passing training data makes supervision eligible; it does not force it.** Under `auto`, the
  immutable policy reads only the declared observable feature schema: effective-unit counts,
  candidate-pool density/coverage, score separation, entity kind, relation support, and structural
  profile. `threshold` is the simple per-component threshold policy; `profile_rule` represents
  crossovers such as E23's label-count × graph-profile surface.
- **Not passing training data selects label-free behavior.** Under `auto`, a component with no
  resolved training reference resolves to `label_free` and never errors.
- **`supervised` and `label_free` are explicit overrides.** `label_free` ignores training data
  even when it is present, which is what makes an honest `target_label_free` measurement
  reproducible from config alone. `supervised` fails loudly rather than silently degrading when
  required labels, safe negative semantics, or a compatible fitted artifact are absent.
- Until E22 reports, `auto_policy.kind: current` reproduces today's component behavior. No
  experimental supervised component becomes an automatic default merely because a training file
  exists.

Every run records the resolved mode per component in its manifest, and the results schema carries
it as the supervision label the programme already requires. It also records the auto-policy hash,
observed feature vector, effective-unit definition/count, and resolution reason. A results row
whose supervision label disagrees with its resolved config is a harness error.

**Deliverable regardless of promotion**: per-component crossover tables/surfaces with confidence
intervals, indexed by effective training unit, entity kind, relation support, and structural
profile where applicable. They supply the threshold defaults and the optional versioned
`profile_rule` artifact. The existing `min_positive_sources: 50` remains a compatibility fallback
until a fitted rule promotes; it is not replaced by an incomparable raw-mapping count.

## Arms & validation

1. **Label-budget curves**: sample {0, 25, 50, 100, 250, 500, 1000, all} unique source annotation
   units from each training split, derive each component's effective-unit count, refit components
   independently, and evaluate against their own label-free counterparts. Use multiple
   independent subsamples per budget and report curves with CIs. The 0-label point is the complete
   promoted label-free configuration, including E15 acceptance, E05 retrieval, analytic fusion,
   and E02 structure where applicable.
2. **Shared-label component value** (RQ22.2): at each fixed annotation budget, make the same
   labelled pool available to every compatible component. Compare pre-registered
   leave-one-component-in and leave-one-component-out configurations, plus a small factorial for
   components sharing headroom. Never divide a reusable mapping budget artificially among
   retrieval, fusion, ranking, and acceptance. A separate allocation analysis is permitted only
   for annotations with genuinely different acquisition costs, such as equivalence-only mappings
   versus typed relation/direction labels; its cost model is pre-registered.
3. **Active versus random** (RQ22.3): uncertainty- and disagreement-based source selection versus
   random, at matched unique-source budgets. Once acquired, a label is reused across all compatible
   components. Selection uses only unlabelled statistics; a rule that consults gold to choose what
   to label is an oracle diagnostic and cannot promote.
4. **Mode resolution** (RQ22.4, RQ22.5): {always supervised, always label-free, current binary
   switch, fitted `auto` rule} compared leave-one-task-out. The rule may read only observable
   statistics.

**Nested validation**: budget points, active-selection rules, crossover definitions, and auto-rule
hyperparameters are developed only from training/validation labels. In each outer
leave-one-task-out fold, fit the auto policy on the other tasks; the held-out task may use only its
declared training labels to fit the component models at the simulated budget. Freeze the complete
matrix, then evaluate each held-out reporting/test split once. Reporting outcomes never update a
crossover, feature, threshold, or policy.

Three run seeds throughout, paired across arms, on every eligible track. Tracks without training
references contribute the label-free end and validate that `auto` resolves correctly for them.

Primary promotion comparison: macro F1 of the fitted `auto` rule versus the current binary switch,
under the nested leave-one-task-out protocol.
Secondary: the crossover table, per-component quality-per-label, active-versus-random label
savings at matched quality, coverage/abstention across regimes, and calibration under each mode.

Because leave-one-task-out over a small number of eligible tasks is weak evidence, the number of
independent tasks is reported beside every rule-level claim, and an underpowered comparison
yields `inconclusive` per the programme's power discipline rather than a promoted rule.

## Promotion

Before reporting results open, declare one of two promotion objectives for the single comparison
`fitted_auto` versus `current_binary`: (a) quality superiority, requiring the 95% CI for macro-F1
delta to exclude zero; or (b) efficiency, requiring the CI lower bound to exceed the
pre-registered −0.5 F1 non-inferiority margin and a declared training/inference-cost reduction
with CI excluding zero. `always_supervised` and `always_label_free` remain diagnostic comparators,
not a post-hoc oracle from which the easier promotion baseline is chosen.

On a run with no training reference, the fitted policy must resolve to the same label-free
configuration as the explicit `label_free` arm and produce byte-identical decisions for the same
seed. This is a construction/assertion gate rather than an unstable “no observed regression”
test.

Two outcomes are equally publishable and equally shippable. If supervised components dominate at
achievable budgets, `auto` resolves toward `supervised` and the crossover table tells users how
many effective training units to provide. If a label-free component matches supervision at realistic budgets,
`label_free` becomes that component's default resolution even when labels exist — the simpler,
cheaper path wins on the evidence, and the supervised path remains available by explicit config.

Thresholds, effective-unit definitions, feature schemas, and any `profile_rule` artifact derived
here are immutable, versioned outputs. They change only through a new experiment, never through
tuning against a reporting result.

**Paper contribution**: this is the programme's integrating result. Ontology-matching papers
typically report a supervised system or an unsupervised one; the supervision-budget question —
how many labels are needed, which components benefit from their shared reuse, and when they stop
paying — is rarely
asked across a whole pipeline on shared data with a shared harness. E18–E21 and E23 supply the
components; E22 supplies the curve, the allocation answer, and a deployable rule that turns the
finding into system behavior instead of a table.

**Effort**: M, compute-heavy but reusing already-fitted heads. **Risks**: budget curves are
noisy at the small end, which is exactly where the crossover sits — spend the seed budget on
subsample repetitions rather than extra budget points; crossovers are conditional on the
candidate-pool fingerprint and must be refit after any retrieval promotion; leave-one-task-out
over few tasks can overstate a rule's generality, so report task counts and prefer a simple rule
over a fitted one when their intervals overlap.
