# E14 — Typed Correspondence Semantics: Equivalence vs Subsumption

**Motivation** (audit obs. 14; WP-G heuristic gap): BioKG/KG-Align-style outputs distinguish
`equivalent`, `source_subsumed_by_target`, and `source_subsumes_target`, while the core score
estimates match compatibility and the current relation typer is only a hierarchy heuristic.
Pair detection and relation typing are different problems. A high untyped F1 can coexist with
systematically wrong relation directions.

Internal direction is fixed throughout this experiment: `<` means `Src ⊑ Tgt`
(`source_subsumed_by_target`); `>` means `Tgt ⊑ Src` (`source_subsumes_target`).

## Research questions

- **RQ14.1**: How much typed performance is lost in pair detection versus relation
  classification, and which relation/direction causes the loss?
- **RQ14.2**: Can a label-free semantic pipeline—high-confidence equivalence anchors followed
  by normalized graph closure or OWL bridge-axiom entailment—outperform the current hierarchy
  heuristic?
- **RQ14.3**: Does a supervised three-way relation model outperform semantic inference, and
  does a semantics-constrained hybrid outperform both?
- **RQ14.4**: Are relation probabilities calibrated well enough to abstain when neither
  equivalence nor either subsumption direction is supported?
- **RQ14.5**: Do conclusions hold for OWL hierarchies and CSV/CSV+Datalog KGs after relation
  normalization?
- **RQ14.6**: Do predicted equivalence/subsumption mappings preserve coherence when added to
  the merged hierarchy, or create cross-KG subsumption cycles and contradictions?

## Hypotheses

Semantic entailment should primarily improve directional precision and eliminate direction
reversals. A learned model should improve recall where anchors/closure are sparse. The hybrid
is expected to obtain the best relation-macro F1: entailed directions are hard constraints and
the learned head handles unresolved cases. Because equivalence is often the majority class,
micro typed F1 alone is expected to overstate quality.

## Methods

All arms consume the same candidate pairs and frozen pair scores:

1. `all_equivalent`: required lower baseline, implemented by the existing
   `matching.relation_prediction: none` mode (the writer emits `=`/`equivalent` for every
   accepted pair).
2. `hierarchy_heuristic`: current WP-G implementation.
3. `semantic_entailment` (**target-label-free**): first detect high-precision equivalence
   anchors using exact/mutual-best multi-channel agreement, then test subsumption. Its portable
   `graph_closure` backend adds anchors as bidirectional cross-source edges and materializes the
   normalized subclass/subproperty graph. A path `Src→Tgt` implies `<`, the reverse implies
   `>`, and paths in both directions imply `=`. Its OWL-only `bridge_reasoner` backend builds a
   temporary merged ontology containing the accepted equivalence bridge axioms and asks the
   configured reasoner whether each named `Src ⊑ Tgt` or `Tgt ⊑ Src` axiom is entailed. Compare
   the two backends on the supported OWL profile. CSV+Datalog consumes only independently
   pre-materialized facts through `graph_closure`. Unsupported cases abstain or fall back to
   equivalence according to a separate arm.
4. `learned_three_way` (**in-pair supervised**): multinomial/ordinal relation head over pair
   score, directional ancestor/descendant coverage, mapped parent/child agreement, lexical
   generality cues, entity kind, and evidence-missingness flags. Split by source entity and use
   class weights; never create negative/directional labels from the test reference.
5. `semantic_then_learned`: semantic entailments are hard predictions when consistent; the
   learned head handles unresolved candidates. A conflict is exposed and abstained, never
   silently overwritten.

Config: `matching.relation_prediction: none|hierarchy_heuristic|semantic_entailment|
learned_three_way|semantic_then_learned`, `matching.relation_semantic_backend:
graph_closure|bridge_reasoner`, with separate equivalence-anchor and relation-confidence
thresholds. Explanations record paths/entailed axioms and anchors or learned feature
contributions, plus `relation_confidence`.

Implementation boundary: extend the pure semantic path in `exact/io/relations.py`; place the
optional learned head with the other model implementations and invoke both through one
relation-typer interface before the existing writers. Extend the config enum and typed
evaluator, but keep `none` and `hierarchy_heuristic` behavior unchanged when new modes are off.

## Validation

Stage A uses an **oracle-pair protocol**: provide true entity pairs and score only their
relation labels. This isolates typing. Stage B uses the full pipeline and reports untyped pair
F1 beside typed mapping F1. Use only pinned tracks whose inventory confirms all needed relation
labels and split provenance; published BioKG/KG-Align is primary, with any other typed OAEI
track admitted explicitly. Synthetic closure fixtures validate direction and consistency but
provide no research result. Three seeds for learned/full-pipeline arms.

Primary: relation-macro F1 under the oracle-pair and full-pipeline protocols. Secondary:
per-relation P/R/F1, typed mapping micro F1, direction accuracy conditional on a correct pair,
abstention/coverage, ECE/Brier score, contradiction count, and performance by entity kind and
format. For coherence, quotient predicted equivalence components, add normalized directional
subsumption edges to the merged hierarchy, and report new cross-KG strongly connected
components/cycles, opposing directions, and cycle size. With `bridge_reasoner`, additionally
report merged-ontology satisfiability and newly unsatisfiable named classes. Distinguish cycles
already present inside an input from those introduced by mappings. Also report graph/reasoner
agreement and runtime on OWL tasks, plus an oracle-anchor semantic arm to separate missing
anchors from bad entailment; it is diagnostic and cannot promote.

## Promotion

A typer promotes only if relation-macro F1 improves with CI excluding zero, each directional
relation has non-zero recall and does not regress by more than 1 F1 point, direction reversals
decrease, and explanations reproduce the semantic path or learned decision. Promotion is
regime-specific: `semantic_entailment` is the `label_free` resolution of the relation component
while a learned/hybrid head is its `supervised` resolution, selected per
`supervision.components.relation`. E22 fits how many relation-balanced effective training units the learned head
requires before `auto` should prefer it.

**Pre-registered criterion-2 override**: each directional-relation slice uses a 1-point
regression bound rather than 0.5 because typed references are smaller and strongly imbalanced.
Relation-macro significance, non-zero recall in both directions, and the default-gate outcome
are still required in the results note. A promoted typer must not introduce more cross-KG
cycles/unsatisfiable named classes than `hierarchy_heuristic`; any increase requires explicit
adjudication and blocks automatic promotion.

**Effort**: L. **Risks**: incomplete references and hierarchies; severe relation imbalance;
wrong equivalence anchors create false cross-KG paths; OWL and Datalog closure may cover
different semantics. Report relation support counts, predicted-anchor and oracle-anchor arms,
and never infer a typed-performance claim from the mini fixture.
