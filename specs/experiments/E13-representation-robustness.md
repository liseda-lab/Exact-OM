# E13 — Representation Robustness: OWL/RDF vs CSV vs CSV+Datalog

**Motivation** (audit obs. 13; WP-G capability gap): Exact-OM accepts OWL/RDF and pure-KG
inputs, including CSV plus Datalog fact files used by BioKG-Align/KG-Align-style tasks. Passing
an end-to-end fixture does not show that evidence is preserved, that the matcher behaves
consistently across serializations, or that optional Datalog materialization is actually useful.
Format and information content must be tested separately.

## Research questions

- **RQ13.1**: When the same asserted graph and labels are serialized as OWL/RDF and CSV, do the
  source abstraction, candidate pools, evidence bundles, scores, and mappings agree?
- **RQ13.2**: When pre-materialized Datalog facts add genuine closure, do they improve
  equivalence retrieval/alignment over CSV-only, and for which entity kinds/evidence channels?
- **RQ13.3**: Are any observed gains caused by useful semantic facts rather than merely more
  edges or representation-specific identifiers?
- **RQ13.4**: Can the CSV and CSV+Datalog paths meet BioKG/KG-Align scale, memory, and typed-TSV
  output requirements without losing entities or candidates?

## Hypotheses

For semantically isomorphic inputs, entity signatures and candidate tables should be exactly
equal and scores should agree within `1e-6`; any larger quality delta is a source-layer defect,
not a modeling win. On naturally sparse CSV KGs, relevant materialized hierarchy/type facts
should improve candidate recall or F1 by at least 1 point, while degree-matched irrelevant facts
should not.

## Experimental design

This experiment has two distinct stages:

1. **Serialization parity**: export one controlled, license-compatible KG pair into semantically
   isomorphic OWL/RDF and CSV descriptors. Preserve entity IDs, label predicates, hierarchy
   direction, kinds, and literal typing. Compare `KnowledgeSource` snapshots, candidate frames,
   evidence bundles, pair scores, and final mappings at every stage.
2. **Information ablation**: on the same CSV inputs compare asserted CSV-only with CSV plus
   pre-materialized Datalog facts. Datalog files contain facts only—the current source does not
   execute rules. The rule/materializer version and hashes of both rules and resulting facts
   are recorded in run provenance. Materialization may use each KG independently but must never
   consume candidates, mappings, or reference labels.

Normalize every hierarchy edge to child→parent and every type edge to instance→class before
comparison. Emit an evidence-coverage report by predicate family so missing parser mappings are
visible before matching.

Implementation boundary: do not add format branches to the scorer. Source normalization stays
inside `exact/io/sources/{owl,rdf,csv_kg,datalog}.py`; paired source snapshots and parity
comparisons belong in the E00 analysis/harness layer. Any matcher change discovered here is a
separate flag-gated experiment, not a parser-specific exception.

## Arms

Parity: {OWL source, generic RDF source, CSV source} over identical asserted information.
Information: {CSV asserted-only, CSV+hierarchy closure, CSV+type/domain/range closure,
CSV+all materialized facts}. Add a negative control containing degree- and predicate-count-
matched irrelevant facts that do not connect a source entity to an aligned target context.
The matcher config, candidates, selector regime, and seeds are identical across arms.

For a native BioKG/KG-Align task, compare its official CSV-only and CSV+Datalog variants when
both exist; do not call this a format comparison because the information differs.

## Validation

Stage 1 uses the controlled paired serialization plus one real pair that can be exported
losslessly. Stage 2 uses eligible published BioKG/KG-Align pairs and the mini fixture for
plumbing only. Run class/property/instance slices wherever the inventory confirms them. Three
seeds for end-to-end quality; parsing and deterministic retrieval need one repeated-run parity
check.

Primary parity endpoints: entity/kind set equality, edge/evidence coverage, candidate-set
Jaccard, maximum absolute score delta, and exact mapping equality. Primary information endpoint:
task×kind macro F1. Secondary: candidate recall@k, MRR, typed-relation metrics where present,
load/index wall-time, peak memory, and typed-TSV verifier success.

## Promotion

Serialization support is considered validated only with equal signatures/candidates and score
delta ≤`1e-6`, or with every intentional semantic difference enumerated and separately ablated.
Datalog facts become a recommended profile input only if the all-facts arm beats CSV-only with
CI excluding zero, the irrelevant-facts control does not, and runtime stays within 1.5×. Native
task claims wait for a published, pinned dataset; the mini fixture can never satisfy them.

**Pre-registered criterion-3 override**: the Datalog information arm may use 1.5× the CSV-only
runtime because materialized closure necessarily loads and indexes additional facts. Report
quality per added second/fact, peak memory, and whether the default 1.2× cost gate also passes.
Serialization-parity arms receive no cost override.

E23 consumes this experiment's normalized graph and its edge-coverage report, and asks a
different question of the same inputs: not whether the representation is preserved, but whether
a supervised structural method should replace TBox-shaped evidence when the input carries little
TBox. Materialized closure and a learned graph channel are two ways to compensate for the same
sparsity, so if both promote their interaction is a required E17 contrast rather than a sum.

**Effort**: M. **Risks**: OWL constructs may have no lossless CSV equivalent; native variants
may differ in more than materialization; Datalog facts can leak mappings if generated jointly.
Limit parity claims to the declared common semantic subset and audit materialization inputs.
