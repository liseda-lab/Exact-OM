# Exact evidence and decision export — B1

Reuse existing scorer, candidate selector, extraction, explanation store and RunReader. This is an observability/export change. Do not change matching scores, thresholds, ranks or alignments as a side effect. Any discovered method defect is recorded and routed to the experiment suite; finish unaffected export work.

## Required implementation

1. Introduce a versioned candidate-decision export joined by run + typed source/target identity, rather than extending an undocumented fixed overlay whitelist indefinitely. Preserve original retrieval channel(s), ordinal rank/tie rule, retrieval/cross-encoder scores if produced, pair values, selector values, acceptance outcome, extraction selection, relation typing and final alignment membership. A disabled stage has status `not_run`. Missing historical stage records have event status `not_recorded` and resource availability `not_exported`; do not infer whether an unrecorded stage ran without evidence.
2. Preserve optional explicit NIL values and joint rank/winner from `exact/impl/models/selector/nil_ranking.py`. Distinguish ontology-wide no-match, no answer within a bounded pool, insufficient evidence and no selection. Do not enable new NIL inference or fit calibration merely for this UI.
3. Emit ordered immutable decision events at actual stage boundaries. Each event binds input/output artifact or values, outcome/reason, stage implementation/config identity and competing pair IDs when selection/extraction makes a trade-off. Record exact-match protection and cardinality constraints. Use the finite stage vocabulary in the shared schema; this is a bounded audit record, not a generic event platform. Do not invent a competitor from final ranks after the fact. Aggregate extraction counts alone are insufficient.
4. Replace display-only evidence IDs with semantic identities per 01. Preserve hierarchy/object/difference IRIs already generated in schema v3. Keep original axiom refs and projection/inference rule refs at feature construction. If upstream projection cannot supply a trustworthy source-axiom mapping, export `provenance_unavailable` and retain feature identity; do not guess by text. Upgrade producer/projector handoff where necessary for supported NCIT–DOID features.
5. Preserve relation/entity kind: class restrictions, property domain/range and individual type closure require different roles. `rel_iri='domain'` is not a real predicate IRI; normalize it to an explicit role/actual semantic predicate. Keep type-closure flags currently lost by hierarchy serialization.
6. Export available/eligible/selected counts per entity/channel with their scope, selection limits and omission reasons. Deduplication must preserve distinct IRIs and multiple source axioms supporting one grouped feature.
7. Bind candidate pool provenance and full decision artifact to the ontology IDs/context version, model/config hashes and final alignment artifact. `saved_alignment_member` must reconcile with actual persisted mappings, rather than merely repeating pair threshold status.

Current starting points: `exact/impl/models/pair_adaptive_scorer.py`, `pair_adaptive_channels.py`, `pair_adaptive_evidence.py`; `exact/impl/datasets/pair_adaptive_context.py`, `base.py`; `exact/impl/trainer/overlays.py`, `runner.py`; `exact/impl/extraction.py`; `exact/io/relations.py`; `exact/runs/`.

## Reconstruction and claims

An adapter may derive ordinal rank from a complete saved candidate set with the original ordering/tie policy, or join final membership from an alignment file. Record `derived_from_saved_artifacts` and dependency hashes. Recomputing a feature with current code is `recomputed`, not proof of the old run's internal state. Neither signed contributions nor an LLM mixture weight is a counterfactual. Optional new counterfactuals would require actual controlled recomputation and a separate protocol.

The minimal decision trace covers the actually enabled pipeline. Production repair is not implemented by this work; absent repair is `not_run`, not 'logically safe'. Named class/property graph-closure typing must disclose anchors; individuals remain unsupported. Disabled/default equality is visibly a convention, with no probability claim.

## Acceptance

- Before/after fixture and bounded real-run comparisons preserve candidate ordering, scores (existing precision/tolerances) and final mappings. Only specified export/provenance artifacts differ.
- A low pair score, a threshold rejection, a selector loss, a target conflict, and an extraction omission are independently inspectable when those stages run. Synthetic fixtures exercise branches absent in the chosen real run.
- Fresh current-schema NCIT–DOID records retain recoverable original facts/expressions for representative hierarchy, attribute and restriction-derived evidence. Missing provenance is explicit and counted; required supported features cannot silently all fall back to missing.
- Equal labels/different IRIs do not collide; evidence links reconcile against ontology package version and selected record IDs.
- Candidate rows, optional NIL fields and final membership survive checkpoint/overlay compaction and export/reopen.
- Legacy bundles remain readable without invented identities. No ground-truth field enters the ordinary candidate/evidence export.
