# Shared data contracts — B0

Target namespace: `exact-explain/1`. Breaking wire/semantic changes require a new major version; additive fields use a minor version and capability flag. Generated OpenAPI and client types are produced from validated backend models. [The schema](protocol/contract.schema.json) fixes shared primitives; route models must additionally enforce this document's cross-resource rules. Do not expose raw scorer dictionaries as the public API.

## Identity and provenance

`EntityRef` = `ontology_version_id`, exact `iri`, `kind` (`class`, `object_property`, `data_property`, `individual`). Punning means kind is part of identity. Run-specific source/target roles belong to pair bindings; one ontology entity does not intrinsically have a side. Labels are display data, never identity. Keep full IRIs even where CURIEs are shown. Annotation-property matching is outside v1.

`OntologyVersion` binds root bytes/hash, ontology/version IRI when present, each resolved import document/hash and import policy, parser/package/options, logical/structural fingerprints, extraction schema and completeness. Separate root-only from resolved closure. Paths are relocatable locators, not identity or participant-facing metadata. Preserve licensing/export treatment per source. `RunContext` binds Exact revision/config, dataset/task/split, mode, ontology IDs, candidate pool and model artifacts, score definitions, runtime capabilities and source universe.

Fact IDs are SHA-256 of canonical semantic content plus ontology version and interpretation basis: exact terms, predicate/axiom/expression, literal lexical form/datatype/language and relevant derivation/projection identity. Preserve original axiom identity/origins independently of canonical grouping. A generated claim has its own ID; it cannot borrow asserted-fact status. Display text changes must not merge facts. One fact can have multiple origins; one projected feature can depend on several facts; mapping is many-to-many.

Allowed interpretation kinds: `asserted`, `structurally_derived`, `reasoner_inferred`, `projected`, `matcher_comparison`, `generated`. Each derivation records rule/provider/version, premises or explicit `premises_unavailable`, assumptions, status and scope. A reasoner hierarchy answer is not a justification certificate. Original snippets/axioms remain available when readable rendering is partial.

## Required resources

| Resource | Required content |
|---|---|
| `OntologyVersion` / `RunContext` | Identity and scope above; independent lifecycle |
| `EntityContext` | EntityRef, labels, definitions, typed synonyms, comments/xrefs as separate categories, hierarchy/restriction/usage counts or pages, capability and completeness states |
| `Fact` / `Axiom` | Stable ID, typed values or OWL expression, original syntax/format or lossless ref, origins, interpretation, availability |
| `HierarchyPage` | Selected hierarchy basis, focal entity, edges/nodes and source-axiom/derivation refs; multiple parents and cycle/equivalence treatment |
| `CandidateSet` | Run/source scope, pool identity, retrieval/score/selection rank names, stage values, final status and paging; no prebuilt graphs |
| `PairDecisionTrace` | Typed pair, ordered stage events, decisions/reasons, score definitions/values, competitors, assumptions, final membership and relation |
| `SelectedEvidence` | Channel/side/role, semantic fact refs, projected-feature refs, selected values/contributions and matcher comparison links |
| `GeneratedExplanation` | Independent profile or pair comparison; claims/citations/unknowns, generation and visibility manifest, grounding status |
| `StudyPack` / `StudyDefinition` | Server-owned versioned conditions, questionnaires, candidate presentations, packages and allocation rules; private answer key separate |
| `StudySession` / `RankingResponse` / `CaseConsultation` | Random study ID, frozen allocation/progress, partial ordered candidate IDs or explicit none/insufficient evidence; external resource self-report |
| `ReviewDecision` | Optional exploration-app relation judgment/action; distinct from the study ranking task |

Entity context must be retrievable without a run ID, candidate or graph node. Do not require the existence of a mapping to inspect an ontology. Pair evidence is immutable run evidence; additional context is explicitly separate. Factual context may grow through a new version; it cannot retroactively change what the old matcher used.

## Collections, missingness and errors

Each page returns `items`, `returned_count`, `total_count` (nullable), `next_cursor` (nullable), `truncated`, `scope`, `status`, `reason`. Counts and search results respect the active visibility policy. Hidden fact counts must not reveal withheld answers. Scope includes ontology/context revision, interpretation basis, query/filter/order and policy hash. Keyset cursors bind all of these and reject reuse after incompatible changes; ordering uses stable IDs as tie-breakers.

Statuses include `available`, `absent_in_scope`, `not_exported`, `unavailable_source`, `unresolved_import`, `unsupported`, `filtered`, `partial`, `failed`, `not_requested`, `not_run`. A zero-length list is not enough to distinguish them. `absent_in_scope` is not false, logically negated or complete domain knowledge. `filtered` is used only where disclosure of filtering is itself permitted. Do not use `total_count=0` for unknown counts.

Envelope errors provide stable machine code, user-safe message, retryability and optional job/context ID. Differentiate unknown ID (404), conflicting/stale cursor/artifact (409), invalid query (422), allowed payload bound (413) and temporarily unavailable job/input (503). A supported category with no assertions is a valid page with status, not a 404. Cross-condition access must return a uniform denial without leaking existence.

## Formal semantics

A literal retains lexical form, datatype and language. Synonyms retain exact/broad/narrow/related/preferred scope when known. A comment or definition citation is not a definition. Domain/range are OWL expressions, not necessarily named classes. Relations retain direction, inverse roles, quantifiers, cardinalities, nesting, negation and source expression. Instance types are not superclass edges. A property describing possibility (such as a 'may have' predicate) must not be paraphrased as a universal clinical observation.

Hierarchy bases: `literal_asserted`, `structural_navigation`, `reasoner_inferred`. The first preserves asserted named endpoints and original expressions; the second may expose named conjunct consequences or reduce transitive links with a documented rule; the third declares effective reasoner/completeness. Do not silently mix these under 'asserted'. Preserve equivalent components/multiple paths with bounded navigation, not exponential enumeration.

Semantic relation judgment: `equivalent`, `source_narrower`, `source_broader`, `not_equivalent`, `unresolved`; exact `<`/`>` normalization records producer convention (source subclass target means source narrower). Review action: `accept`, `reject`, `defer`. Matcher selection, benchmark relation/reference status and participant judgment are different fields/resources. Graph-closure conclusions carry their assumed cross-ontology anchors. Compatibility-mode `=` with 1.0 is a default convention, not confidence.

Every numeric value has `name`, `stage`, `value`, `range` when known and `meaning`; calibration carries artifact, population and validation identity, or explicitly `not_established`. Ordinal ranks identify their ordering and tie rule. `P_rank` is not an ordinal rank. Mixture weights and evidence quality do not imply correctness or necessity.

## Legacy and fixture requirements

Read old run layouts through RunReader. Preserve legacy IDs under a namespaced alias; resolve facts only with unambiguous ontology/version/IRI evidence. Otherwise return `not_exported`/`unresolved` with the historical display text. Recomputed context/comparisons are a derived artifact with new dependencies, not historical run evidence.

Ship contract fixtures for a real well-described class, a missing-definition class, multiple inheritance, same label/different IRI, punning, imported unlabelled predicate, every supported expression constructor, partial context, optional unsupported provider, stale cursor, withheld study data, interrupted text job and legacy evidence without IRIs. A toy complete fixture alone cannot establish readiness. Fixture provenance must distinguish actual ontology data, actual current Exact output and synthetic edge cases.

## Study wire and lifecycle extension

[Study schema](protocol/study.schema.json) defines design-time shared envelopes; B0 must complete all route models, payload limits, invariants and OpenAPI in 12. No private `CaseKey`, acceptable target IDs, case-kind/negative flag, assignment quota or token secret belongs in a participant case payload. The runtime derives participant/condition access from server state.

`RankingResponse` binds frozen case/presentation identity, typed submitted answer, ordered unique subset of the five candidates, revision, saved/submitted status and timing reference. Empty draft, explicit none and insufficient evidence differ. Nonempty ranking cannot accompany none/insufficient evidence. Questionnaire codes/versions and consultation skip rules follow 13. Preserve source/candidate IRIs internally even when opaque presentation IDs are used on screen.

Invitation secret and research participant ID are different: only the secret authorizes a session, and it is excluded from analytics/export. Mutable answers/progression use idempotency keys plus expected revisions; invitation exchange uses transactional replay-safe lookup. Event IDs deduplicate append-only telemetry without an answer-state revision; page-instance IDs support timing reconciliation. See 12 for durable transactions, authority, conflict behavior and deployment boundaries. Final JSON models may add required runtime fields; they must not weaken these invariants.
