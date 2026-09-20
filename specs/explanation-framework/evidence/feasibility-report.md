# Explanation framework: feasibility and boundaries before implementation specs

19 September 2026 · Exact-OM `e81865aceed0cd1257580097bc788329cd9084ae` · Exploration, not implementation specifications

**Recommendation: use Bio-ML 2026 NCIT–DOID as the main development pair, build an ontology-context service independent of the matcher, and separate frontend design from backend implementation through a shared, versioned data contract.** Current Exact and its ontology library already supply much of the underlying information. The main work is preserving and exposing that information faithfully, exporting the remaining decision stages, and introducing independent entity/comparison explanations. A matcher or ontology-parser rewrite is not necessary.

This investigation refines the previous UI review: the historical OMIM–ORDO bundle is poorer than the current Exact producer. Current explanation schema v3 already emits entity kinds, selected evidence IDs, hierarchy endpoint IRIs, and object/difference predicate IRIs. We should not turn historical bundle omissions into requirements to rebuild capabilities that already exist.

Evidence consists of source inspection; checksum-verified scans of the pinned full NCIT, DOID and FMA RDF/XML files; public training-cohort context measurements; three real entity-pair context examples; and a separate small synthetic capability probe against current pyOWLCore. The probe passed all 12 checks, including metadata and snapshot round-trip preservation. No matching campaign, LLM generation, or participant study was run. These measurements do not establish native-parser memory consumption, reasoner completeness, clinical equivalence, or user benefit. The application repository was not modified.

**1. What the newer datasets actually contain.**

I resolved the Bio-ML `2026` revision to `c454644a334ab43754bc1070c0fb7fdd56a90a1d`. NCIT, DOID and FMA are bundled in this release; SNOMED is supplied separately. The current tasks use whole-ontology NCIT–DOID, SNOMED–FMA and SNOMED–NCIT, unlike the older restricted modules. [Official ontology inputs](https://bio-ml.oaei-ml.org/ontologies/).

The primary comparison below uses unique endpoint entities appearing in the **public local training reference rows**. It is not a random sample of all ontology classes, not the full set of distractor candidates, and not the held-out test. NCIT and DOID are the two sides of NCIT–DOID; FMA is only the target side of SNOMED–FMA. SNOMED was not profiled.

| Measured context | NCIT source | DOID target | FMA target |
|---|---:|---:|---:|
| Unique training-reference endpoint entities | 3,741 | 3,625 | 6,653 |
| With a textual definition | 3,741 (100%) | 2,577 (71.1%) | 548 (8.2%) |
| With an OWL restriction in the class description | 1,785 (47.7%) | 1,754 (48.4%) | 3,553 (53.4%) |
| With multiple literal named `subClassOf` parents | 492 (13.2%) | 1,053 (29.0%) | 0 |
| With named direct children | 649 | 1,172 | 3,205 |

Among the 3,845 NCIT–DOID local training rows, both endpoints were found in the pinned root files; 2,759 rows (71.8%) have definitions on both sides, and 2,799 (72.8%) have a restriction on at least one side. These counts indicate that meaningful entity cards and relation inspection are possible without inventing descriptions. The remaining missing-definition cases are useful design requirements, not cases to discard.

For the whole root files, the scan found 211,958 named class declarations in NCIT, 14,703 in DOID, and 104,721 in FMA. These include deprecated or non-task classes and are not alignment eligibility counts. NCIT has 181,014 classes with definition text; DOID has 10,528; FMA has 2,140. Full-file sizes are approximately 747 MB, 28 MB and 208 MB respectively; these are disk sizes, not RAM requirements.

Important measurement qualifications:

- The scan counts explicit RDF/XML syntax, without reasoning or resolving imports. A named parent inside an equivalent-class conjunction is not counted as a literal `subClassOf` edge. NCIT has many such logical definitions: 1,798 of its training-cohort entities have an equivalent-class expression. A simple subclass-only browser would therefore miss substantial context.
- Definitions use identified predicates: NCIT `P97`/`P325`, DOID `IAO_0000115`, and FMA's `definition`. We did not classify citation/source/reviewer fields as definitions.
- Restriction presence says that an expression exists. It does not prove a mapping, imply a complete logical characterization, or guarantee useful plain-language context.
- DOID declares an external import, `http://purl.obolibrary.org/obo/doid/obo/ext.owl`. Many predicates used in its restrictions lack labels in the root document. Import resolution and pinned predicate descriptions are needed to make those relations readable. No latest remote import was silently merged into the measured snapshot.

The [machine-readable profile](context-profile-summary.json) records hashes, predicate mappings, scopes and counts. The [paired-cohort counts](context-profile-summary.json) retain the row-based denominator separately from unique-entity counts. The three [real context examples](real-entity-context-examples.json) contain original definitions, synonyms, parents and class-expression fragments for cardiovascular cancer, a sparse-definition Patau syndrome case, and Dravet syndrome. They explicitly mark matcher decisions and generated prose as not produced; they are not a finalized API or a study answer key.

**2. Dataset roles for the new framework.**

| Dataset/pair | Proposed role | Why and limits |
|---|---|---|
| **NCIT–DOID 2026** | Primary backend/frontend development and initial formative cases | Strong measured source definitions, useful target definitions, substantial restrictions and multiple inheritance; an open reproducible pair. Choose a coherent disease subset for formative work that matches recruited expertise, then test a representative broader sample. |
| **SNOMED–FMA 2026** | Small anatomy and formal-relation stress panel | FMA is structurally rich but definition-poor. It should test whether hierarchy and relation descriptions remain useful when prose is sparse. Requires the correct SNOMED conversion and permission to use/package it. Profile the SNOMED side before choosing human-study cases. |
| **SNOMED–NCIT 2026** | Later clinical/scale validation | Broader clinical setting, but it shares ontologies with the other pairs and is not a wholly independent ontology holdout. |
| **Old OMIM–ORDO bundle** | Regression fixtures | Preserve the original source sparsity failure. A switch to richer data must not hide missing-information or export-loss defects. |
| **Suitable OAEI property pair** | Only when validating actual property correspondences | Class restrictions already exercise property labels, directions and quantifiers. A separate property-alignment reference is needed for claims about matching properties themselves. |
| **BioKG-align case** | A later, small individual/KG capability check | Use the available actual dataset binding, with its own context adapter and semantics. The checked-in BioKG descriptor is currently a stub; that does not mean the user's dataset is unavailable elsewhere. |

Do not begin with an exhaustive pair-by-model experiment. First prove one NCIT–DOID path: real ontology context → current Exact record → correctly linked evidence → independent descriptions → usable comparison. Add small stress cases where that pair does not exercise the required semantics. NCIT–DOID's observed restriction forms do not comprehensively cover every OWL constructor; synthetic contract fixtures should cover universal restrictions, negation, inverses, cardinalities, cycles, duplicate labels and failures as well.

**3. What we can already obtain from Exact.**

| Product | Already implemented | Remaining qualification |
|---|---|---|
| Pair identity | Source/target IRI and entity kind, explanation schema v3 | Bind these to the ontology version; one IRI can have several OWL roles. |
| Numeric evidence | Channel scores, quality, weights, importance, centered contributions and LLM adjustment | These explain the pair score, not necessarily final alignment selection; they are not all calibrated probabilities. |
| Selected evidence | Hierarchy/relational/difference triples, attributes, per-item IDs, scores, some entity/predicate IRIs and comparison links | Selected evidence is a subset. Projected edges and matcher comparison links must not become asserted ontology facts. |
| Selection | Pair score, selection score, ranking scores/probabilities, margins, entropy, abstention, winner/reason and cardinality information | Keep threshold status, selector status, extraction status and saved-alignment membership distinct. |
| Candidate provenance | Gold-free pool manifest, retrieval configuration, data/model/config hashes, ordered pool fingerprint | Some per-candidate origin and later-stage fields are lost by the current inspection whitelist. |
| Optional relation typing | Equality/directional output, path evidence, abstention and reasons | Current graph-closure typing is conditional on cross-ontology anchors; it is not unrestricted OWL proof. Named classes, object properties and data properties are supported; individuals are explicitly unsupported. |
| Storage/recovery | Indexed explanation shards, overlays, legacy/current readers, rationale completion checkpoints | Reuse these foundations; add explicit context/prompt/schema dependency versioning. |

Code anchors: [schema v3 and score components](../../../exact/impl/models/pair_adaptive_scorer.py), [selected evidence](../../../exact/impl/models/pair_adaptive_scorer.py), [hierarchy identities](../../../exact/impl/models/pair_adaptive_channels.py), [candidate pool manifest](../../../exact/impl/datasets/base.py), [selection overlays](../../../exact/impl/trainer/overlays.py), [relation typing](../../../exact/io/relations.py), [resumable explanation store](../../../exact/runs/store.py). Paths are verified against this checkout; the report's linked source files locate the producer, not proof of a new dataset run.

**4. What the backend still needs Exact to export or preserve.**

The backend should not infer these from labels or reverse-engineer them from a rendered graph:

1. **A trace across decision stages.** Record retrieval, prefiltering, pair scoring, LLM invocation, reranking, thresholding, source/target cardinality, extraction and relation typing. Each relevant event should include the decision, rule/config identity, score meaning, and any competing pair that displaced this pair. Existing extraction diagnostics are mainly aggregate counts; they cannot reliably explain every rejected alternative.
2. **A complete candidate-decision export.** Preserve retrieval channels and ranks, reranker scores, final ranks, extraction selection, pool scope and completeness, and explicit NIL outputs where enabled. Current explicit NIL fields exist in candidate dataframes but are omitted from the fixed inspection overlay. A no-selection result must not be presented as proof that no counterpart exists anywhere in the ontology.
3. **Semantic evidence identity.** Current evidence IDs hash display text rather than all underlying IRIs and typed values. Bind IDs to snapshot, entity/predicate identity, language/datatype, original axiom/expression and evidence origin to avoid collisions between identically named entities.
4. **Fact-to-feature provenance.** Preserve original axiom/fact references and the projection or inference step responsible for a matcher feature. A three-string edge alone cannot recover quantifiers, inverse direction or the original logical expression. Existing predicate fields also need consistent semantics: some property-context fields currently use `domain`/`range` strings rather than predicate IRIs.
5. **Context-selection accounting.** Record available, eligible, selected and exported fact counts with side/channel scope and reason codes. The ontology service adds loaded/import/visible counts. This distinguishes sparse source information from a selection or export limit.
6. **Explanation generation provenance.** Add independent entity-profile and pair-comparison stages, with input context/evidence hashes, prompt text hash, schema, model/provider, decoding settings, language and completion/error state. An old nonempty rationale must not be silently retained after a requested prompt/model regeneration.

Current evidence: [display-based IDs](../../../exact/impl/models/pair_adaptive_evidence.py), [explicit NIL fields](../../../exact/impl/models/selector/nil_ranking.py), [extraction diagnostics](../../../exact/impl/extraction.py), [domain/range feature representation](../../../exact/impl/datasets/pair_adaptive_context.py), [rationale cache identity](../../../exact/impl/models/scorer_common.py).

Some missing information can be reconstructed from final alignment files, saved candidate tables and overlays. Mark it as reconstructed and record its dependencies. A recomputed evidence link is not necessarily the evidence used in a historical run. Never recover identity from a label when multiple IRIs could match.

Do not promise three capabilities that are not currently established: genuine causal counterfactual explanations, general reasoner-backed mapping proofs, or production logical-repair traces. Contribution arithmetic does not show that an LLM/fact was necessary. Optional relation mode `none` writes `=` with confidence 1.0 by convention, not verified certainty; bridge-reasoner and learned relation heads are explicitly deferred. Repair specifications/reference models exist, but production integration was not found in this checkout. [Relation-mode behavior](../../../exact/io/relations.py).

**5. What must come directly from the ontology.**

Exact retains the full shared ontology snapshot through [OwlOntologySource](../../../exact/ontology/store.py). The context backend should use that snapshot rather than extending the matcher's simplified feature methods into a complete ontology API.

| Context required by the interface | Available underlying information | Backend work |
|---|---|---|
| Entity meaning | Original labels, definitions, synonyms, language/datatype, annotation metadata and citations | Predicate registry, language policy, original/paraphrased distinction and provenance-preserving serialization |
| Hierarchy | Asserted class axioms, equivalent expressions, parent/child indexes | Search, multiple-parent navigation, bounded ancestors, pagination, original-axiom links and explicit derived views |
| Defining restrictions | Typed OWL axiom/expression structures | Faithful structured JSON and deterministic readable templates preserving quantifier, negation, nesting and cardinality |
| Surrounding context | Axiom reference postings, including constructor position and role | Distinguish defining restrictions from appearances in other classes' descriptions and from actual individual relations |
| Property meaning | Labels/definitions, complex domains/ranges, inverses, chains and characteristics | Property inspector and incoming/outgoing usage index; resolve imported predicate descriptions |
| Individual meaning | Class/object/data assertions, negative assertions, same/different individuals | Instance-specific cards; types are not class parents; missing assertions are not negative assertions |
| Source reliability/status | Document identities and hashes, imports, diagnostics, structural fingerprints and optional source spans | Portable manifest and clear unavailable/incomplete/filtered/truncated states |

The [synthetic API probe](capability-probe-scope.md) demonstrates retrieval of these richer structures from current pyOWLCore source. It uses its Python backend and a tiny Functional Syntax fixture, not native parsing of the full Bio-ML files. Native RDF/XML performance and capability parity still need a bounded implementation check.

Two semantic details must be explicit in the shared contract. First, Exact's current “asserted” hierarchy facade performs reduction and adds some consequences of equivalent expressions. Literal asserted axioms, structurally derived navigation edges, and reasoner-inferred edges are different products. Second, an existential class restriction is a statement about instances of a class; it is not an ordinary factual link between two class nodes. The readable renderer must preserve that distinction. [Hierarchy adapter](../../../exact/ontology/store.py).

Source spans require `preserve_source_map=True`; the current library default is false. Source document/axiom provenance can still be useful without a byte span, but an old artifact cannot acquire reliable original locations from labels. Reasoner output needs requested/effective reasoner, import completeness, unsupported features, timeout/fallback and query scope. The current hierarchy reasoner API does not automatically supply justification proofs. A resolved import closure is also not complete biomedical knowledge.

An independent context package should contain the frozen ontology/input manifest, verified snapshot or regeneration reference, and an indexed store for entities, annotations, original axioms, references, hierarchy and search. A read-only SQLite-style index is a pragmatic option. Build lightweight identity/hierarchy/search indexes once; retrieve larger expressions on demand. Do not scan the whole projected graph for every click or materialize every possible root path. Optional inference can be a separately cached artifact, so basic browsing does not wait for whole-ontology reasoning.

For RDF/CSV KGs, use separate capability adapters. The CSV source supplies labels, attributes and configured edges, not OWL restrictions or formally declared domains/ranges. The normalized RDF source skips blank-node structure, although its raw graph remains available. Unsupported context must return an explicit capability status. The current BioKG binding is a [stub](../../../exact/tracks/builtin/biokg.yaml); connect the user's real data before claiming an end-to-end KGA case.

**6. Proposed ownership split.**

A frontend/backend separation is appropriate, but it needs a small Exact producer workstream as well. Otherwise the backend agent will be forced to guess information the engine did not preserve.

| Owner | Owns | Boundary |
|---|---|---|
| Shared contract/fixtures | Entity identity, fact and expression types, evidence origins, score semantics, capability/missingness states, paging, condition redaction, real and synthetic examples | Agree before parallel implementation; version changes explicitly. This is not a pixel-layout specification. |
| Exact integration agent | Producer changes and run-artifact adapter: evidence provenance, full candidate/decision trace, selection accounting and generation inputs | Preserves existing scoring unless a separately authorized matching change is needed. Exports facts, not UI layout. |
| Explanation/context backend agent | Ontology package/index, predicates and readable expression renderer, APIs, study redaction, OpenRouter orchestration, generation caches and restart behavior | Joins ontology context with saved Exact artifacts while retaining different origins. Does not invent missing matcher stages. |
| Specialized frontend agent | Information architecture, entity comparison, hierarchy navigation, optional graph, responsive behavior, accessibility, stable interactions and review flow | Designs against fixture/API contracts. Does not parse OWL, call LLMs directly, compute confidence, infer equivalence or decide what omitted data means. |
| Integration/validation work | Contract conformance, real-data vertical path, missingness and semantic fixtures, performance checks, formative study | Tests frontend/backend together before freezing a human-study version. |

The shared contract should expose four independent categories: **ontology facts**, **matcher-selected/projected evidence**, **derived results with assumptions**, and **generated prose**. Suggested resources are `RunContext`, `OntologyVersion`, `EntityContext`, `HierarchyPage`, `CandidateSet`, `PairDecisionTrace`, `SelectedEvidence`, and `GeneratedExplanation`. These are proposed boundaries, not a finalized API.

An entity key should include ontology version, IRI and kind. Collections should return items, stable pagination, returned/total counts where known, scope and status. Missing information must distinguish absent from the loaded scope, unresolved import, not exported, unsupported capability, filtered, truncated, and failed extraction. Keep reference/adjudication answers in a separate server-side study resource.

The specialized frontend agent starts full design and implementation only after the shared contract, Exact integration and backend have passed their readiness gate. There is no early frontend-agent review dependency. A separate backend reviewer checks the proposed backend before implementation specs are finalized. The frontend handoff includes a working API, a frozen contract, real-data examples and intentional failure cases rather than a mock that assumes every entity has a definition and every pair has a confident answer. The three context examples produced here are a starting point; they intentionally contain no fabricated matcher score or verdict.

The backend readiness gate requires a reproducible local launch, documented API responses and errors, a current-schema NCIT–DOID example with real ontology context and matching evidence, independently versioned text outputs, redacted study fixtures, pagination/search/missingness checks, a verified interruption-and-resume path, and measured memory/query behavior on the baseline node. This is an integration gate; it does not require the final human study, every optional provider, or complete whole-ontology reasoning before frontend work begins.

**7. Dataset integration issues to fix before trusting a new bundle.**

The current [Bio-ML descriptor](../../../exact/tracks/builtin/bioml_hf.yaml) lags the released package: it says all ontologies are user-supplied, refers to `SHA256SUMS` instead of `SHA256SUMS.txt`, exposes train references but not public validation references, and points to removed repaired local candidate pools. For NCIT it also uses the original ZIP pin where the supplied OWL needs its own hash. The release documents these package changes. [Official changelog](https://bio-ml.oaei-ml.org/changelog/).

Use hosted OWL hashes rather than filename pin prefixes: NCIT's extracted OWL hash begins `1a7182a7`, whereas its filename/source ZIP pin begins `bd7a9a8a`. DOID's verified OWL hash begins `611355c4`, FMA's `beb3dc47`. The profile records the complete hashes. Verify file bytes before parsing; two local mirror files initially returned empty reads despite nonzero metadata, and were re-read or retrieved and checked before reporting data. This was a local acquisition issue, not evidence that the published ontologies are empty.

Reference semantics also matter. The official global repaired metric credits surviving pairs irrespective of their remaining relation, including weakened subsumptions. Local ranking uses the standard equivalence reference and a fixed pool containing a reference answer; it does not naturally provide ontology-level NIL cases. [Global protocol](https://bio-ml.oaei-ml.org/tasks/global/), [local protocol](https://bio-ml.oaei-ml.org/tasks/local/).

The public repaired training files contain:

| Pair | `=` | `<=` | `>=` | Total |
|---|---:|---:|---:|---:|
| NCIT–DOID | 3,085 | 165 | 12 | 3,262 |
| SNOMED–FMA | 406 | 2,809 | 24 | 3,239 |
| SNOMED–NCIT | 16,901 | 354 | 28 | 17,283 |

These are measured file contents, retained with hashes in the [relation audit](public-reference-relation-audit.json). The filename `refs_equiv` does not make every row an equivalence. Preserve original relation, normalized direction, reference basis and repair status separately from reviewer judgment. Human semantic truth still needs adjudication.

Full ontology annotations can also expose cross-ontology mapping links or cross-references related to reference construction. Retain them in the context store, but define a condition-specific visibility policy and give the baseline tool the same permitted information. Do not silently treat mapping xrefs as independent biomedical evidence, or inject reference-derived answers into descriptions. Pin imported resources and keep optional external enrichment separate from the matching snapshot.

**8. What to settle in the specifications, and what still needs a bounded check.**

We have enough evidence to fix the major architectural decisions now: NCIT–DOID first; independent ontology browsing; preserved original logical context; separate Exact decision trace; entity-independent profiles and score-blind comparison; distinct frontend/backend work; resumable versioned artifacts. Remaining uncertainty should become explicit implementation checks rather than reasons to delay design:

- Load the pinned full NCIT–DOID pair through the supported native pyowl-core/Exact path on the target node, record import resolution and peak RAM, and verify that representative definitions/expressions agree with the raw-file profile. The successful small Python-backend probe is not a substitute for this check.
- Resolve and freeze DOID imports, including predicate labels/definitions, using the same scope for matching and explanation or an explicitly identified additional-context layer.
- Generate a small fresh current-schema Exact run on public development entities. Verify score-to-evidence-to-axiom links and final decision stages. Do not reuse the old study bundle as the canonical contract.
- Define an annotation visibility policy and semantic relation handling before creating study-facing exports.
- Measure index build size, cold-open time, hierarchy/search queries and bounded entity-detail queries on the 64 GB baseline, with a separate 128 GB profile where needed. Do not promise that whole-ontology reasoning fits merely because the files are under 1 GB.
- Bind and profile the actual SNOMED/KGA data only for their targeted feature checks. Do not make the first UI delivery depend on every optional provider.

Use content-addressed ontology packages shared across runs; per-entity profile caches shared across candidates; and per-pair comparisons tied to evidence/context versions. All LLM calls remain through OpenRouter. Indexing, optional inference, profile generation, comparison generation and export should have independent completion manifests and atomic publication. Prompt fixes invalidate dependent text, while context/extraction fixes invalidate only the affected downstream products. Preserve the historical matcher evidence instead of silently regenerating its meaning.

The larger matching experiments can retain the single-node 2–3-week plan. This UI/backend feasibility work does not require rerunning all pairs or generating explanations for every possible candidate. Start with a bounded, varied development set and expand only at validation gates. Human recruitment and study execution remain a separate schedule.

**The practical conclusion is that the richer interface is technically feasible with existing foundations.** The largest missing pieces are a trustworthy ontology-context/export layer and a clearer separation of entity meaning, matcher behavior, and generated interpretation. The specifications should establish those boundaries first, then let the frontend specialist design the best presentation against real, explicitly incomplete data.
