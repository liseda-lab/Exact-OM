# Independent ontology context package — B2

The backend owns one reusable context package per frozen ontology version/closure and extraction schema, independent of matching runs. Use the retained `OwlOntologySource.owl_snapshot()` and pyowl-core indexes, not projected triples or matcher annotations as the complete ontology model. Reuse the supported shared parser/snapshot; do not create another OWL parser.

## Acquisition and installed-runtime gate

Resolve the dataset per 09, verify hosted-file hashes, and pin all imports. DOID declares `doid/obo/ext.owl`; resolve it through a recorded local import map or record root-only scope explicitly. The scope used by matching and context must agree, or additional context is labeled as an extension never used by the matcher. Do not fetch mutable imports silently at request time.

Test the installed supported pyowl-core 0.2.x distribution, matching Exact's compatibility requirements, with the native RDF/XML path on full pinned NCIT–DOID. Record distribution/wheel hash, options, source/closure hashes, diagnostics, fingerprints, runtime/RAM, and representative parity with raw source facts. Sibling-source Python/Functional Syntax probe results are feasibility evidence only. Pin a supported patched release if a necessary upstream fix is found; do not make undocumented source-path injection the production runtime.

Keep document and axiom provenance. Enable `preserve_source_map=True` when affordable/required and report its actual availability; spans are optional, original axiom and source identity are not. An old snapshot without source maps can be reloaded into a new context artifact; never pretend to reconstruct spans from labels.

## Index and representation

Publish a manifest, verified snapshot or deterministic regeneration reference, and indexed entities, terms/search, annotation facts, axioms/expressions, entity-reference postings, hierarchy adjacency, source origins and capabilities. SQLite or an equivalently bounded local index is sufficient. Keep a faithful typed/original expression artifact separate from display summaries. Root-path enumeration and full projected-edge scans per click are prohibited.

Index named direct hierarchy and explicit equivalence structures once; expose optional structural navigation with rule/source-axiom provenance. Optional inference has its own reasoner/options/import/unsupported-feature/completion manifest and can be absent without disabling asserted browsing. Do not require whole-ontology reasoning to open the UI.

Use explicit annotation-predicate mappings: NCIT P97 definition, P325 alternate definition, P90 full synonym with term metadata; DOID IAO_0000115 definition and OBO synonym scope predicates; FMA definition/synonym predicates for that adapter. Keep preferred labels, synonyms, comments, definitions, definition citations, mappings and xrefs distinct. Do not classify by substring alone. Preserve language, datatype, annotation qualifiers and citations; language fallback is visible.

Restrictions retain typed constructors and operands, including some/all/hasValue, cardinalities, inverse properties, intersections/unions and negation. Provide faithful templates for common required forms; unsupported verbalizations expose original syntax and a structured expression reference. Do not collapse nested restrictions into unqualified class-to-class relations. Render ontology-specific predicates without turning names such as 'may have' into stronger biomedical claims.

A property profile contains identity/description, hierarchy, domains/ranges as expressions, characteristics/inverse/chain axioms when supported, and usage references. An individual profile contains asserted types and typed incoming/outgoing assertions. Unsupported provider categories are explicit, not empty lists implying absence. RDF/CSV adapters may be deferred after the class-pair gate, but their capability contract cannot pretend to be OWL-complete.

## Completeness and cache behavior

Each category records scope, extraction coverage, known totals, filters and truncation separately. Import completeness is not domain completeness. A cache hit is usable only for the requested context/schema/policy and field coverage. A partial entry must not suppress a richer compatible index/native source; merge by fact identity with provenance or rebuild a versioned entry. Never merge different snapshots because IRIs match.

Context loads by EntityRef, even for an entity never scored by Exact. All named entities in the frozen scope are browsable/searchable subject to policy, while alignment eligibility is a separate flag. Keep same-label entities and edges joining existing nodes. Pagination extends beyond one hop; source and target histories are independent frontend state, not a backend single-expanded-node slot.

## Acceptance

Real NCIT–DOID checks recover known definitions, annotation sources, multiple inheritance, equivalent-class expressions and restrictions. A documented predicate resolver handles imported labels or returns a precise missing-import status. Compare expected original facts by identity against the stored/reopened index, including a sparse-definition case.

Synthetic fixtures cover punning, duplicate labels, cycles, multiple parents, anonymous expressions, typed literals/languages, nested citations, inverse/negated/cardinality constructs, unresolved imports and unsupported provider capabilities. No semantic information disappears merely because the prose renderer does not support it. Reopen a copied package with the original input directory unavailable and verify the declared portable features. Record exactly which raw inputs are necessary for regeneration versus browsing.
