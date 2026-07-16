# WP-F — Property & Instance Matching

**Depends on**: WP-B (needs `KnowledgeSource.entities(kind)` and kind-aware hierarchy). Runs
parallel to WP-G (coordinate on the candidate-table schema — contracts §6). **Size**: L.
**Behavior**: additive — default `entity_kinds: ["class"]` reproduces class-only behavior
exactly (same candidate pools, same outputs).

## Context

Today the pipeline matches **named classes only**: candidate pools seed from
`get_all_classes()` (`exact/core/entities/ontology.py:760`, consumed at
`core/contracts/dataset.py:688`), `Entity` is class-specific, and `candidate_generation.py`
has no notion of type. Properties appear only as context edges; individuals not at all. OAEI
property/instance tracks and BioKG-style KGs need first-class support. The good news: the whole
scoring stack is signal-based (labels → lexical channel; hierarchy/neighbors → structural
channels; annotations → attribute channel), so the extension is mostly about *feeding* the same
channels kind-appropriate evidence — the models don't change.

## Design principle

**Within-kind matching only.** Classes match classes, object properties match object properties,
data properties match data properties, individuals match individuals. No cross-kind candidates
(out of scope; the schema leaves room via `SrcKind`/`TgtKind` columns).

### F1. Kind plumbing

1. `EntityKind` (contracts §1) in `exact/core/entities/kinds.py`.
2. Config: `matching.entity_kinds: ["class"]` (v2-native key — WP-J lands before this WP;
   validated subset of EntityKind values). This is the single user-facing switch for
   property/instance matching; everything below keys off it.
3. `BaseAlignmentDataset` (post-WP-B home of candidate building): iterate configured kinds;
   per kind build source/target pools from `source.entities(kind)` minus
   `excluded_from_alignment()`; run lexical retrieval per kind (pools never mix); concatenate
   candidate frames with `SrcKind`/`TgtKind` columns. Cache fingerprint incorporates
   `entity_kinds` (stale-cache safety, contracts §6).
4. `utils/candidate_generation.py`: `CandidateLabel` gains `kind`; TF-IDF/embedding index built
   per kind (a property pool of 50 shouldn't share char-gram statistics with 100k classes).
5. Reference/candidate files: unchanged formats (`SrcEntity, TgtEntity, Score`); IRIs determine
   kind at load time (lookup against the source's signatures; unknown IRIs default to the
   configured primary kind with a warning).

### F2. Evidence per kind (feeding existing channels)

| Channel (existing) | CLASS (today) | OBJECT/DATA PROPERTY | INDIVIDUAL |
|---|---|---|---|
| Lexical (SapBERT label variants) | rdfs:label + synonyms | same (labels on property IRIs) | same |
| Hierarchy family (`is_a`, `part_of`, …) | `direct_parents` + existential families | `subPropertyOf` parents (families beyond `is_a` empty) | asserted `rdf:type` classes as parents (+ their class-parents 1 level up, flagged `type_closure`) |
| Similarity/difference triples (projection neighborhood) | owl2vecstar edges | domain/range classes rendered as pseudo-triples `(prop, domain, C)`/`(prop, range, C)`; plus inverse-of/characteristics | object-property assertions `(ind, p, ind2)` from `projection_edges` extended with ABox edges |
| Attributes (literal annotations) | non-label annotations | same | same + data-property values |

Implementation: `pair_adaptive_context.get_entity_features(iri)` dispatches on kind via small
`_bundle_for_class/_bundle_for_property/_bundle_for_individual` builders, all through
`KnowledgeSource` (no OWL imports here). WP-B's store must expose ABox edges in
`projection_edges` when individuals are in scope (`include_abox=True` source option — add to
contracts §4 as an optional keyword when implementing; append-only contract change).

### F3. Scoring & outputs

- Scorer/selector/trainer: **no model changes.** Verify no hidden class-only assumptions
  (e.g. IRI-shape heuristics, `use_in_alignment` applied per kind) — fix at the dataset level.
- The selector's calibration groups by `Src`; with multiple kinds, group by (`Src`,`SrcKind`) to
  keep pools separate (guarded so single-kind behavior is bit-identical).
- Outputs: TSVs gain no new columns by default (kind is recoverable); `full_explanations.json`
  records `kind` per pair. WP-G's `typed-tsv`/`oaei-rdf` writers carry kinds naturally.
- Evaluation: builtin evaluator filters references by kind when `entity_kinds` excludes some
  (e.g. class refs present but only properties matched → warn, don't silently score 0).

### F4. `use_in_alignment` / deprecation semantics

`excluded_from_alignment()` already covers classes; extend the same annotation check to property
and individual IRIs in WP-B's store (the OAEI annotation is IRI-agnostic).

## Tests

Fixtures from WP-B already contain 2 object properties, 1 data property, 3 individuals:
1. `entity_kinds: ["class"]` → byte-identical candidate frame vs baseline (regression).
2. `["object_property"]` → pool contains exactly the fixture properties; hierarchy bundle =
   subPropertyOf; domain/range pseudo-triples present in features.
3. `["individual"]` → type-based hierarchy bundle; ABox neighbor triples; data values in
   attributes.
4. `["class","object_property","individual"]` → no cross-kind pairs anywhere in the frame;
   per-kind lexical indexes (assert index sizes).
5. End-to-end fixture run over all kinds → alignment TSV rows for each kind; local eval computes
   per-kind and overall metrics.
6. Cache safety: switching `entity_kinds` in the same output dir invalidates the dataset cache
   (fingerprint test).

## Out of scope

Cross-kind matching; complex correspondences; annotation-property matching (enum value exists,
pool building deferred); learned relation typing (WP-G heuristic only); any change to scorer
internals.

## Acceptance criteria

1. Default config: candidate frame, alignment outputs, and eval results byte-identical to
   pre-WP-F baseline on fixtures + conference smoke.
2. All six tests above green; conformance suite green (if contracts §4 gained `include_abox`).
3. A fixture run with all kinds completes end-to-end (LLM fakes) and its `full_explanations.json`
   entries carry correct `kind`.
4. Config reference (generated in WP-H) documents `entity_kinds`.
