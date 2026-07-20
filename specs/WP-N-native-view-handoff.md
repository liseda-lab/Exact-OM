# WP-N — Encoded native-consumer compatibility handoff

**Target release:** Exact-OM `2.1.0` as part of WP-M M5 unless that line has already shipped, in
which case the next patch release. **Status:** implementation checkpoint in progress; release
acceptance remains open.

**Depends on:** WP-M M0–M4; a released/candidate pyowl-core WP17 encoded-view contract; pyELK
WP14; pyHermiT WP18; and projector P7. It does not require all three consumers to finish before
work starts: their compatibility tests may proceed in parallel against the frozen core schema.

## 1. Purpose and boundary

Exact already owns one `pyowl_core.OntologyView` per ontology and passes that identity to the
projector and optional reasoners. WP-N makes Exact compatible with their optimized encoded-native
compilers and supplies the missing workflow-level proof needed by WP-M M5.

Exact MUST NOT decode `EncodedStructuralView`, import any native extension, select private compiler
entry points, or cache schema-local IDs. Each consumer negotiates the public core capability and
owns its private IR. Exact continues to pass only its existing snapshot/provider, overlay, or
composite identity through public APIs.

This is a compatibility/performance handoff, not a matching-methodology change. Alignment scores,
projection profiles/options, asserted default hierarchy, reasoner choice, repair semantics, and
public output formats remain unchanged.

## 2. Required behavior

- `OwlOntologySource.owl_snapshot()` returns the exact retained view as today; no eager call to
  `iter_axioms()`, `signature()`, wire encoding, or view materialization is added.
- Projection passes that identity to projector P7. ELK/HermiT adapters pass it to the matching
  reasoner successor without a path, bytes export, or consumer-specific conversion.
- Source/target plus bridge repair/evaluation composites retain their bases and delta segments.
  Exact never requests a flattened encoded view.
- In-process workflows use the retained owner directly. Worker workflows continue using the
  stable core wire/mmap boundary once, after which the consumer may request an encoded view from
  the mapped snapshot without parsing OWL.
- Scalar-only core providers and pure consumer wheels remain supported. Missing acceleration may
  change diagnostics and timing only, never output behavior or installability.

Capability absence is not an Exact error unless a new explicit performance-only diagnostic mode
requires encoded acceleration. Corrupt or incompatible encoded data propagates the consumer/core
protocol error; Exact does not retry from a path after partial work.

## 3. Dependencies and version negotiation

Implementation pins compatible released ranges for:

- core package/API/adapter plus the supported encoded schema;
- projector package/API/profile/compiler-cache schema;
- pyELK package/compiler/native ingestion schema; and
- pyHermiT package/compiler/native ingestion schema.

The base installation still requires only core and the projector. Reasoners remain in the
`reasoning` extra. No direct dependency on Rust crates, Horned-OWL, OWLAPI, Java, OAEI, or private
accelerator distributions is introduced.

Ranges change only after exact wheel/sdist combinations are installed and tested. Exact continues
to work with a compatible scalar-only core/consumer combination inside the documented range;
optimized capability support is negotiated, not inferred from a package version string.

## 4. Provenance and cache compatibility

Extend the existing `ontology_stack` provenance with an additive `consumer_handoff` record:

```text
core encoded schema/descriptor digest and storage backend
consumer selected ingestion path: scalar-python | scalar-native/wire | encoded-native
consumer compiler/native schema and implementation version
encoded owner kind: direct | decoded | mmap | overlay | composite
materialized scalar rows, copied structural bytes, core-wire and parser counter deltas
```

Only values exposed by stable public diagnostics are recorded. Object IDs, pointers, buffer
addresses, private arena IDs, paths, and credentials are forbidden.

Projection and reasoner cache keys add the encoded schema/descriptor and consumer compiler schema
where the consumer reports them. Old caches are invalidated or rebuilt with an actionable reason;
Exact never reinterprets dense encoded IDs. Matching/result cache semantics do not change merely
because compilation became faster.

## 5. Verification

Hermetic tests cover projector Python/scalar-native/encoded-native and optional pyELK/pyHermiT
scalar/encoded paths on Python 3.10–3.12 at minimum. Instrumentation proves:

- each source/target is parsed and resolved exactly once;
- Exact, projector, and reasoners observe the original in-process view identity;
- encoded-native compilation increments no scalar axiom/term, core wire, parser, resolver,
  per-row FFI, or base-flattening counter;
- mmap workers parse zero OWL documents and retain the mapped encoded owner through completion;
- projection edge counters/digests and all WP-M classified NCIT differences remain exact;
- asserted/ELK/HermiT hierarchy bundles, coherence results, alignments, scores, and output files are
  backend/ingestion-path identical; and
- consumer absence, pure wheels, scalar-only core, corrupt/incompatible capability, timeout,
  cancellation, and fallback diagnostics remain actionable and Java-free.

Dependency-DAG scans continue to prove shared packages do not import Exact and Exact does not
acquire OAEI in its base/reasoning installations.

## 6. Performance acceptance

Rerun WP-M's exact content-addressed NCIT–DOID evidence plus GO and the largest licensed workflow
available. Record core load, encoded-view publication, projector/reasoner compilation, first
result, complete result/artifact, cache hit, wall/CPU time, RSS, copied bytes, scalar objects, and
result digests.

WP-N does not invent a weaker threshold. WP-M M5's maximum 25% wall-time regression plus its
pinned-runner RSS/no-second-representation gates remain authoritative against the same Exact `2.0`
workflow, with semantic workload differences classified at the existing projector rule level. In
addition:

- one loaded view must feed projector and selected reasoner without another ontology-sized
  representation in Exact;
- native consumer compilation must meet each consumer's own release gates; and
- no performance claim may combine a warm encoded cache on one side with a cold parse on the
  other or omit output generation.

If the core native loader still misses its Horned-equivalence gate, Exact records that separately;
fast consumer compilation cannot conceal slow loading.

## 7. Owned implementation surface

The implementation is expected to make small coordinated changes in:

- `exact/ontology/projection.py`, `reasoning.py`, `provenance.py`, and `_reasoner_worker.py`;
- `exact/ontology/store.py` or `exact/io/sources/owl.py` only where public capability/provenance
  forwarding requires it;
- cache/run-manifest version handling, dependency ranges/extras, focused integration tests,
  benchmark evidence, and user compatibility/performance documentation.

It MUST NOT restore deleted local parser/projector/reasoner rules or edit matching/training
algorithms. Metadata/version changes remain owned by WP-M M5/release integration.

## 8. Acceptance and handoff

WP-N completes when all identity/materialization/copy/digest gates pass in pure/scalar/native
matrices; WP-M performance evidence is rerun and accepted; caches/provenance/docs/ranges are
coordinated; and the exact core, projector, reasoner, OAEI, and Exact revisions tested are
published. It supplies evidence to WP-M M5 rather than independently declaring `2.1.0` released.
