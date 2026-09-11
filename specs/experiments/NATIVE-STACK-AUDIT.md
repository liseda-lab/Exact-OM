# Native ontology stack: efficiency assessment

Date: 2026-09-11. Scope: requested source audit of pyOWLCore, the OWL2Vec* projector,
pyHermiT and pyELK. All four repositories were cloned into the user's home directory and
left unchanged. This is an assessment, not an implementation or a new campaign benchmark.

The stack has genuine native engines, retained ontology ownership and useful coarse native
interfaces. Its main opportunities are repeated work and data representation: a native
annotation lookup scans the whole node table per annotation, consumers repeatedly validate
or reconstruct ontology-sized data, and reasoner queries rebuild data that could be shared.
Keep Python for configuration, orchestration and small result objects; keep ontology-scale
indexing, validation, traversal, compilation and canonical result handling native. Language
choice alone will not repair a quadratic native algorithm.

## Repositories and evidence

| Repository under `/home/pgcotovio` | Reviewed commit | Installed Python source correspondence |
| --- | --- | --- |
| `pyOWLCore` | `d39fe9c9bb9513db8c14fe2bc6d4864377901ad1` | Five inspected hot files match 0.2.0 |
| `pyOwl2Vec-Star-projector` | `d7e4dc147dc7303d75353887352afb66e7840588` | All 17 files match 0.2.0 |
| `pyHermiT` | `c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af` | All 92 files match 0.2.0 |
| `pyELK` | `b7b373165f6cbc3452f022c542731d7b5f2f73be` | All 38 files match 0.2.0 |

Rust findings refer to the pinned source commits; correspondence of Python source does not
prove reproducible-build identity of the installed native binaries. Local detailed audits,
repository identities and the small pyELK diagnostic are retained in
`data/experiments-v2/native-stack-audit/`.

The existing full-NCIT diagnostic measured 2,464.4 seconds of projection, including 314.2
seconds at encoded ingestion, 1,229.1 seconds at preparation and 920.4 seconds at edge
iteration. These are nested boundaries: preparation includes ROOT view publication and
validation; iteration includes generation, Python object publication and canonical handling.
Exclusions separately took 1,025.0 seconds. Neither reasoner ran in that diagnostic.
[Measurement details](NATIVE-PREPROCESSING.md) preserve input and output identities.

Historical scope clarification: revision `20929f0`, `exact/core/entities/ontology.py:181`,
calls `projector.project(self.ontology)`, not the reasoner. A reasoner had been created, but
its presence does not establish that the projector used its inference state. The old timing
also includes out-edge/graph construction before logging at line 187. Different ontology
inputs and phase boundaries prevent a controlled Java/native speed ratio.

## 1. Projector: replace repeated native class-membership scans first

For each supported annotation, `contains_class_iri` scans every encoded node until it finds
an entity with class kind and the exact subject IRI. Counting and emission both execute the
lookup. Worst-case work is O(A * N), where A is eligible annotations and N is encoded nodes.
The composite variant scans its reachable class list per annotation, O(A * C).
This is the strongest localized algorithmic optimization found in the audit.
Sources: [lookup](https://github.com/OAEI-ML/pyOwl2Vec-Star-projector/blob/d7e4dc147dc7303d75353887352afb66e7840588/native/src/encoded_direct.rs#L5261),
[count](https://github.com/OAEI-ML/pyOwl2Vec-Star-projector/blob/d7e4dc147dc7303d75353887352afb66e7840588/native/src/encoded_direct.rs#L7367),
[emission](https://github.com/OAEI-ML/pyOwl2Vec-Star-projector/blob/d7e4dc147dc7303d75353887352afb66e7840588/native/src/encoded_direct.rs#L10772).

Build one native membership index during preparation and reuse it for counting and emission.
A set offers expected O(N + A) work; a sorted vector gives O(N + C log C + A log C).
Preserve exact IRI spelling, class punning, the annotation-property whitelist (including
NCIT synonyms), ROOT annotation provenance, literal rendering and composite reachable-class
selection. Use the index only for membership so output ordering stays unchanged. Charge its
memory to the existing limits. No speedup factor or fraction of the 41-minute projection
has yet been isolated for this function.

Two further projector opportunities:

- **ROOT reuse:** with literals enabled, it acquires another full encoded ROOT view, creates
  and discards a compiler, then compares buffers with CLOSURE. The measured one-document
  NCIT input ultimately uses equal buffers. Prove scope equality before preparing another
  table, or expose ROOT annotation postings over retained closure data. Imported and
  transformed views require their actual selection; table-local row IDs are not interchangeable.
  [Source](https://github.com/OAEI-ML/pyOwl2Vec-Star-projector/blob/d7e4dc147dc7303d75353887352afb66e7840588/src/pyowl2vec_star_projector/native.py#L1608).
- **Canonical output:** native batches become Python Edge/string objects, are sorted and
  spilled even when only one small chunk exists, then reconstructed. Keep compact edge IDs
  through native sorting/deduplication and expose final batches or a canonical artifact;
  add a bounded in-memory path. Current NCIT wrote only about 90 MB in three runs with no
  intermediate merge passes; these counts do not establish how much of the 920 seconds
  canonical handling consumed.
  [Source](https://github.com/OAEI-ML/pyOwl2Vec-Star-projector/blob/d7e4dc147dc7303d75353887352afb66e7840588/src/pyowl2vec_star_projector/streaming.py#L675).

## 2. pyOWLCore: stop reconstructing native data in Python

**Repeated validation and canonical reconstruction.** Even trusted retained native buffers
enter `_validate_columns`, which loops over the column structure and reconstructs canonical
bytes for every node/root in Python. Expanded nested encodings can exceed compact DAG size.
The projector then repeats reference checks in Python before its native semantic checks.
Move structural validation and canonical stream checks into a native pass, and reuse a
validated-view receipt tied to the exact immutable owner, schema, scope, buffers and limits.
Keep full validation for external or changed inputs; a caller-provided trusted flag is not
sufficient. Sources: [unconditional validation](https://github.com/OAEI-ML/pyOWLCore/blob/d39fe9c9bb9513db8c14fe2bc6d4864377901ad1/src/pyowl_core/backends/native_views.py#L1933),
[canonical reconstruction](https://github.com/OAEI-ML/pyOWLCore/blob/d39fe9c9bb9513db8c14fe2bc6d4864377901ad1/src/pyowl_core/backends/native_views.py#L2381).

**Native annotation queries.** Exact's exclusion preparation decodes all annotation
assertions into Python although its final policy tests two properties. Scalar native rows
are decoded, re-encoded for canonical validation, recursively sized and admitted into a
small object cache; eviction triggers rebuilding the nonempty OrderedDict. This is a poor match for a
millions-row scan. Expose native property/subject filtering and return compact selected rows;
keep exclusion policy in Exact and materialize Python values only when requested. This
addresses the path behind the measured 17-minute phase, whose individual costs remain
unseparated. Sources: [Exact annotation cache](../../exact/ontology/store.py#L473),
[scalar cache](https://github.com/OAEI-ML/pyOWLCore/blob/d39fe9c9bb9513db8c14fe2bc6d4864377901ad1/src/pyowl_core/document/native_storage.py#L395).

**Typed indexes and repeated scans.** Generic typed `iter_axioms` decodes all axioms before
filtering. Upstream class, property and domain/range builders repeat this three, five and
six times respectively. Annotation indexing also gathers all roots even when nested
annotations were not requested. Route these through native constructor partitions and
build numeric adjacency/indexes there. Exact already avoids the general class builder;
other lazy views were not separately timed. Typed index metadata also expands postings and
sizes into millions of Python integers and performs per-row budget accounting: use compact
arrays/ranges and native aggregate validation instead.
Sources: [typed filtering](https://github.com/OAEI-ML/pyOWLCore/blob/d39fe9c9bb9513db8c14fe2bc6d4864377901ad1/src/pyowl_core/document/native_storage.py#L2217),
[annotation roots](https://github.com/OAEI-ML/pyOWLCore/blob/d39fe9c9bb9513db8c14fe2bc6d4864377901ad1/src/pyowl_core/index/annotations.py#L211).

Native canonical sorting also compares structural streams byte by byte, repeatedly, with
cost depending on common-prefix length. Instrument its comparison-work counters before
choosing rank/prefix acceleration. Preserve exact canonical order; sorting by hashes would
change the contract. This is a secondary candidate, not an isolated measured bottleneck.

## 3. pyELK: reuse query state and demand only necessary reasoning

On a class-query result-cache miss, the native session requests base classification and
realization even for a satisfiability check. These base results are computed once and then
reused/cloned. For each distinct encoded class expression in a consistent session, it also
constructs a full remapped ontology overlay and property closure and retains them in an
unbounded query-evaluation map. Existing named-class queries have no bypass on this path.
Repeated identical queries reuse results; different query kinds share their expression
state. Inconsistent and unindexed queries skip overlay construction.
Sources: [session dispatch/cache](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-core/src/session.rs#L196),
[full overlay](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-core/src/query.rs#L28),
[query preparation](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-core/src/query.rs#L450).

Use cached native taxonomy/context lookup for named classes. For complex expressions,
share the immutable compiled base and build only a query delta, with a byte-bounded state
cache. Dispatch by query task so satisfiability does not force realization. This can remove
O(Q * ontology-size) retained base copies for Q distinct expressions. It does not imply that
the entire overlay is classified per query: satisfiability saturates its own root only.
Exact's inferred hierarchy adapter uses the named-query interface, so this matters when
inferred features are enabled.

A bounded installed-package check on an eight-class synthetic chain took 0.1096 seconds.
The first `is_satisfiable(C0)` changed diagnostics from no taxonomy/realization to both
cached; distinct C1 increased retained class-query count from one to two; repeated C0 kept
two. Parsing and ingestion were native, with compiler GIL release. This verifies dispatch
and reuse, not large-ontology throughput or memory savings.

Additional native opportunities: query saturation rebuilds ontology-wide rule indexes for
each demanded root although classification already has reusable prepared workspaces;
selection recomputes all-node superclass closure; realization enumerates every individual
pair. Reuse prepared indexes and immutable adjacency, demand local reachability, and find
individual equivalences through indexed entailed edges. Worst-case transitive closure is
quadratic in space, so caching every closure unconditionally is not a solution.
Sources: [query saturation](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-core/src/query.rs#L480),
[individual pairs](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-core/src/taxonomy.rs#L147).

## 4. pyHermiT: retain the permanent program across semantic tests

**Classification query copies.** Native classification builds a batch of counterexample
queries, each copying full symbol/predicate and role/datatype models. Each test then combines
permanent and query data, rebuilds rule state, and checkpoints by cloning permanent engine
state. A batch can hold O(batch-size * base-model-size) copies before execution. Share an
immutable permanent program and indexes with small local witness deltas; use byte-bounded
batches and rollback trails or copy-on-write state. Equality, nominal introduction,
datatypes, blocking and failure rollback make this a larger change than a lookup index.
Sources: [batch/model copies](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/native/src/classification_bridge.rs#L376),
[checkpoint and rule reload](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/native/src/native_tableau.rs#L122).

**Generic query session rebuilds.** For complex queries outside named-domain shortcuts,
a Boolean-cache miss constructs a core overlay and recompiles a fresh native session.
The shared input avoids source parsing but does not avoid reasoner compilation. Introduce
a native query-delta compiler over the permanent program, retaining full rebuilds when
required by global constraints. Unlike pyELK's evaluation cache, this Boolean cache is
bounded at 4,096 entries by default; do not describe it as retaining unbounded whole
ontologies.
[Source](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/src/pyhermit/facade.py#L927).

**Hierarchy adjacency.** `MutableHierarchy.boundary` scans every edge to find neighbors of
each visited node. Maintain parent/child adjacency alongside the canonical edge set so
neighbor discovery is proportional to degree. This is a localized native algorithm change
with a clearer correctness boundary than reworking tableau reuse.
[Source](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/native/src/services/classification.rs#L848).

**Updates and Python metadata.** Any nonempty encoded flush replaces the session and clears
precomputation. Coalesce edits now; implement incremental encoded compilation only for
proven eligible changes. Python also builds complete origin/provenance tuples and decodes
full public symbol JSON during session setup; native sidecars and lazy result objects can
avoid this materialization while preserving profile-error provenance.
[Source](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/src/pyhermit/facade.py#L972).

## Native execution must describe the whole path

Both reasoners can accept encoded owners that cannot provide recognized retained byte
slices. Their native ByteSource then calls Python memoryview indexing per byte while
holding the GIL. Normal recognized packed-byte owners use the genuine detached path;
this finding does not say the measured NCIT projection took the fallback. Sources:
[pyHermiT byte access](https://github.com/OAEI-ML/pyHermiT/blob/c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af/native/src/lib.rs#L100),
[pyELK byte access](https://github.com/OAEI-ML/pyELK/blob/b7b373165f6cbc3452f022c542731d7b5f2f73be/rust/pyelk-pyo3/src/lib.rs#L69).

Likewise, selecting a native inference backend does not always require native compilation:
capability absence can lead to Python scalar compilation feeding that backend. Provide a
public strict encoded-native execution contract that rejects incompatible owners before
expensive work, or support them with safe retained immutable buffer leases. Record actual
scalar/byte-access counters and detached execution; constant zero counters alone do not
prove absence of callbacks. Preserve lifetime, mutation and malformed-input checks.

The projector similarly permits scalar-native compilation under its public native backend
option. Exact's strict adapter rejects that fallback, and the recorded NCIT run confirmed
encoded-native execution. A public upstream requirement would remove the need for that
consumer-specific bridge.

Python also reconstructs and validates bulk taxonomy results; some native result serialization
runs after reacquiring the GIL. Return retained numeric views or packed batches and create
Python entities on demand. Bounded configuration checks and presentation can remain Python.

## Implementation order and acceptance

1. **Localized algorithm fixes:** projector class-membership index; pyHermiT hierarchy
   adjacency; avoid unnecessary nested-annotation root collection. Use focused differential
   fixtures and operation counts before large timings.
2. **Current Exact preprocessing:** shared core/projector native validation and ROOT reuse;
   native filtered annotation queries; compact typed-index metadata. Keep these as small
   coordinated changes with explicit owner/schema/scope contracts.
3. **Reasoner query reuse:** pyELK named-query dispatch and byte-bounded overlays;
   pyHermiT immutable permanent program/query deltas. Validate each reasoner's completeness,
   equality, datatype, range-isolation and rollback contracts separately.
4. **Residual measured costs:** split native generation, Python publication and canonical
   handling, then choose compact native output, in-memory sorting and canonical-order
   acceleration. Improve concurrency after repeated work and memory amplification are removed.

On this 8-core/16-thread, roughly 62-GiB machine, pyELK already has actual Rayon parallelism;
setting more Python workers cannot parallelize GIL-held validation. Budget total worker
threads across concurrent processes and measure 1/2/4/8 native workers on a bounded fixture.
The NCIT process peaked near 11.66 GiB; do not multiply full ontology processes to fill every
logical CPU. Use local scratch for spill and preserve completed artifacts for reuse, but
CPU/wall agreement makes disk placement a secondary explanation for the observed delay.

Acceptance should cover identical ontology/import bytes, feature settings and source
population; exact edge digest, exclusions and representative hierarchy/query answers;
malformed buffers, scope/punning/literal semantics, limits and cancellation. Record stage
wall/CPU time, peak RSS, copy/byte-access counts, query cache bytes and parallel worker counts.
Use public/generated fixtures and ontology structure only; private test references have no
role in tuning these changes. Reasoning complexity itself can remain large after overhead
is removed, especially for expressive OWL; no fixed latency guarantee follows from Rust.

No package was rebuilt, patched or installed, and no long experiment was launched. This
review used source inspection plus the tiny pyELK diagnostic. The four packages cover the
ontology substrate and reasoners; Exact still owns graph/feature assembly, model encoding,
retrieval and experiment artifact reuse, which remain separate optimization surfaces.
