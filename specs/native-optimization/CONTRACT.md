# Behavior and native execution contract

**Normative for N00–N11; specified, not implemented.**

## C1. Equivalent inputs and intended outputs

Compare identical ontology bytes and resolved imports, root/closure selection, overlays and
composite roles, parsing options, projection profile/literal/duplicate/order settings,
reasoner feature policies, source entities and consumer options. Do not shrink inputs,
drop unsupported constructs, disable literals, weaken completeness, change candidate
populations or use private references to make an optimization appear faster.

| Surface | Must remain equivalent |
| --- | --- |
| Parsing/views | Accepted structural content, entity kinds/punning, imports, scopes, signatures, defined fingerprints and anonymous identities |
| Projection | Exact edge content, raw/unique counts where specified, literal rendering, duplicates and the requested canonical/encounter order |
| Annotation/features | Values with datatype/language/IRI identity, exclusions, asserted relations and selected raw features |
| Reasoning | Consistency, satisfiability, entailments, equivalent classes, direct/transitive hierarchies, realization and completeness metadata |
| Lifecycle | Query isolation, immutable owners, transactions, cancellation cleanup, close/fork/mutation protection and invalid-input rejection |
| Artifacts | Published result schema and deterministic semantic payload; compatible cache reuse follows its existing identity contract |

Timing, RSS, cache hits, operation counts, build identity and explicitly diagnostic traces
may change. A faster valid run may finish before a resource deadline that previously stopped
it; it must still honor that deadline and budget. Preserve documented error categories and
semantic diagnostics, not arbitrary internal stack frames or the exact timing of failure.
Byte-for-byte comparison applies to specified canonical representations; unordered sets
and reasoner graphs are compared under their declared equivalence relation. Canonical
blank-node normalization is permitted only where the existing contract already permits it.

Use pinned reference fixtures and documented semantics alongside before/after parity. A
baseline disagreement with the intended semantics is a correctness issue, not permission to
silently change output in an optimization commit. Isolate its reproducer and resolution;
never bless new golden outputs simply because the optimized implementation produced them.

## C2. Native work and the Python boundary

Python may validate bounded configuration, choose an operation, manage a lifetime, receive
small summaries and construct explicitly requested result objects. The native pipeline must
perform ontology-scale parsing, structural validation, indexing, projection, canonical
sorting/deduplication, normalization/clausification, reasoning and bulk graph/result validation.
Python iteration that merely yields requested final result pages is an interface; rebuilding
an ontology, querying by whole-graph Python scans or validating every encoded byte is not.

Introduce the common logical option `require_native_pipeline: bool = False` on existing
projector/reasoner configuration surfaces, keyword-only where applicable. N00 locks the
exact placement and diagnostics vocabulary before cross-repository edits. Reuse existing
configuration objects; do not create a second execution-mode framework. Core retains
`LoadOptions.backend=NATIVE` for parsing and supplies native view/capability contracts.

- Default package behavior remains compatible, including documented non-strict backends.
  With the strict option true, `auto` selects a conforming native path or fails; an explicit
  Python backend is a configuration error. Existing native/Rust backend names remain valid.
- A strict operation requires supported native ingestion, native bulk validation/processing
  and native canonical/result handling for its requested policy. Configuration/backend
  conflicts fail explicitly. Accept no Python scalar-compiler or indexed-byte fallback.
- Negotiate capabilities before consuming ontology-sized input where possible. A path known
  to be incompatible must fail before expensive preprocessing; data-dependent incompatibility
  must fail before publishing partial results. Never swallow a native failure and rerun Python.
- Unsupported owner types either gain a safe retained immutable lease or are explicitly
  rejected in strict mode. A read-only view alone does not prove its backing owner immutable.
- Reusing an owner never requires XML/RDF reparse. Native pipelines may retain compact
  derived indexes and private IR; avoid cloning another complete base per consumer query.
- Release the GIL around safe native bulk work, including applicable validation/serialization.
  GIL release is necessary execution evidence, not proof of multicore work or zero copying.

Strict compliance is per operation and supported input family, not a global library label.
Existing encoded-native ingestion counters alone do not certify Python-free validation or
canonical handling. Until the applicable implementation exists, advertise partial capability
accurately and reject an unsatisfied strict request. Exact opts in at N11 after relevant paths
pass. Do not claim that today's private bridge already enforces this stronger full contract.

## C3. Validation, ownership and reusable state

Core owns structural validation; consumers retain their own semantic checks. Reuse only a
validation result bound to the exact immutable owner/exporters, schema/descriptor, selected
scope/roots, anonymous-scope mapping, revision and relevant options/limits. A caller-supplied
Boolean or a digest detached from owner lifetime is insufficient. Validate external or
changed input in native code. Keep malformed-data, overflow, nesting and resource checks.
Do not duplicate the OWL schema in a new cross-package intermediate representation.

Reuse the current snapshot/session cache infrastructure. Equal ROOT and CLOSURE results
need evidence from the same owner/selection contract; independently allocated row IDs are
never assumed equal. Imports, excluded composite roots and overlay tombstones must not
leak into an optimized selection. Unsupported reuse falls back to correct native work,
not altered scope or Python. A smaller requested budget still applies to cached results.

Reasoner permanent state may be shared immutably. Query-local witnesses, fresh symbols,
polarity/occurrence effects, equality, blocking, role/range consequences and datatype state
must remain isolated. Reusing two structurally similar queries is valid only when all their
semantic contexts match. Eviction changes cost, not answers. Failure and cancellation must
leave the session in its documented usable/closed state, never a partially updated state.

## C4. Observable evidence and bounded memory

Extend existing diagnostics rather than building a tracing service. At minimum distinguish:
input parse/retention; encoded publication; native and Python validation; native preparation;
native generation/reasoning; Python result publication; canonical handling; annotation/index
queries; per-query base/delta construction and retained cache bytes. Mark nested timing
boundaries explicitly. Performance timestamps/counters are not semantic output fields.

Capture actual counts at relevant boundaries: native class-index builds and membership
lookups, scalar object materializations, Python indexed-byte accesses, copied/staged bytes,
ROOT table publications, per-query base builds, retained query-state bytes and worker count.
Unavailable counters are unavailable, not an invented zero. Strict tests must trip forbidden
fallback hooks as well as checking reported capabilities. Do not reject legitimate lazy
materialization of explicitly requested final entities as a scalar OWL compilation fallback.

Account native indexes, temporary buffers, spill, retained owners and cache state against
existing operation/process budgets. Bound batches and caches by bytes where item sizes vary;
keep existing count ceilings as additional guards. Reserve before growth and retain overflow,
cancellation and allocation-failure behavior. No unbounded full closure cache or expanded
canonical-byte cache may be introduced solely to improve a timing.

## C5. Compatibility and deliberately limited scope

Keep current projection profiles and public result semantics. Prefer additive public
capabilities with explicit minimum-version checks, tested with supported/unsupported package
combinations. Existing callers without the strict option retain documented behavior. Do not
silently change global defaults or dependency ranges, or mutate existing experiment bundles.

A Rust reference algorithm may remain test-only for small differential cases. Do not add duplicate production implementations or retain superseded optimization paths;
preserve existing documented backend choices. Do not add per-run dual execution, hard-coded
NCIT shortcuts or an Exact-owned native compiler. A new general API must serve a demonstrated requirement;
optional packed outputs and advanced caches require residual-cost evidence.
