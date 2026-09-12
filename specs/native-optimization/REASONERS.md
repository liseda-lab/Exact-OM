# Reasoner optimization specifications

Status: **proposed; not implemented or benchmark-qualified**. Scope: N05–N08 only.
These changes preserve public reasoning behavior under [CONTRACT.md](CONTRACT.md).
Acceptance follows [VALIDATION.md](VALIDATION.md). The [source audit](../experiments/NATIVE-STACK-AUDIT.md) supplies the inspected baseline:
pyELK `b7b373165f6cbc3452f022c542731d7b5f2f73be`;
pyHermiT `c4bdbbc7581281cd46803a5c0eafd1f8a1f3c2af`.

## Shared boundaries

- Preserve package backend defaults, supported profiles, inference semantics and public APIs.
  N02 proposes `require_native_pipeline: bool = False`; N11 enables it in Exact only after
  used paths conform. Known unsupported combinations fail before expensive work.
- N02 owns rejection of per-byte Python buffer access and scalar compilation fallback.
  N05–N08 must use its retained-owner, schema, scope and detached-execution guarantees.
- N05/N07 consumer conformance moves ontology-wide provenance, symbol and bulk-result
  validation into native code. Python may validate bounded arguments and construct requested
  result objects; Rust calling Python through ByteSource is prohibited under strict execution.
- Reuse existing ownership, caches, cancellation, diagnostics and tests; add no cache service,
  persistence format or reasoner framework.
- Cache keys include the effective ontology/import identity, semantic configuration, query
  task/polarity and relevant scope. A canonical expression alone is insufficient.
- Retain exact ordered/canonical outputs where promised; otherwise compare canonical sets.
  Preserve completeness indicators, fresh-entity handling and public error categories.
  Resource policy remains explicit; timing and internal work counters may improve.
- Share immutable native data only. A session's mutable query state cannot cross requests,
  generations, threads or forks contrary to its existing concurrency contract.
- Large timing studies are optional; required bounded acceptance gates still apply. Unit
  tests do not authorize larger ontologies, more workers, wider support or campaign execution.

## N05 — pyELK task-aware queries and shared complex-query state

**Dependencies:** N02 for strict integration; N05a can precede N05b.
**Primary code:** `rust/pyelk-core/src/session.rs:196`, `query.rs:28,450,480`.

The current query path demands base classification and realization on cache misses; these
base results are computed once and then reused/cloned. Distinct expressions can retain
full ontology overlays. This specification does not claim base reclassification per query
or full overlay classification for each satisfiability query.

### N05a: named-query fast path and task dispatch

1. Define a native dispatch table for satisfiability, equivalence, superclass/subclass and
   instance tasks, including positive/negative polarity, directness and result completeness.
2. For an indexed named class, reuse established native taxonomy/context data without
   building a query overlay, remapping base symbols or preparing another property closure.
   Preserve the current canonical result and treatment of top, bottom and equivalent nodes.
3. Demand only semantically necessary work. Satisfiability and hierarchy-only tasks must
   not initialize realization merely because they share a class-expression entry point.
   Necessary consistency checks remain; instance queries may legitimately need realization.
4. Respect the existing fresh/unindexed entity policy and inconsistent-ontology behavior.
   Do not manufacture a complete answer from an absent index entry.
5. Keep range/domain constraints and saturation context separate where the current algorithm
   distinguishes them. A named shortcut must not reuse an incompatible polarity/context.

### N05b: complex queries without retained full-base copies

1. Retain one immutable compiled base, rule indexes and property-closure representation per
   ontology generation. Represent each complex expression with local nodes/rules and an
   isolated saturation context; reuse prepared workspaces only when their contract permits.
2. Do not rebuild global rule indexes or copy the full base for every distinct expression.
   Compute demanded reachability locally; do not substitute an unconditional all-pairs
   closure cache, which can itself require quadratic memory.
3. Bound the existing expression-state cache by retained bytes, accounting shared storage
   once and query-owned allocations individually. Keep compatible cross-task reuse.
   Configure the bound through existing resource policy or an explicit additive option;
   document the value and eviction order without changing inference/backend defaults.
4. Eviction releases query-owned state and cannot invalidate live result owners or the base.
   An entry larger than the cache budget runs uncached within the operation's resource limit;
   cache admission alone must not turn a supported query into a new semantic failure.
5. Preserve the existing rejection/completeness policy for unsupported EL constructs.
   Query-local range facts, fresh symbols and witnesses must not contaminate later queries.

### Acceptance

- Differential fixtures cover named/complex classes; polarity; top/bottom/equivalence;
  direct/indirect answers; role chains; domain/range interactions; fresh/unindexed entities;
  inconsistency; unsupported constructs and completeness/error metadata.
- Query-order metamorphic checks compare A→B→A, B→A, repeated A, cache eviction and a fresh
  session; mix satisfiability, hierarchy and instance tasks for the same expression.
- Named queries allocate zero full-base copies, overlays or query-specific property closures;
  report permitted initial taxonomy/consistency work separately.
  A hierarchy-only sequence leaves realization uninitialized unless semantically required.
- Track base preparation count, overlay/base-copy bytes, query-owned/cache bytes, cache hits,
  evictions and saturation work. Many distinct complex queries respect the configured cache
  bound; results before/after eviction equal fresh-session results.
- The performance stage compares fixed tasks and inputs, including cold and repeated calls;
  it must not count avoided realization as a speedup when the compared task requires it.

## N06 — pyHermiT hierarchy adjacency

**Dependencies:** independent local algorithm change; N02 for strict end-to-end validation.
**Primary code:** `native/src/services/classification.rs:820–917`.

1. Maintain native parent and child adjacency alongside the canonical edge representation.
   `MutableHierarchy.boundary` must obtain neighbors from adjacency, not filter all edges
   for every visited node. Preserve deterministic traversal and representative selection.
2. Update both directions and the canonical edge set atomically on insertion, rewiring and
   equivalence handling. An invariant check must detect inconsistent representations.
3. Preserve bottom/top membership, SCC/equivalence grouping, direct transitive reduction and
   semantic oracle decisions. Do not remove expensive correctness checks as an optimization.
4. Keep allocation and cancellation accounting for the additional adjacency storage.
   Use existing bounded collections; do not add an unbounded reachability matrix.

### Acceptance

- Compare exact canonical node memberships and edges with the prior implementation on
  chains, stars, diamonds, multiple inheritance, disconnected classes, equivalences and
  inconsistent inputs; cover class, object-property and data-property classification.
- Vary input/insertion order and repeat classification in fresh and cached sessions.
  All public results and completeness/error behavior remain equivalent.
- Instrument neighbor enumeration: no whole-edge scan per visited boundary node, and work
  is proportional to enumerated adjacency plus existing search/oracle work.
  Report semantic-test counts separately so a changed search cost is not a changed answer.
- Optional bounded scale measurements must demonstrate the mechanism with operation counts
  before claiming wall-time improvement; N06 does not alter the tableau algorithm.

## N07 — pyHermiT shared permanent program and isolated native query deltas

**Dependencies:** N02; N06 is compatible but not required.
**Primary code:** `native/src/classification_bridge.rs:376–418,593`;
`native/src/native_tableau.rs:122–174,490–560`;
`src/pyhermit/services/checks.py:306–331`; `facade.py:927–956`.

Current classification batches clone base domains/models and reload rule state per test.
Generic encoded queries outside named shortcuts construct fresh sessions on Boolean-cache
misses. The existing Boolean-result cache is bounded at 4,096 entries by default; preserve
its policy rather than describing or replacing it as an unbounded ontology cache.

1. Retain the native permanent program, symbol/predicate domains, role/datatype models,
   compiled join plans and immutable indexes once. Local witnesses and query clauses refer
   to that base without cloning its ontology-sized vectors into each counterexample.
2. Compile eligible generic encoded query deltas against this retained program. Route
   multiple eligible queries through the existing native batch interface, preserving order.
   Named-domain shortcuts continue to use cached coarse results.
3. State a query eligibility predicate before coding. Queries changing global role/schema
   constraints, compiler strategy or unsupported constructs retain the existing complete
   native rebuild. Rebuild reasons must be observable; never drop query axioms to qualify.
4. Replace deep permanent-state snapshots with existing trail machinery or bounded
   copy-on-write checkpoints whose rollback contract covers every mutable component.
   Share immutable plans, not a nondeterministic model as a universally valid consequence set.
5. Restore equality merges, nominal introduction, dependency sets, learned branches,
   existential queues, blocking, datatype constraints and rule indexes after each query.
   Cancellation, timeout, resource exhaustion and contained failure must restore a usable
   prior state or preserve the existing explicit poisoned/disposed-session behavior.
6. Bound batches by estimated live bytes as well as existing count limits. Construct
   query-local state incrementally; do not accumulate B complete bases before B tests.
   Do not parallelize mutation of one session to conceal repeated setup costs.
7. Key retained results by immutable base generation and complete query identity. Preserve
   fresh-entity policy, anonymous scopes, symbol namespaces, query polarity and datatype
   literal identity. Cross-query sharing must not change witness interpretation.

### Acceptance

- Differential answers against a fresh full native rebuild cover deterministic/disjunctive
  programs, nominals, equality/inequality, cardinality, role chains, datatypes and complex
  reductions. Retain existing Python/native differential fixtures as an additional oracle.
- Compare sequential versus batched calls, A→B→A versus fresh sessions, duplicate queries,
  batch partitions and changed query order; include satisfiable/unsatisfiable mixtures.
- Inject cancellation/failure after installation, merging, branching and datatype work.
  Verify rollback invariants and subsequent answers; never publish partial cache results.
- Counters separate base compilation/index construction, base-copy bytes, local-delta
  bytes, checkpoint/rollback bytes, batch peak bytes and semantic-test work.
  Eligible queries perform zero full-base compilation, domain/model copy or whole-base
  rule-index rebuild after setup; classify required complete rebuilds separately.
- Memory bounds include live checkpoints and result owners. Optional timings use identical
  query populations and cache conditions; no speed target weakens reasoning completeness.

## N08 — conditional pyHermiT encoded incremental updates

**Dependencies:** N02 and the permanent-state/rollback contracts established by N07.
**Primary code:** `src/pyhermit/facade.py:958–1045`.
Conditional scope: defer N08 without a proven useful envelope; retain full native rebuild.

1. Specify a narrow initial envelope: no-op changes, then only explicitly proven additions
   of ground positive named-class assertions over existing symbols. Logical additions are
   eligible only when the permanent program and native update machinery satisfy the
   documented strategy/global-constraint restrictions; assertion shape alone is not proof.
2. Removals, new symbols, TBox/RBox edits, equality/nominal/datatype changes and every
   unproven interaction use full native reconstruction. Record the reason; do not silently
   widen eligibility. A later envelope extension requires separate semantic acceptance.
3. Apply eligible deltas transactionally, with exact generation/fingerprint updates and
   compatible generated-symbol namespaces. Publish the new state only after validation.
   Failures leave the previous committed state intact or explicitly unusable as contracted.
4. Invalidate affected query/taxonomy/realization results. Initially clear all derived
   results after a logical edit unless narrower dependency invalidation is proven.
   Retaining the compiled base does not justify retaining stale logical answers.
5. Coalesce buffered edits without changing public flush semantics or import/scope policy.
   Reuse the existing update outcome/cache infrastructure and resource controls.

### Acceptance

- For each eligible update sequence, compare consistency, taxonomy, realization and query
  answers with a fresh native rebuild after every committed generation.
- Test add→query→add, duplicate/no-op edits, buffer/flush grouping, reordered independent
  additions, cancellation, failure and unsupported removals/global edits.
- Counters prove retained base/index reuse for eligible cases and an explicit native rebuild
  for all other cases. Verify result-cache invalidation and ownership after disposal.
- An optional performance gate may justify proceeding with this package; it cannot widen
  the envelope, change package defaults, enable unsupported reasoning or admit experiments.
