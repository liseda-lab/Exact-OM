# Validation, resources and implementation evidence

Status: specified only. These are future implementation gates; writing this plan starts
no builds, benchmarks, G0 validation or long campaign. [CONTRACT.md](CONTRACT.md) governs
semantic equivalence. Reuse current package tests and Exact's existing preprocessing tool;
add focused fixtures/counters rather than a new test or execution framework.

## N00: Freeze a reviewable baseline before code changes

Record the audited commits from [the source assessment](../experiments/NATIVE-STACK-AUDIT.md),
Exact commit, installed package/native-binary identities, compiler/build mode, interpreter,
encoded/model schemas, relevant options and machine limits. Build an isolated native baseline
from the pinned sources when a source-level performance comparison needs it; do not assume
an installed extension was built from the checkout just because Python sources match.
Do not rebuild a large ontology merely to recreate already trustworthy evidence.

Keep `native-preprocessing-03` immutable. It is a measured installed-package reference:

| Saved NCIT field | Baseline |
| --- | --- |
| Input SHA-256 | `1a7182a7327ebc4181f7d6b0f7e81ed04dd258f1a86bd8f560e4a0d61439d58a` |
| Effective axioms / classes | 3,506,377 / 211,958 |
| Projection edges | 630,404 |
| Canonical edge SHA-256 | `370ef2ea5f7c8d4ba53dee63b4e371b27f7d1e6321bf63f14599925242ff6244` |
| Projection / exclusions | 2,464.4 s / 1,025.0 s |
| Total / peak RSS | 3,999.5 s / 11.66 GiB |

Load exact options, import manifest, frozen 64 source IRIs and remaining digests from saved
artifacts, not assumptions based on this summary. Annotation, hierarchy and feature outputs
must also be compared; an edge digest alone does not cover them. Pin upstream oracle versions
and their declared feature scopes. Legacy mOWL subset timings are context, not this baseline.

N00 also locks the minimal public strict-option placement, native validation/selection
capability vocabulary and counter meanings from CONTRACT. Document ownership/lifetime and
package compatibility before parallel consumer changes. Begin with direct immutable snapshots;
list separately supported transformed and mapped owners instead of claiming blanket support.

## T1: Correctness and native-path fixtures

Every changed operation passes applicable existing native, differential, lifecycle and
malformed-input suites plus targeted cases below. A Python reference may run as a small
**test oracle**; strict production paths must not call it. Keep reference helpers test-only.

| Fixture family | Required observations |
| --- | --- |
| Direct/imported/ROOT/closure, overlays and composites | Exact selected structural content, exclusions, provenance and anonymous scopes |
| Punning, absent entities, label/synonym whitelists, literal variants | Exact annotation membership, values, multiplicity and order |
| Empty, chain, diamond, multiple inheritance and equivalence graphs | Canonical projection, direct/transitive hierarchy and complete results |
| Fresh/complex queries, inconsistent inputs, role/range interactions | Same answers/completeness under different query orders and cache states |
| Nominals, equality, disjunction, cardinality and datatypes | HermiT query/batch correctness and complete rollback |
| Forged/mutated owners, invalid closed-handle operations, invalid columns and unsupported schemas | Explicit rejection before partial result publication; no hidden scalar work |
| Tiny memory/page/spill budgets, allocation/disk faults and cancellation | Bounded resources, correct error class, clean or explicitly invalidated state |

Extend the existing suites under core `tests/differential` and `tests/unit/indexes`, projector
native differential/encoded/streaming tests, pyELK `tests/backends` and `tests/parity`, and
pyHermiT `tests/native`, `tests/differential` and `tests/integration/classification`.
Run each owning repository's normal relevant Rust/Python checks; no obsolete performance
thresholds from another migration become acceptance requirements here.

Require query-order and batch-partition independence: fresh A, A→B→A, B→A, eviction→A and
cancellation→A agree when the lifecycle contract permits the final query. Incremental cases
compare with full native rebuild after every committed generation. Preserve exact documented
ordering; normalize only sets/graphs whose public contract permits normalization.

Native-path tests instrument forbidden scalar compiler, canonical decoder and indexed-byte
fallback hooks, not just a returned backend label. Verify actual counters and GIL-release
capability where bulk work is expected. Exercise both strict and unchanged default behavior.

## T2: Demonstrate reduced work on bounded generated inputs

Use deterministic fixtures that vary the suspected cost dimensions independently. Assert
operation counts from the work-package specs; do not write tests that merely mirror the
new implementation or assert flaky sub-millisecond speed ratios.

- N01: vary encoded-node count and eligible annotations; no whole-node scan per membership.
- N02/N03: cold publication versus compatible reuse; no Python column reconstruction and
  no second ROOT table/compiler when the scope-equivalence proof applies.
- N04: grow unrelated annotations/axioms; selected results crossing Python and warm query
  work stay bounded by requested results, with native index construction charged separately.
- N05/N07: grow base size, distinct queries and batches separately; retain one compatible
  base, bounded local state and no duplicate complete program per eligible query.
- N06: vary graph size and degree; count enumerated neighbors separately from oracle calls.
- N09: vary total edge bytes, duplicates and spill boundaries; native canonical parity,
  bounded working storage and no Python sorting/heap traversal on the strict path.
- N08/N10 when selected: declare the exact update/phase workload and expected removed work
  before measuring; preserve the complete native path for cases outside the optimization.

For timing, a default small-case comparison uses one warm-up and five paired samples,
interleaving baseline/candidate order. Report medians, spread, CPU/wall and peak/retained
bytes; record cold setup separately from warm reuse. Keep each diagnostic invocation within
60 seconds by selecting bounded fixture sizes in advance. A timeout is incomplete evidence;
do not shrink only the candidate's input or repeatedly restart an unchanged timed-out job.

Reduced work plus semantic parity is required to accept an optimization mechanism. A claimed
runtime/memory benefit also needs matched measurements above noise and an explanation of
tradeoffs. A change that only moves cost between phases is not an end-to-end improvement.
No fixed speedup is a substitute for these checks. Conditional N08/N10 or richer packed
interfaces proceed only when a recurring residual cost and a bounded remedy are documented.

## T3: Installed artifacts and cross-package integration

Build release-mode candidate wheels using existing package workflows, with recorded hashes.
Use an isolated test environment with a coherent core/projector/reasoner set. Run relevant
installed-package smoke, schema/owner compatibility and normal repository quality checks.
Do not test new Python wrappers against an unnoticed old native extension. Retain the baseline
environment; installation/publishing into the active experiment stack is a separate step.

Test supported matching versions, absent capability, incompatible schema, unavailable native
extension and supported/unsupported owner types. Strict mode rejects incompatible combinations;
non-strict defaults remain compatible. Exact tests cover public projection, labels/exclusions,
asserted and supported inferred hierarchy, fingerprints/cache identity and owner retention.
The private bridge is removed only after its public replacement passes these checks.

## T4: One bounded real-input integration measurement

After T1–T3 pass for the relevant packages, measure the saved NCIT–DOID ontology inputs with
`tools/benchmark_native_preprocessing.py`, the frozen source set and the existing stage profile.
The first measurement uses the declared canonical/literal options and unchanged import closure;
verify signature, projection, exclusion and selected raw-feature digests. Keep source and target
in separate fresh processes with unique output paths. Reuse trustworthy completed outputs;
do not repeat full baselines after every small commit.

Use the existing detached launcher/job-record mechanism. Predeclare a total deadline before
launch: 120 minutes for the full NCIT diagnostic provides headroom over the observed 67 minutes;
30 minutes for the DOID diagnostic. These are ceilings, not predicted runtimes. Persist phase
progress, actual worker exit and result status. Preserve a timed-out/failed attempt and inspect
its last completed boundary before any extension/retry; never discard evidence and start over
blindly. Use stage timers and fatal-only trace capture in the current Python 3.12.3 environment,
not the periodic traceback watchdog that previously crashed.

This tool is model/reference-free and uses asserted hierarchy. It does not validate pyELK or
pyHermiT at NCIT scale. Reasoner parity comes from T1/T3 and bounded T2 workloads; any later
real-ontology reasoning measurement needs its own selected task set and resource declaration.
Do not automatically add full FMA/SNOMED classification, GPU/LLM work or a corpus cross-product.

A diagnostic that exceeds the short interactive bound is detached with its session name,
artifact/log paths and a single monitoring command handed to the user. Do not token-poll a
long-running job. Nothing in these gates starts the long scientific experiment campaign;
G0 admission and budget/recovery decisions remain governed by the experiment specs.

## Machine budget and evidence record

On the current 8-core/16-thread, roughly 62-GiB host, budget runnable native/build workers
across all tasks. Start with two compiler jobs per build and no more than two concurrent
builds. During a real ontology measurement, pause competing builds/benchmarks and run one
heavy process; do not duplicate full ontologies to occupy every logical CPU. Small source
reviews and low-memory fixture tests can proceed independently.

For pyELK scaling, compare 1/2/4/8 native workers on bounded cases; account aggregate workers
when multiple tests run. Keep at least 12 GiB of host memory headroom and honor any stricter
scheduler/cgroup cap. A 48-GiB per-heavy-job ceiling is an initial upper bound, not memory
that a job should consume. Charge live owners/caches and scratch usage; use local `/tmp`
scratch when appropriate, with explicit available-space checks and bounded spill cleanup.

Keep one concise per-change record beside existing measurement artifacts: work ID; package
commits and wheel hashes; inputs/options/queries; completed tests; semantic comparison result;
actual native/scalar/copy/work counters; cold/warm timing spread; peak/cache/spill bytes;
worker counts; limitations; and final status. Do not add an evidence database or parallel runner.
Use statuses `specified`, `in_progress`, `validated`, `blocked` or `deferred_with_reason`;
only `validated` means the work's required gates passed. Existing failed attempts, charges,
scientific selections and private-label boundaries remain unchanged.
