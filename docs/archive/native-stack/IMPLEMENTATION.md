# Native stack implementation record

Status: **validated**, 2026-09-14. Implementation and applicable T1–T4 gates pass.
NCIT and DOID each pass exact comparison of 64 ordered feature rows under the documented
deterministic feature policy. The long scientific campaign and G0 have not been started.

## Scope and current gates

Native code lives in the four home clones on `perf/native-pipeline`; Exact uses their
public APIs. The baseline Exact environment remains unchanged. Candidate release wheels
are tested in `/tmp/exact-native-candidate`; its own four ontology packages must precede
shared, unchanged research dependencies. These are local candidates with upstream `0.2.0`
version numbers, **not published releases**. Versions alone cannot identify this stack.

| Work | State | Implemented mechanism / remaining evidence |
| --- | --- | --- |
| N00 | validated | Baseline commits and immutable NCIT/DOID artifacts retained |
| N01 | validated | Native exact-IRI membership index, one per selected projection |
| N02 | validated | Core receipts, consumer preflight and native metadata/results; final coherent installed-wheel checks pass |
| N03 | validated | Proven one-document ROOT reuse; general scope selection remains explicit |
| N04 | validated | Native annotation postings, class/property features, selected typed rows and domain/range indexes; installed Exact parity |
| N05 | validated | ELK task-aware stages, shared query base, bounded LRU and native taxonomy adjacency |
| N06 | validated | HermiT native parent/child adjacency with degree-proportional enumeration |
| N07 | validated | Native assertion deltas share immutable rules, joins, role/datatype/blocking state; isolated bounded batches and rebuild parity |
| N08 | deferred_with_reason | Exact's current static hierarchy adapters make no recurring committed update calls |
| N09 | validated | Native canonical sort/dedup/spill/merge with bounded working storage |
| N10 | deferred_with_reason | Residual loading/projection costs measured; no specific additional canonical/compiler pass selected |
| N11 | validated | Final installed integration and NCIT–DOID T4 pass, including 128 exact ordered feature rows |

`validated` above applies to the package mechanism and its completed T1/T2/installed
fixtures. N11 includes the completed integrated gates below. No general speedup is claimed
from statuses or operation counts. T4 preprocessing does not pass
G0, admit an experiment block, or establish large-ontology reasoner performance.

## Final installed candidates

| Repository | Tested source | Wheel SHA-256 |
| --- | --- | --- |
| pyOWLCore | `5fd93c8` | `f66c068f3a88d0ae59553103fefef4994c43ecca1282300b4dd2486bcbd905d9` |
| projector | `a067601` | `53c752e345d8c18ac1784000609902118b724d61fd8b59d7a79e5cb693d1e0fd` |
| pyELK | `bce95f8` | `3ae4a551d2288b0d703c43edc116b9d5a458f663a36812ac19801f1595b4caef` |
| pyHermiT | `bbf31c2` | `0bf0121e856afd9954bc32ef7358c3c28ec9b0de7ca9fe01eaeff8c5331fffa5` |

Exact adoption is committed in `637f7c8`, the scale fixture update in `542af42`, stricter
counter gates in `a34f08e`, and the separate published-baseline/candidate manifest in
`76e5659`. Final artifact bindings and gate status are in `release/core-compatibility.json`.
The candidate status is `validated`: artifact bindings, installed T3 and real-input T4 pass.

The tested runtime is CPython 3.12.3 on Linux x86_64, with core model/encoded schema 2,
wire 1.2 and the unchanged encoded descriptor
`c51d0eb7ecf6f29ad3495fe7c40a2ea6741cf03a7cf194d51417bb810df90f51`.
No other interpreter/platform matrix is claimed. All four installed modules and native
binaries are identified in `candidate-installed.json`; strict admission and native index
probes pass in `capabilities.json`. Retained wheels support reconstructing this local stack
without replacing the original `.venv`.

## Compatibility and correctness boundaries

Upstream strict options default to `False`; existing non-strict APIs remain available.
Exact requests strict native execution and verifies actual admission/result counters.
Missing binary/capability and unsupported owner shapes fail explicitly without Python
compilation, graph traversal or canonical-output fallback. Python may convert the bounded
requested result to Exact's public representation.

- Full strict projection and inferred hierarchy require retained, directly loaded native
  snapshots, including supported import scopes. Decoded, mmap, transformed and composite
  owners are not currently admitted to that complete pipeline.
- Core native indexes support ROOT/DOCUMENT/CLOSURE selection. Supported annotation-only
  overlays reuse or extend native annotation/index state. Logical overlays that affect an
  unsupported index shape reject explicitly. This is operation-specific support, not a
  claim that an overlay can enter full strict projection/reasoning.
- Verified-wire reasoner workers and ELK process deadlines are currently rejected before
  serialization; mapped native receipts are not implemented. HermiT's in-process deadline
  remains supported. Explicitly requested asserted fallback remains visible in provenance.
- Class/individual punning retains typed identity. Exact's historical property hierarchy
  combines object/data properties by IRI, which is ambiguous when those kinds share one
  IRI. Native typed graphs cannot silently substitute a different result: Exact rejects
  such property hierarchy queries with `BackendProtocolError` (`NATIVE_VIEW_REQUIRED`).
  Declaration-only punning is detected too. Upstream typed property APIs remain available.

The ambiguous property reproducer is `EquivalentObjectProperties(P O)`,
`EquivalentDataProperties(P D)`, `SubObjectPropertyOf(O OP)` and
`SubDataPropertyOf(D DP)`: the old IRI graph gave both parents to P/O/D, while typed native
components distinguish the two kinds. Rejecting this unsupported Exact presentation keeps
that discrepancy explicit; it is not recorded as semantic parity for this case.

Exact's execution contract is `exact/native-pipeline/v2`. Existing scientific selections,
exclusion predicates, projection profiles, import content, label policies and reference
boundaries stay fixed. Old cached measurements are not relabeled as candidate measurements.
The release compatibility manifest preserves the published baseline separately from the
unreleased candidate record.

## HermiT query scope and resource contract

N07 is committed in `622ce0b` and `bbf31c2`, with independent serializer/parity tests in
`0aa9e54` and `2930627` and shared-rule/datatype/blocking changes in their preceding commits.
The native base retains rules, joins, role automata, datatype registries and existential
shapes. Query-local clauses, facts, witnesses and mutable tableau state are isolated; batches
retain at most one active query state. Existing bounded Boolean-result cache policy remains.
Classification and internal realization counterexamples use the same retained native base.

Admitted generic queries include Boolean classes, nominals/equality, known role assertions,
existing source literals, universal restrictions and ground existential witnesses. Unknown
roles/new literal payloads, newly required inverse-role blocking, nested existential expansion
under variable-valued universals and global schema/role changes retain the full native rebuild
in default mode. Strict mode rejects these unsupported cases before publication. Public strict
realization and committed updates remain unadmitted; internal realization optimization does
not imply strict publication support.

Query syntax has a 1-MiB request and 16-MiB/4,096-item batch ceiling, with additional native
local-domain bounds. Resource exhaustion does not enable fallback. Native construction checks
cancellation periodically; rollback and contained failures preserve the existing lifecycle
contract. Diagnostics distinguish native delta loads, local plan/peak records, full-program
loads and default fallback rebuilds. These counters show reduced setup work, not a measured
large-ontology reasoning speedup.

## Exact execution identity and recovery costs

Exact commits `74ebf1f` and `28d6694` bind cached ontology work and extraction artifacts to
actual installed Python/native package content, including equal-version local rebuilds.
Only dependencies used by the selected reasoner enter its identity. Absolute installation
paths do not enter the digest; shadowed Python sources outside the identified distribution
are rejected. Installed dependencies are treated as immutable for a running process.
The execution contract and implementation identity remain separate from scientific inputs.

Commit `5eb4fac` preserves an original execution measurement inside the immutable extraction
artifact. Relocation and evaluator repair retain its bytes and origin attempt. Aggregated
scientific wall time and peak memory use that measurement; operational manifests retain the
current attempt's elapsed time, also exposed as separate report columns. A replay without
an original measurement, or a checkpoint continuation without complete elapsed-time evidence,
reports unavailable cost. Existing inference timing and budget/request ledgers remain intact.

These changes pass 126 harness/recovery/identity checks, 57 identity/projection/reasoner
checks after the source-shadow guard, and 20 runtime plus 110 harness/recovery checks after
the cost change. Scoped mypy, Black, isort and flake8 pass for the changed files.

## Retained evidence

Baseline identities: `data/experiments-v2/native-optimization-01/baseline.json`.
Native core operation evidence:
`data/experiments-v2/native-optimization-01/core-n04-index-evidence.json`.
Installed package and binary identities are frozen in `baseline-installed.json` and
`candidate-installed.json` in that directory. Final candidate wheels are retained under
`wheels/`, bound to full source commits and SHA-256 in `artifacts.json`. The earlier
`candidate-installed-pre-n07.json` and `artifacts-pre-n07.json` remain distinct checkpoints.
Final installed commands/results are in `installed-T3-exact.json` and
`installed-T3-hermit.json`, with corresponding logs. Native query operation/resource and
build evidence is retained in `pyhermit-n07-evidence.json`; final quality commands/results
are in `quality-final.json`.
The original completed NCIT result remains `native-preprocessing-03/source.json`; DOID is
`native-preprocessing-01/target.json`, both under `data/experiments-v2`.

- Core: 157 native encoded-view/index tests before the final domain extension. Native
  class features also pass 17 differential/ownership/resource fixtures. Growing unrelated
  annotations from 0 to 1,000 leaves a selected annotation query at one visited/published
  row; growing typed roots from 1 to 1,001 leaves three selected queries at three decoded
  rows. Cold index construction is charged separately.
- Projector: 2,673 coherent package tests, 152 native tests and 37 refreshed installed
  strict tests. Membership, ROOT reuse and canonical spill boundary fixtures assert exact
  edges, counts, ordering, cleanup and absence of scalar fallbacks.
- ELK: 1,255 source tests (three benchmark namespace skips), 373 installed tests, native
  boundary fixtures and scoped production checks. Query tests count shared base ownership,
  bounded cache eviction and neighbor access independently of unrelated ontology size. An
  installed 40-class fixture matches the Python oracle at 1/2/4/8 workers; records are in
  `native-optimization-01/elk_workers.json`. Timings are descriptive, not speedup evidence.
- HermiT: 711 encoded compiler/lifecycle tests, eight strict admission tests, four public
  query-order/batch parity tests (including five independently parsed and compiled full native
  rebuild comparisons), and five actual resource/failure/work-count tests pass. The combined
  Rust suite passes 358 unit, eight integration and six wire tests; five public realization
  tests preserve Python/native parity, cache, interruption and inconsistency behavior.
  Query tests over four versus 128 unrelated base rules use one delta load and equal local
  plan/peak-record counts, with zero full-program or fallback loads. Resource fixtures bound
  infinite iterators at 4,097 consumed items, limit pairwise reduction expansion, and verify
  that a failed batch cannot publish a successful prefix into the Boolean cache.
- Exact: 177 integration tests pass against the final four installed wheels, plus nine
  HermiT public query/resource tests against the same environment. Six compatibility-manifest
  checks pass after recording the final artifact bindings. 917 tests passed in the broader offline
  run, with its one stale benchmark assertion corrected and all 27 scale tests then passing.
  Full mypy passes for 221 source files, all five import-layer contracts pass, and scoped
  Black/isort/flake8 checks pass.

Scoped Ruff/mypy/Rust checks accompany upstream commits. Known unrelated quality baselines:
ELK mypy reports an existing unreachable statement in `reasoning/completeness.py`; core's
newer Rust clippy rejects the pre-existing large `NativeError` result type, also used by new
bindings. Neither is represented as a clean full-repository check.

## First T4 result and feature-order correction

The immutable first attempt, `native-optimization-01/job.json`, stopped with
`semantic_mismatch`. NCIT finished normally in 502.7 seconds; inputs, effective axiom count,
signatures, canonical edges, exclusions, labels and selected entities matched. The 64-feature
digest differed, so DOID did not start. This attempt is not recorded as passed.

The investigation reproduced a pre-existing correctness defect: Python set iteration supplied
neighborhood triples to a stable score-only sort and truncation. Identical 12-edge graphs
retained different top-four neighbors under four interpreter hash seeds. Both affected files
were unchanged since baseline `00941f5`; their blob IDs and the full reproduction are retained
in `native-optimization-02/correction-evidence.json` and `baseline-ordering-repro.json`.

Separate correctness commit `eb3d86a` sorts only the requested unique neighborhood triples by
(source, relation, destination) before the existing ranking and truncation. This establishes a
stable tie order without changing edges, scores or budgets. Dataset evidence schema 3 rejects
cached features selected under the old policy; recovery already binds extraction to graph code.
This is an explicit baseline-bug resolution, not a claim that one historical arbitrary ordering
must be reproduced. Commit `eb1abcb` saves exact hashed feature rows in `.features.jsonl`, without
normalizing lists or overwriting old evidence.

The correction passes 42 focused regressions; scoped formatting/type checks pass. The original
class logic matches 1,920 tiny comparisons across 32 graphs, and annotation fixtures match across
imports, nested annotations, language tags, datatypes and duplicate roots. The complete paired
wrapper passes six ordered feature rows in `native-optimization-02/smoke02/source.oracle.json`.
The first tiny wrapper failure is preserved in `smoke/`; its overly strict asserted-adapter guard
was corrected before the real input was restarted.

## Reference timeout and recovery

The second attempt, `native-optimization-02/job.json`, preserved a complete NCIT candidate
measurement of 495.4369 seconds and exact 64-row feature JSONL. The subsequent test-only
reference constructor exceeded its 120-second limit while the generic asserted-view accessors
repeatedly serialized class axioms. No new semantic mismatch was established; DOID did not start.
The failed report and original candidate measurement remain unchanged.

The third attempt keeps the pinned original feature algorithms, replacing only their generic
asserted-view input accessor with a small test-only endpoint map over native-selected raw axioms.
It preserves per-kind restriction order and avoids repeated Python axiom serialization. This
reference change passes 2,688 comparisons against the previous reference across 32 small graphs,
including cycles, equivalences, unions/intersections, built-ins and annotation variants. Row/time
limits and progress reporting are tested. The constructor now has a 1,800-second allowance.

The NCIT candidate report, code/input/import identities, exact ordered entity list, feature-file
hash and installed package identities are verified before reuse. The saved 495.4369-second result
is never relabeled as a new measurement. The native snapshot must reload because the old process
exited. The previous projection spool was transient and no full edge artifact survived; the graph
is restored once through the public native projector, checked against its saved 630,404-edge digest,
and checkpointed before reference construction. Later retries can read that verified edge artifact.
Checkpoint corruption, option mismatch and incomplete publication reject explicitly without retrying
projection; seven small checkpoint tests pass.

The complete wrapper passes three small paths: saved-candidate recovery with projection restoration,
recovery using the durable projection checkpoint, and fresh candidate execution for the pending target.
It recomputes the full ordered reference feature rows from a fresh reference graph/dataset. Only the
native snapshot/selection substrate and verified projection are shared; original annotation conversion
and hierarchy algorithms are evaluated independently. No candidate feature fields are copied into
the reference result.

## Completed T4 result

Attempt three passed at 2026-09-14 19:07:27 UTC. Both worker exits were zero; the detached
session has ended. Authoritative results are in
`data/experiments-v2/native-optimization-03/job.json` and `completed-summary.json`.

For both NCIT and DOID, input/import identities, effective axiom counts, signatures, canonical
edges, exclusions, labels and entity selections match the saved structural baseline. All 64
complete ordered feature rows per ontology match the independent reference: 128 rows, zero
mismatches. The old nondeterministic feature digests remain recorded as historical mismatches;
acceptance uses the separately documented deterministic policy applied to both implementations.

| Measured phase | Saved baseline | Native candidate |
| --- | ---: | ---: |
| NCIT preprocessing | 3,999.53 s | 495.44 s |
| NCIT projection | 2,464.37 s | 177.86 s |
| DOID preprocessing | 184.28 s | 40.60 s |
| DOID projection | 98.21 s | 15.48 s |

These are single diagnostic observations, not a repeated timing study or a general large-ontology
speedup claim. Peak RSS was 11.82 GiB for the NCIT candidate and 1.31 GiB for DOID. NCIT's completed
candidate result was reused without changing its timing. Attempt-three restoration and reference
work are separate: source reference 99.74 seconds, target reference 8.68 seconds. The final recovery
job took 573.15 seconds including restoration, target preprocessing, reference checks and process
setup; that duration is not substituted for either preprocessing measurement.

`source.restore.json`/`target.restore.json` record restoration and checkpoint costs;
`source.oracle.json`/`target.oracle.json` record exact comparisons and reference stages. Canonical
`.edges.jsonl` files and verified receipts remain available, as do both sets of feature JSONL rows.
The same local wheels, frozen populations, configuration and locked imports were used throughout.
Native parsing/annotation selection and verified graph edges are shared; the reference independently
evaluates the pinned hierarchy and annotation-conversion logic with fresh graph/feature caches.

The jobs ran sequentially with no competing builds, GPU visibility disabled, two BLAS threads,
an eight-thread native import cap, fatal-only tracing and a per-process address-space cap below
48 GiB. All earlier failed attempts remain unchanged. No models, encoders, reference labels or
G0 calls were used. N11 is now validated; scientific campaign admission and large-ontology reasoner
performance remain separate. N08 and N10 retain their explicit conditional dispositions.
