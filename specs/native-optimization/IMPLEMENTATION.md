# Native stack implementation record

Status: **in_progress (T4)**, 2026-09-14. Implementation is complete; the applicable T1–T3
checks pass. One bounded NCIT–DOID validation is running detached. The long scientific
campaign has not been started. This record distinguishes package fixtures, installed-stack
integration and the pending real-input result.

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
| N10 | deferred_with_reason | No post-change real-input evidence yet selects another recurring canonical/compiler phase |
| N11 | in_progress | Public Exact integration and final T3 pass; one detached T4 comparison is running |

`validated` above applies to the package mechanism and its completed T1/T2/installed
fixtures. N11 remains open until the relevant integrated gates are recorded. No end-to-end
speedup is claimed from these statuses or operation counts. T4 preprocessing does not pass
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
The candidate status is `installed_validated`; full `validated` requires T4 to pass.

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

## Detached T4 handoff

The second gate is running in tmux session `exact-native-validation`, launcher PID `1167866`.
Its authoritative terminal result will be `data/experiments-v2/native-optimization-02/job.json`.
The tracked manifest records the launch state, not an automatic promise of success. Monitor from
the Exact repository with:

```sh
watch -n 10 cat data/experiments-v2/native-optimization-02/job.json
```

The same native wheels, frozen 64-entity selections, configuration and locked imports are used.
The gate still compares input/import identities, effective axiom counts, signatures, canonical
edges, exclusions and labels with the saved baselines. The unstable historical feature hash is
retained and reported separately; it is not replaced with a candidate-generated golden.

After candidate timing finishes, the same retained native snapshot feeds a bounded test-only
reference using the pinned original hierarchy algorithms and original annotation conversion.
Both feature builders use the separately corrected deterministic neighborhood policy. Acceptance
requires exact ordered agreement for all 64 complete feature rows, including matching digests and
counts. Candidate and reference rows are retained separately. No second ontology load, projection
or whole-ontology annotation materialization is required.

The reference shares native parsing, native annotation selection, verified edges and unchanged
IC statistics. Its selection substrate is tested separately against complete small raw-axiom
scans; this is not an independent full ontology parser/index oracle. Class hierarchy construction
uses independently consumed typed rows and original algorithms, with a 400,000-row/120-second
constructor bound. It rejects limits without truncation. Oracle wall time and lifetime peak RSS
are recorded separately from the completed candidate preprocessing measurement.

`source.stages.jsonl`/`target.stages.jsonl` show candidate phase progress; `source.oracle.json`/
`target.oracle.json` show the subsequent feature check. Each side also has `.log`, `.fatal.log`,
result `.json` and terminal `.exitcode` files. Source and target run sequentially in fresh
processes, with 120- and 30-minute total ceilings. The second attempt's address-space cap is
50,082,639,872 bytes (below 48 GiB), retaining 12 GiB initial headroom. BLAS uses two threads;
native imports retain their eight-thread cap. GPU visibility is disabled. No compiler builds
compete with the job; fatal-only trace capture is used.

Failures or mismatches stop the sequence and preserve the attempt. No automatic retry, model,
encoder, reference-label or G0 calls are made. N11 remains `in_progress` until the comparison
passes; neither large-ontology reasoner performance nor scientific campaign admission follows
from this preprocessing gate. N08 and N10 retain their explicit conditional dispositions.
