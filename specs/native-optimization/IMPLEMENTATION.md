# Native stack implementation record

Status: **in_progress**, 2026-09-14. The user authorized implementation of this suite.
The long scientific campaign has not been started. This record distinguishes package
fixtures from installed-stack integration and the bounded NCIT–DOID measurement.

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
| N02 | in_progress | Core receipts, consumer preflight and native metadata/results; final coherent wheel integration pending |
| N03 | validated | Proven one-document ROOT reuse; general scope selection remains explicit |
| N04 | validated | Native annotation postings, class/property features, selected typed rows and domain/range indexes; installed Exact parity |
| N05 | validated | ELK task-aware stages, shared query base, bounded LRU and native taxonomy adjacency |
| N06 | validated | HermiT native parent/child adjacency with degree-proportional enumeration |
| N07 | in_progress | Shared HermiT immutable plans and lazy batches; complete native delta integration pending |
| N08 | deferred_with_reason | Exact's current static hierarchy adapters make no recurring committed update calls |
| N09 | validated | Native canonical sort/dedup/spill/merge with bounded working storage |
| N10 | deferred_with_reason | No post-change real-input evidence yet selects another recurring canonical/compiler phase |
| N11 | in_progress | Public Exact integration and bounded fixtures; final T3/T4 pending |

`validated` above applies to the package mechanism and its completed T1/T2/installed
fixtures. N11 remains open until the relevant integrated gates are recorded. No end-to-end
speedup is claimed from these statuses or operation counts. T4 preprocessing does not pass
G0, admit an experiment block, or establish large-ontology reasoner performance.

## Installed candidate checkpoints

| Repository | Tested source | Wheel SHA-256 |
| --- | --- | --- |
| pyOWLCore | `5fd93c8` | `f66c068f3a88d0ae59553103fefef4994c43ecca1282300b4dd2486bcbd905d9` |
| projector | `a067601` | `53c752e345d8c18ac1784000609902118b724d61fd8b59d7a79e5cb693d1e0fd` |
| pyELK | `bce95f8` | `3ae4a551d2288b0d703c43edc116b9d5a458f663a36812ac19801f1595b4caef` |
| pyHermiT N02 checkpoint | `8132940` | `ebef1a880219dfa2e4fd1cac7ac844c6eb3ca291d10a1792c401701de9394300` |

Exact adoption is committed in `637f7c8`, the scale fixture update in `542af42`, and the
separate published-baseline/candidate manifest in `76e5659`. HermiT N07 remains an active
change; its checkpoint above must not be represented as the final N07 wheel.

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

## Evidence retained so far

Baseline identities: `data/experiments-v2/native-optimization-01/baseline.json`.
Native core operation evidence:
`data/experiments-v2/native-optimization-01/core-n04-index-evidence.json`.
Installed package and binary identities are frozen in `baseline-installed.json` and
`candidate-installed-pre-n07.json` in that directory. Candidate wheels are retained under
`wheels/`, bound to source commits and SHA-256 in `artifacts-pre-n07.json`; HermiT N07 will
produce a separately identified final artifact.
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
- HermiT: N02 admission/result fixtures and N06 adjacency pass. N07 evidence is still being
  completed and will replace this interim statement before claiming full-stack completion.
- Exact: 175 installed-stack integration tests pass; 917 tests passed in the broader offline
  run, with its one stale benchmark assertion corrected and all 27 scale tests then passing.
  Full mypy passes for 221 source files, all five import-layer contracts pass, and scoped
  Black/isort/flake8 checks pass.

Scoped Ruff/mypy/Rust checks accompany upstream commits. Known unrelated quality baselines:
ELK mypy reports an existing unreachable statement in `reasoning/completeness.py`; core's
newer Rust clippy rejects the pre-existing large `NativeError` result type, also used by new
bindings. Neither is represented as a clean full-repository check.

## Remaining sequence

1. Finish domain/range adoption and HermiT shared query-delta execution; complete package
   correctness, rollback, resource and reduced-work fixtures.
2. Install all four release wheels together; record source commits, hashes, native binary
   identities, schemas and capability checks. Run Exact's applicable installed-stack tests.
3. Run one detached model/reference-free NCIT–DOID diagnostic with frozen entities and
   semantic digests, separate fresh processes, source 120-minute and target 30-minute
   ceilings. Pause competing builds. Keep at least 12 GiB memory headroom and cap the job
   at 48 GiB or the stricter host/cgroup allowance.
4. Hand off its job, phase logs and terminal comparison record for monitoring. Do not start
   G0 or the long experiment campaign as a consequence of preprocessing success.
