# WP-N encoded native-consumer handoff checkpoint

Date: 2026-07-27. Exact-OM implementation subject: `d172cfa355a5d2683fc47824a5d8f2ed24cf9125`.
The committed short compatibility matrix binds pyOWLCore
`af9bdb0b9178766b5f15806fb6a2f00b05e00e22`, projector
`53a23e2d385696e2be042568ade0d178580c6de4`, pyELK
`a909cfcea341834ab6d6598f80445a697b338f13`, pyHermiT
`04bd8163b532f623044d7391706ff728d1aed4b1`, and OAEI-Bio-ML-eval
`04573c09dd0e62825c3fa7c5b2490b43d5a22874`.

## Outcome

The repository-owned WP-N compatibility layer is implemented as a fail-closed checkpoint. Exact
continues to own one public core view and passes that identity to the projector and optional
reasoners. It records only bounded public compiler diagnostics; it does not decode encoded
columns, import native extensions, flatten composites, cache dense IDs, or change matching and
repair semantics.

WP-N is **not accepted for release**. pyOWLCore structural-columns v1 and pyHermiT's
`encoded-structural-compiler-v1` are now advertised. Exact revision `d172cfa` requires the public
pyHermiT native and verify handoffs to report `encoded-native`, the exact compiler schema,
released-GIL evidence, zero staging copy, and retained zero-copy buffers. pyELK and projector
remain deliberately unadvertised, and the required pure/scalar/native interpreter matrix,
accepted scale evidence, artifact audit, released dependency ranges, and published compatibility
table are still open. No file below `specs/exact-repair/` or `specs/experiments/` is part of this
implementation checkpoint.

## Implemented repository-owned slice

- Projection and reasoner cache identities include the frozen public encoded descriptor and the
  consumer compiler/schema identities, invalidating incompatible caches instead of reinterpreting
  schema-local IDs.
- Source, overlay, and composite ownership remains layered. In-process projector/reasoner adapters
  receive the original view identity; verified worker mode encodes one core artifact, reopens it
  with `mmap=True, verify=True`, and retains the mapped owner through shutdown.
- `ontology_stack` provenance contains a versioned `consumer_handoff` block for core, projector,
  and reasoner public contracts. It records selected ingestion paths, compiler digests and schemas,
  implementation/ABI versions, encoded owner/storage kinds, phase timings, and an allowlisted
  copy/materialization/wire/parser/FFI counter ledger.
- Unknown, malformed, path-bearing, incompatible, or internally inconsistent diagnostics fail
  closed. Scalar paths cannot claim an encoded identity, encoded-view timing, nonzero encoded
  buffers/copies/private IR, or released-GIL evidence.
- `benchmarks/owl_stack_scale.py` schema 5 records one-load identity, exact encoded-schema
  attestations, core-operation deltas,
  projection and hierarchy result digests, publication/compiler timing, first and complete result
  timing, CPU/wall/RSS, materialization/copy coverage, cache identity, verified-worker facts, and
  structured counter acceptance. `--require-encoded-consumers` rejects scalar fallback, missing
  counters, forbidden work, direct staging copies, absent GIL release, and unexpected core calls
  rather than relabelling incomplete evidence as native.
- Scalar-only core/projector/reasoner combinations remain supported, reasoners stay optional, and
  no Java, OAEI, or private accelerator dependency enters Exact's base installation.

The original implementation sequence is recorded by `d12e11d`, `c1a3537`, `3e6687a`, `d7c6b6f`,
`a5b03e2`, `21d74b3`, `3c3cfce`, `acabe6c`, `87fad5a`, and `680112e`. Revisions `3bbde7d`
through `d172cfa` then tightened zero-copy evidence, exact capability/version negotiation, public
stack guards, and the advertised pyHermiT handoff without changing matching, repair, or experiment
semantics.

## Local verification at this checkpoint

The current committed core compatibility record runs Exact revision `d172cfa` against the exact
core subject above: 82 focused provenance, reasoner-adapter, shared-stack, scale-benchmark, and
public-boundary tests pass, with one expected native-fallback warning on the scalar-only path.
Earlier slices also
proved one worker-wire encoding, verified mmap reopening, zero worker OWL parses, exact snapshot
identity, backend-independent compiler digests, projector edge/cache digest stability, and bounded
path-free provenance.

This source-tree verification does not replace the final installed distribution and corpus matrix.

## Acceptance ledger

| WP-N requirement | Checkpoint state |
|---|---|
| Exact source/projector/reasoner share the original in-process identity | Implemented and focused-tested |
| Overlay/composite bases remain retained without Exact flattening | Implemented and focused-tested |
| Worker handoff uses one verified core wire/mmap artifact with zero OWL parses | Implemented and focused-tested |
| Versioned cache/provenance compatibility | Implemented for candidate schemas |
| Public timing and copy/materialization/FFI/wire ledger | Complete projector vocabulary accepted and preserved; schema-5 release mode enforces exact encoded-schema, complete zero-work, direct-copy, GIL, and core-call evidence for every selected consumer |
| Scalar-only and optional-reasoner behavior | Preserved in source tests |
| Advertised core and pyHermiT public handoff | Implemented and focused-tested at the exact revisions above; pyELK/projector promotion remains external |
| Python 3.10–3.12 pure/scalar/native installed matrix | Open |
| Exact NCIT–DOID, GO, and licensed-scale semantic/performance evidence | Open |
| Maximum 25% wall regression and pinned-runner RSS gates | Open; no speedup claim is made |
| Final core/projector/pyELK/pyHermiT ranges and wheel/sdist audit | Open |
| Published exact-revision compatibility matrix and WP-M M5 handoff | Open |

## Promotion decision

Exact does not infer acceleration from package versions and does not require an encoded capability
for ordinary workflows. Keep `--require-encoded-consumers` as an evidence-only fail-closed gate
until all coordinated producers advertise their reviewed capability and the complete installed
matrix and scale records pass. The repository-owned checkpoint is ready for those external/release
runs; it does not independently declare Exact-OM `2.1.0` accepted.
