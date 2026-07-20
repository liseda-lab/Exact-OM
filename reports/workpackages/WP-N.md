# WP-N encoded native-consumer handoff checkpoint

Date: 2026-07-20. Exact-OM revision: `acabe6c`. Coordinated source candidates:
pyOWLCore `34b9e84`, projector `46086fb`, pyELK `886f6a3`, pyHermiT `3c56fc2`, and
OAEI-Bio-ML-eval `fcebff1`.

## Outcome

The repository-owned WP-N compatibility layer is implemented as a fail-closed checkpoint. Exact
continues to own one public core view and passes that identity to the projector and optional
reasoners. It records only bounded public compiler diagnostics; it does not decode encoded
columns, import native extensions, flatten composites, cache dense IDs, or change matching and
repair semantics.

WP-N is **not accepted for release**. The coordinated encoded capabilities remain unadvertised,
and the required pure/scalar/native interpreter matrix, accepted scale evidence, artifact audit,
released dependency ranges, and published compatibility table are still open. No file below
`specs/exact-repair/` or `specs/experiments/` is part of this implementation checkpoint.

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
- `benchmarks/owl_stack_scale.py` schema 3 records one-load identity, core-operation deltas,
  projection and hierarchy result digests, publication/compiler timing, first and complete result
  timing, CPU/wall/RSS, materialization/copy coverage, cache identity, and verified-worker facts.
  `--require-encoded-consumers` rejects scalar fallback rather than relabelling it as native
  evidence.
- Scalar-only core/projector/reasoner combinations remain supported, reasoners stay optional, and
  no Java, OAEI, or private accelerator dependency enters Exact's base installation.

The implementation sequence is recorded by `d12e11d`, `c1a3537`, `3e6687a`, `d7c6b6f`,
`a5b03e2`, `21d74b3`, `3c3cfce`, and `acabe6c`.

## Local verification at this checkpoint

The latest handoff hardening passed 39 focused provenance, reasoner-adapter, and scale-benchmark
tests. Black, flake8, and strict mypy passed for the changed runtime modules. Earlier slices also
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
| Public timing and copy/materialization/FFI/wire ledger | Implemented where consumers publish fields; missing fields remain unavailable |
| Scalar-only and optional-reasoner behavior | Preserved in source tests |
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
