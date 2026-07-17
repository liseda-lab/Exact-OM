# Runs, caching, and resume

Every new output directory is a versioned run. `run_manifest.json` records the Exact version,
layout/schema versions, producing session IDs, artifact sizes, and checksums for deliverables.

```text
RUN/
  run_manifest.json  config.yaml  timings.json  exact.log
  alignment/         evaluation/  stats/        plots/
  explanations/      checkpoints/ cache/
```

Open old and new layouts uniformly:

```console
exact run info RUN
```

## Explanation store

`explanations/index.json` maps each source IRI to one compressed shard, so a viewer reads one
shard for a source rather than loading a monolithic JSON array. Writes are append-safe: shard
bytes are flushed before the index advances. Post-selection corrections are transient overlays
that are compacted and removed at successful finalization.

Generate compatibility views on demand:

```console
exact run export RUN --what explanations --format jsonl
exact run export RUN --what explanations --src SOURCE_IRI --format json
```

## Checkpoints

Inference checkpoints contain fingerprints, processed work, cumulative inference timing, and a
record boundary in the explanation store. On resume, Exact rejects a mismatched dataset/model
fingerprint. If a process stopped after a shard append but before its checkpoint advanced, the
uncheckpointed suffix is discarded before scoring continues.

After a successful run, `output.retention.checkpoints` applies `latest`, `all`, or `none`.
Manual cleanup is foreign-file safe:

```console
exact run clean RUN --dry-run
exact run clean RUN
exact run clean RUN --all       # also manifest-owned dataset cache files
```

## Timing semantics

`timings.json` is an atomically rewritten ledger whose command sessions are append-only. Each
session has a UUID and a resolved configuration fingerprint; explanation records carry the same
session ID. Stage records label work as fresh, cache-hit, resumed, or skipped. Cumulative compute
totals include only real work, while the current-session report makes resume overhead visible
separately.

Dataset and model caches include normalized ontology/source fingerprints, candidate inputs,
selected entity kinds, and relevant configuration. Changing one of these inputs invalidates the
cache rather than reusing stale data.

Exact 2.1 increments the ontology cache identity for the shared snapshot/projector stack.
Pre-2.1 metadata is rejected with a migration-focused warning and rebuilt from source bytes;
legacy ontology objects are never deserialized. Projector and reasoner cache keys include
their package/API/schema versions, semantic options, backend choice, and the core structural
fingerprint.
