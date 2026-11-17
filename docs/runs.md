# Artifacts, storage and resume

A standard `exact align` run contains alignment files, a run manifest,
statistics, checkpoints and a compressed explanation store. Inspect and export
without the unfinished visualization product:

```console
exact run info runs/demo
exact run export runs/demo --format jsonl --output evidence.jsonl
exact run clean runs/demo --dry-run
```

Read `exact run clean --help` before deleting artifacts. Preserve resume
checkpoints until the run is complete. Keep final alignments, configuration,
model/input hashes and metrics even when intermediate data is archived.

Staged runs have independent directories:

| Stage | Primary outputs |
|---|---|
| Native | `pairs.jsonl`, `sources.txt`, `anchors.json`, `scope.json` (global), source shards, candidate Parquet, identity and completion receipts |
| Decisions | Immutable plan/request files, SQLite probability cache, scored JSONL |
| Acceptance | Model bundle and fold checkpoints; CSV/RDF submission and completion receipt |
| Local ranking | Fitted JSON model, fold checkpoints, candidate-ranking TSV and receipt |

Staged artifacts use their own commands and receipts; `exact run` manages
standard alignment run layouts. Re-run identical stage commands to resume.
Do not edit files under an existing completed stage. Use a new directory after
a code/model/configuration change. Input and output hashes detect stale or
altered artifacts.

Native shards retain selected evidence and make interrupted scoring restartable.
The consolidated JSONL also makes downstream streaming portable; it duplicates
the shard payload for convenience. Archive one representation only after
validating the complete result and deciding that native resume is no longer
needed. Do not compress or relocate live inputs behind an active process.

Model bundles containing joblib/pickle must come from trusted sources.
Caches and snapshots are local trusted artifacts, not an untrusted interchange
format. Move whole model bundles with their manifests.
