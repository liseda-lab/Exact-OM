# WP-I — Dataset & Track Retrieval (`exact/tracks/`)

**Depends on**: WP-A. Independent of WP-B (pure data plumbing); coordinates with WP-G on the
BioKG/CSV layouts and with WP-J on the `data:` config section. **Size**: M.
**Behavior**: additive — explicit `-s/-t/-f/-c` paths keep working unchanged; tracks are a
convenience layer on top.
**Status**: Done (2026-07-16).

## Context

Dataset acquisition today is one script (`data/get_data.py`): Zenodo Bio-ML archive
(record 13119437) + OAEI-2025 Conference zips, no integrity checks, no change detection, and
the layouts it produces are implicit. The tracks we must support, each with a different shape:

| Track | Source | Layout (verified 2026-07-15) |
|---|---|---|
| **Bio-ML 2026** | HF `OAEI-ML/bio-ml`, revision `2026` | Per pair (`NCIT-DOID`, `SNOMED-FMA`, `SNOMED-NCIT`): `local.{train,valid,test}.cands.tsv`, `refs_equiv/`, `repaired/` variants, `SHA256SUMS`, `resolved_versions.json`. **Ontologies NOT redistributed** (licensing) — acquired separately per the pinned versions. |
| **DISO 2026** | HF `OAEI-ML/diso-oaei`, revision `v2026` | `ontologies/*.owl` (10), `references/` (RDF, repaired+unrepaired, 6 pairs), `pools/<pair>/pools.jsonl` (50 candidates/query **incl. a NIL option**), `archives/*.zip`. |
| **Bio-ML legacy** | Zenodo 13119437 | current `get_data.py` layout (owl + tsv refs/cands). |
| **Conference** | OAEI site zips | owl files + reference-alignment RDF (already parsed by `get_data.py:24-62`). |
| **Anatomy** | OAEI site | mouse/human `.owl` + `reference.rdf` (classic pair). |
| **OAEI Knowledge Graph** | OAEI site | per-pair KG dumps + reference alignments (classes, properties, **instances** — pairs well with WP-F). |
| **BioKG-Align** | HF (org TBD — **not published yet**) | kit-style CSV triples + candidates (WP-G's `csv_kg` layout). Provider ships as a stub with the layout mapping ready; repo id + revision become config the day it publishes. |

PyLogMap's pattern (fetch script + committed mini oracle bundles for hermetic tests + nightly
real-data runs) is the testing model to copy.

## Design

### I1. `TrackProvider` protocol + registry

Contracts §12 (added in this WP). `exact/tracks/`:

```
exact/tracks/
  __init__.py        # registry: builtin providers + "exact.tracks" entry-point group
  provider.py        # TrackProvider protocol, TaskLayout, VerificationReport
  lockfile.py        # datasets.lock.json read/write
  http.py            # DeclarativeHttpProvider — URL/zip tracks driven by YAML descriptors
  hf.py              # HfProvider base — huggingface_hub snapshot_download, revision pinning
  builtin/
    bioml_hf.yaml    bioml_zenodo.yaml   diso.yaml   conference.yaml
    anatomy.yaml     oaei_kg.yaml        biokg.yaml   # declarative descriptors (see I3)
```

`TaskLayout` is the canonical materialized shape the pipeline already consumes: paths for
source/target (owl file **or** csv-kg dir), reference splits (train/valid/test where they
exist), candidates file, plus `extras` (e.g. repaired refs variant) and `provenance`.

### I2. Integrity & change detection (the "test for change" requirement)

- Every materialization writes/updates `<data_root>/datasets.lock.json`: per task —
  provider, upstream id, **pinned revision** (HF revision / URL + ETag/Last-Modified), per-file
  sha256, `retrieved_at`, provider version.
- **Pin by default**: a materialized task never changes silently. `verify` re-hashes local
  files against the lock (detects local tampering/corruption) and, for HF, compares the pinned
  revision against the repo's current revision for that tag/branch (detects upstream movement).
  Where the dataset ships its own `SHA256SUMS` (bio-ml), verify against **both**.
- `status` reports: `ok | local-drift | upstream-moved | not-materialized` per task.
  `pull --update` re-pins to the newest upstream revision explicitly; nothing else does.
- Runs record the lock entry (provider, revision, hashes) into `run_stats.json` provenance so
  results are traceable to exact data versions.

### I3. Declarative descriptors ("streamlined config way")

Built-in tracks are **YAML descriptors, not code**: upstream (hf repo+revision | urls+sha256),
unpack rules, and a mapping from the upstream layout to `TaskLayout` fields (glob patterns +
per-track transforms named from a small registered set: `alignment_rdf_to_tsv` (exists in
`get_data.py:_read_alignment`), `pools_jsonl_to_cands_tsv` (DISO: JSONL pools → `TgtCandidates`
TSV; **drop the NIL entry** and record `nil: true` in extras — NIL-aware matching is future
work, flagged in the descriptor), `flatten_refs_equiv` (Bio-ML legacy, replaces
`move_files_and_cleanup`)). Adding a future dataset = adding a YAML (users can point
`data.descriptor: path/to/custom.yaml` — this is the "any dataset, config-only" extension), and
third parties can ship providers via the `exact.tracks` entry-point group.

**Licensed ontologies** (bio-ml 2026: SNOMED/UMLS-derived): descriptors declare
`user_supplied: {snomed: {sha256: <from resolved_versions.json>, help: "obtain from ..."}}`.
`pull` fails actionably listing what to place where; supplied files are hash-checked against the
pins (mismatch = warning with both hashes, not a hard fail — pinning docs may lag).

### I4. CLI + config

- `exact data list | pull <track>[/<task>] [--root data/] [--revision X] [--update] | verify | status`
  (new `data` subcommand group; `data/get_data.py` is deleted, its logic absorbed into the
  descriptors + transforms; `create_validation_samples` moves to `exact/tracks/sampling.py`).
- **This WP owns the `exact` subcommand scaffolding** (first WP to need it): add argparse
  subparsers to the `exact` entry point such that `exact data ...` works while the current
  flat-flag invocation (`exact -s ... -t ...`) keeps working unchanged (no positional args
  today, so an optional leading subcommand is backward-compatible — add a regression test).
  WP-J's `exact config ...` group plugs into the same scaffolding.
- Config (`data:` section, v2 naming per WP-J — additive `dataset_track:` shim until WP-J lands):
  `data: {track: bioml, task: ncit-doid, root: data/, revision: "2026"}` resolves through the
  registry to a `TaskLayout` and fills source/target/refs/candidates automatically; explicit
  path keys/flags always override. `pull` happens lazily on first use (with a log line) or
  eagerly via the CLI.
- HF dependency: `huggingface_hub` under extra `[hf]`; selecting an HF-backed track without it
  installed → actionable install error (same pattern as WP-E's extra).

## Tests

1. Hermetic: committed **mini track fixtures** under `tests/fixtures/tracks/` — a fake HF
   snapshot dir (bio-ml shape incl. `SHA256SUMS` + `resolved_versions.json`), a fake DISO shape
   (pools.jsonl with NIL), a fake conference zip; providers run against them with
   `snapshot_download` monkeypatched. Cover: materialize → lock written → verify ok; corrupt a
   file → `local-drift`; bump fake upstream revision → `upstream-moved`; `--update` re-pins;
   missing user-supplied ontology → actionable error; NIL dropped from pools with extras flag.
2. Transform unit tests: `alignment_rdf_to_tsv` (reuse the WP-G round-trip),
   `pools_jsonl_to_cands_tsv`, `flatten_refs_equiv`.
3. `requires_data` (network): real `pull` of conference + anatomy + one bio-ml task; e2e align
   run consuming a `data:` config block end-to-end.
4. Provenance: `run_stats.json` contains the lock entry after a track-based run.

## Out of scope

NIL-aware candidate scoring (descriptor flags it; matching semantics are future work);
BioKG HF wiring beyond the stub (activated when the org publishes); mirroring datasets we may
not redistribute; a generic "HF datasets-library" loader (we snapshot files, we don't need
`datasets`).

## Acceptance criteria

1. `exact data pull conference --root data/ && exact data verify --root data/` green from a
   clean clone (network test).
2. `exact -y config_with_data_block.yaml -o exp/x` runs with **no** `-s/-t/-f/-c` flags on a
   materialized track; explicit flags still win when given.
3. Deleting `data/get_data.py` leaves no references (`git grep get_data`).
4. Lockfile survives and correctly reports all four `status` states in hermetic tests.
5. A new dataset is addable by YAML descriptor alone — prove it in a test that loads a custom
   descriptor from a temp path.

## Deviations

The literal `git grep get_data` gate is scoped to runtime/source references. WP-H is required to
retain `data/get_data.py` in migration documentation, so those historical references remain;
the removed downloader has no live callers.
