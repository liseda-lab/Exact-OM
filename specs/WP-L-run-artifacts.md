# WP-L — Run Artifacts & Explanation Store

**Depends on**: WP-C (manifest references `timings.json`), WP-D (its `audit_io.py` consolidation
is the seam this WP replaces), coordinates with WP-K (the viewer reads through this WP's
`RunReader` API). **Size**: M–L. Lands early wave 3 (Agent 2, after WP-D).
**Behavior**: final deliverables (alignment TSVs, eval results) unchanged byte-for-byte; the
*internal* explanation/audit artifacts are restructured behind a versioned layout with readers
for both versions.

## Context — what a run directory looks like today, and why it hurts

A single run writes, across several conventions:

- Deliverables: `model/alignment/src2tgt.maps_{global,local}.tsv`, and JSON/CSV under
  `model/alignment/<sub_dir or "default">/` (the literal `"default"` subdir is an API wart).
- Explanation data in **up to four overlapping representations**:
  audit shards (JSONL+zstd, incremental, `_append_audit_records`), candidate-record shards
  (separate JSONL+zstd stream), post-inference **final-overlay** manifests (corrections layered
  over the shards), and — when `save_json: True` — a monolithic `full_explanations.json`
  assembled from the above (this is the only form `exact-inspect` and `user_study` can read
  today, so users enable it, duplicating everything again, uncompressed).
- Resume state: `checkpoints/inference_*.json` manifests that **accumulate without bound**
  (restore scans newest-first; nothing prunes), each pointing at shard/overlay sidecars.
- Loose top-level files: `exact.log`, `times.txt`/`timings.json`, `run_stats.{json,csv}`,
  `summary_metrics.csv`, `llm_calibration.json`, `evaluation_results.csv`, plot images from two
  different stages.

Consequences: run dirs get heavy (large tasks: shards + their overlay copies + the monolithic
JSON), the viewer needs a full `full_explanations.json` load for a single pair, humans can't
tell deliverables from resume debris, and nothing says which files belong to which session when
experiments are layered in one dir (the WP-C problem, artifact edition).

## Design

### L1. Versioned layout + manifest (`exact/runs/` — new package)

```
exact/runs/
  layout.py       # RunLayout: resolves every artifact path; supports layout v1 (today) and v2
  manifest.py     # run_manifest.json read/write
  store.py        # ExplanationStore: sharded write, indexed read, compaction
  reader.py       # RunReader protocol impl (contracts §14) — what exact-inspect/user_study
                  #   consume; `exact run ...` subcommands plug into WP-I's CLI scaffolding
  gc.py           # retention policy + `exact run clean`
```

Layout v2 (new runs; v1 dirs remain readable forever via `RunLayout`):

```
<output_dir>/
  run_manifest.json      # schema_version, layout_version, artifact index (path, kind, schema,
                         #   producing session run_id, sha256 for deliverables), exact version
  timings.json  exact.log
  alignment/             # deliverables only: maps_global.tsv, maps_local.tsv, alignment.rdf, …
  evaluation/            # evaluation_results.json/csv
  explanations/          # THE single source of truth (see L2)
    shards/00000.jsonl.zst …
    index.json           # src_iri -> shard id (+ record counts, schema_version)
  stats/                 # run_stats.json/csv, summary_metrics.csv, llm_calibration.json
  plots/                 # all figures, prefixed by stage (dataset_*, post_*)
  checkpoints/           # resume state ONLY — prunable at any time after a finalized run
```

All path knowledge moves into `RunLayout`; grep-gate: no other module builds these paths by
string concatenation. Consumers to migrate: trainer save paths (`ITrainer.alignment_dir` etc.),
`analysis/user_study.py` (hardcodes `run_dir/model/alignment/...`), `exact_inspect`,
`tools/aggregate_results.py`, `analysis/alignment_diagnostics.py`.

### L2. Explanation store — one representation, indexed, derived views

- **Write path** (trainer, replacing WP-D's `audit_io.py` internals behind the same call
  sites): one record per scored pair — the union of today's audit record + candidate-record
  fields (schema documented + versioned; overlaps de-duplicated, a migration table maps old
  field names). Appended to bounded zstd-JSONL shards (`output.explanations.shard_mb`, default
  32 MB compressed), sharded by source-IRI hash so one source's candidates colocate. Still
  append-only and crash-safe → **resume keeps working exactly as before** (shards + manifest
  pointers replace the parallel audit/candidate streams).
- **Overlays become transient**: post-inference/selector corrections still write overlay files
  during the run (crash safety), but **finalization compacts them into the shards** and deletes
  them. After a successful run, the store is self-contained: shards + `index.json`.
- **Read path**: `ExplanationStore.get(src_iri) -> list[Record]` decompresses exactly one shard;
  `iter_all()` streams. This is what `exact-inspect open` uses — no monolithic JSON needed.
- **Derived exports** (machine- and human-readable views, generated on demand, never a second
  source of truth): `exact run export <dir> --what explanations [--src IRI] [--format
  json|jsonl|csv]`. `save_json: True` (→ v2 key `output.save.full_explanations_json`) still
  emits `full_explanations.json` at finalization for backward compat — implemented as this
  export, deprecation-noted, removed in 2.1 once `exact-inspect`/`user_study` read the store.
- Schema is versioned (`explanations/schema_version`); records carry the producing session
  `run_id` (joins with `timings.json` sessions — solves "which experiment wrote this" in
  layered dirs).

### L3. Finalization & retention

- `_finalize_artifacts()` at successful run end: compact overlays → shards, write/refresh
  `run_manifest.json`, prune checkpoints to the **latest valid manifest + its sidecars**
  (policy `output.retention.checkpoints: latest|all|none`, default `latest`), log a size report
  (per-directory MB, before/after compaction).
- `exact run clean <dir> [--keep-resume] [--dry-run]`: manual GC for old/crashed runs — keeps
  deliverables+stats+manifest, drops checkpoints and (with `--all`) the dataset cache. Prints
  what it frees. `exact run info <dir>`: renders the manifest (sessions, artifacts, sizes,
  schema versions) for quick human inspection.
- Never delete anything not listed in the manifest or matching known checkpoint patterns —
  layered experiment dirs may contain foreign files.

### L4. Consumer updates

- `exact_inspect` (WP-K): `open` mode reads via `RunReader` (store-backed when
  `explanations/` exists, `full_explanations.json` fallback for v1 dirs). Wins: instant
  startup, per-source lazy loading, and the bundle precompute (`exact-inspect bundle`) shrinks
  to a thin selection layer.
- `analysis/user_study.py`: consumes `RunReader` instead of the monolithic JSON.
- Benchmarks (03-performance): add `explanation_store_write` (records/s during inference
  append) and `explanation_store_read` (single-source lookup latency on a 100k-record store —
  budget: <300 ms cold).

## Config additions (v2-native)

| Key | Default | Meaning |
|-----|---------|---------|
| `output.explanations.shard_mb` | `32` | Max compressed shard size |
| `output.save.full_explanations_json` | `false` | Emit legacy monolithic JSON at finalization (deprecated) |
| `output.retention.checkpoints` | `"latest"` | Checkpoint pruning at finalization |

## Tests

1. Round-trip: write store during a fixture run → `get(src)` returns records identical to the
   legacy audit+candidate+overlay merge (golden comparison against a pre-WP-L run).
2. Resume: crash mid-run (kill after N batches) → restore from shards+checkpoint → final store
   equals an uninterrupted run's store (reuses WP-C/WP-D checkpoint tests' harness).
3. Compaction: overlay corrections visible via `get()` after finalization; overlay files gone;
   `full_explanations.json` export byte-equivalent to legacy output for the same run.
4. Layout: `RunLayout` resolves a committed v1 fixture run dir and a v2 dir; `user_study` and
   `exact_inspect` tests pass against both.
5. GC: `run clean` on a layered dir removes only manifest-known/checkpoint-pattern files
   (foreign-file safety test); `--dry-run` deletes nothing.
6. Size regression: fixture run dir total bytes with defaults ≤ the pre-WP-L equivalent with
   `save_json: False` (and document the measured reduction vs `save_json: True`).

## Out of scope

Changing deliverable formats (TSV/RDF are WP-G's); a database backend (sqlite index is allowed
later if `index.json` shows limits — note in code); retention of dataset caches beyond the
`clean --all` path; any change to what is *recorded* per pair (schema unions existing fields).

## Acceptance criteria

1. Deliverables byte-identical on fixture + conference smoke; store round-trip test green.
2. A resumed heavy-run scenario (test 2) ends with **one** checkpoint manifest, zero overlay
   files, and a manifest listing every artifact.
3. `exact-inspect open` on a v2 run dir serves a pair's explanation without reading more than
   one shard (assert via instrumentation), and still opens v1 dirs.
4. `exact run info` / `exact run clean --dry-run` output reviewed in PR (human-readability is
   an explicit deliverable).
5. Grep gates: no `"model/alignment"` string outside `RunLayout`; no module writes into
   `checkpoints/` except `exact/runs` + trainer checkpointing.
