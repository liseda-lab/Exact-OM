# Output files

`run_manifest.json` is the authoritative artifact index. Not every run emits every optional
file, and consumers should use `RunReader` rather than concatenate paths.

| Path | Purpose | Retention |
| --- | --- | --- |
| `run_manifest.json` | Layout/schema version, sessions, artifact sizes, provenance, deliverable hashes. | Keep. |
| `config.yaml` | Resolved config-v2 snapshot. | Keep. |
| `timings.json` | Append-only command sessions and stage records. | Keep. |
| `times.txt` | Deprecated human-readable rendering of cumulative stage minutes. | Regenerable. |
| `exact.log` | Optional command log. | Keep as needed. |
| `alignment/maps_global.tsv` | Filtered global mapping deliverable. | Keep. |
| `alignment/maps_local.tsv` | Candidate ranking deliverable. | Keep. |
| `alignment/align.rdf` | Optional OAEI Alignment-RDF output. | Keep. |
| `alignment/alignment.typed.tsv` | Optional typed BioKG-compatible output. | Keep. |
| `alignment/alignment.json` | Optional mapping records with relation/kind. | Keep. |
| `evaluation/evaluation_results.*` | Builtin/upstream evaluation reports. | Keep. |
| `stats/summary_metrics.csv` | Pair-level numeric/selector fields. | Keep. |
| `stats/run_stats.json` | Aggregates, timing summary, provenance, calibration. | Keep. |
| `stats/run_stats.csv` | Spreadsheet view of run aggregates. | Optional. |
| `stats/llm_calibration.json` | Calibration messages and learned metadata. | Optional. |
| `plots/*` | Dataset/postprocess figures. | Optional. |
| `explanations/index.json` | Source-to-shard index and store schema. | Keep. |
| `explanations/shards/*.jsonl.zst` | Complete per-pair audit records, colocated by source. | Keep. |
| `explanations/full_explanations.json` | Deprecated derived compatibility export. | Regenerable. |
| `checkpoints/*.json` | Resume manifests and stage checkpoints. | Prunable after success. |
| `cache/*` | Dataset/model caches. | Regenerable. |
| `dataset/*` | Materialized candidate/features table and its cache metadata. | Regenerable. |

Explanation overlays may exist during a crashed/in-progress run. Successful finalization merges
them into the shards and removes them. `output.retention.checkpoints` controls automatic
checkpoint pruning.

```console
exact run info RUN
exact run export RUN --what explanations --format csv
exact run clean RUN --dry-run
```

Layout-v1 directories remain readable, but all new producers write layout v2 only.
