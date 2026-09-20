# Migrating through 2.1

Exact-OM 2.1 moves the shared Java-free OWL stack to the released pyOWL 0.2 package family.
The 2.0 configuration, data-acquisition, viewer, and run-layout migrations remain in effect,
and immutable completed run artifacts remain readable.

## 2.0 to 2.1 OWL-stack changes

The base package requires `pyowl-core>=0.2.1,<0.3` and
`pyowl2vec-star-projector>=0.2.1,<0.3`; 0.2.1 is the minimum supported native package release.
Optional Java-free reasoners use `pyelk-reasoner>=0.2.1,<0.3` and
`pyhermit>=0.2.1,<0.3`. Bio-ML evaluation uses
`oaei-bioml-eval>=0.2.1,<0.3`.

Core 0.2 introduces model schema 2, including corrected anonymous-individual and repeated
isomorphic-component scoping. Exact retains one immutable core view for each OWL source.
The source facade, structural views, projector, and selected reasoner consume the same native
snapshot. The strict native pipeline requires its validation receipt; decoded, mmap, overlay,
and composite owners are not supported for this execution path.
RDFLib remains restricted to generic RDF/OAEI formats.

### Ontology-cache boundary

Every ontology-derived schema-1 cache is incompatible with core model schema 2. Exact rejects
parsed-ontology, projection, compiler, dataset, and resumable-workflow cache metadata before
unpickling or interpreting dense identities. It then rebuilds from the original source using
core 0.2.

There is no forward or backward cache converter. Matching a source path, mtime, or digest does
not make a schema-1 cache safe. Completed immutable run outputs remain readable as historical
results, but a run cannot resume through a schema-1 ontology cache.

The base wheel remains Java- and build-tool-free. Install `exact-om[reasoning]` only for
optional pyELK or pyHermiT. New OWL runs record path-free public core/projector/reasoner
versions, model/wire/encoded contracts, fingerprints, closure/document digests, bounded
ingestion diagnostics, options, backend selection, and cache state under `ontology_stack`.
Published 0.2.1 wheels replace the local development builds; their installation identities
remain distinct for cache and experiment recovery.

## Required actions

1. Upgrade Exact-OM and reinstall the extras used by the workflow.
2. Remove Java/mOWL setup and JVM heap flags; OWL parsing and reasoning integrations are
   Java-free.
3. Allow Exact to rebuild ontology-derived caches from the original OWL sources. Do not copy
   or convert schema-1 caches.
4. Run `exact config migrate old.yaml -o config.yaml` and review the report.
5. Replace `data/get_data.py` workflows with `exact data pull|verify|status`.
6. Install `exact-om[viz]` and use `exact-inspect` for the alignment viewer.
7. Treat `run_manifest.json` as the artifact index and use `exact run` maintenance commands.

## Compatibility map

| 1.x | 2.0+ |
| --- | --- |
| `EvalutionRunner` | `EvaluationRunner` (old alias warns). |
| `exact.utils.paths` | `exact.utils.graph_search` (old module warns). |
| `exact.utils.llm_routing` | `exact.llm.routing` (old module warns). |
| `bioml-eval` | `exact-eval` (both scripts remain). |
| `times.txt` | Atomic `timings.json` session ledger. |
| Config without `config_version` | Config v2; auto-migrated with warning. |
| `study_visualizer_runtime`, `exact-study-viz`, `EXACT_STUDY_*` | `exact_inspect`, `exact-inspect`, `EXACT_INSPECT_*`. |
| Viewer bundle scripts and `exact.analysis.study_visualizer` | `exact-inspect bundle` and `exact_inspect` APIs. |
| `data/get_data.py` | Version-pinned `exact data` providers. |
| Written `full_explanations.json` | Indexed explanation store and `exact run export`. |
| `model/alignment/...` | Layout-v2 directories plus `run_manifest.json`. |

The deprecated action classes delegate to plain `run_alignment`/`run_evaluation` functions.
Layout-v1 directories remain read-only compatible through `RunReader`.

## Configuration moves

The migration command is driven by the same `V1_TO_V2` map as the loader. Major moves are
`dataset_params` to `dataset`, candidate settings to `candidates`, decision settings to
`matching`, model declarations to ordered `pipeline` entries, LLM profiles/routing under
`llm`, evaluation K under `evaluation`, and save/plot/sanity fields under `output`. Removed
reasoner-process fields are reported rather than ignored silently.

## Artifacts and resume

New checkpoints reference the indexed explanation store. Old full/schema-v2 checkpoints remain
readable when they do not depend on an incompatible ontology cache; a compatible resume
migrates records into the store. Successful finalization compacts overlays, writes the
manifest, and applies checkpoint retention. Opening an old run never rewrites it.

The complete command examples, extras table, key-map summary, and compatibility schedule live
in the repository-level
[MIGRATION.md](https://github.com/liseda-lab/Exact-OM/blob/main/MIGRATION.md).
