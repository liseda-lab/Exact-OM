# Migrating through 2.1

Exact-OM 2.0 changes the runtime backend, configuration shape, data acquisition, viewer name,
and run layout. Historical configs and run directories remain readable during the documented
compatibility window.

## 2.0 to 2.1 OWL-stack changes

Exact 2.1 replaces the transitional `py-horned-owl` parser with one immutable
`pyowl-core` view per OWL source and the released shared OWL2Vec* projector line. Concrete
snapshots, overlays, composites, and provider results are retained by identity; the source
facade, structural views, projector, and selected reasoner consume that exact object. RDFLib is
restricted to generic RDF/OAEI formats.

The base wheel remains Java- and build-tool-free. Install `exact-om[reasoning]` only for the
optional pyELK or pyHermiT adapters. Pre-2.1 ontology/projection cache metadata is rejected and
rebuilt from source bytes without unpickling legacy ontology objects. Each new OWL run records
path-free core/projector/reasoner versions, fingerprints, closure/document digests,
encoded/compiler schema identities, bounded ingestion diagnostics, options, backend selection,
and verified-wire/mmap state under `ontology_stack`. Encoded acceleration is negotiated from
public capabilities and remains optional throughout the compatible 0.1 package ranges.

## Required actions

1. Remove Java/mowl setup and JVM heap flags; OWL parsing is Java-free.
2. Run `exact config migrate old.yaml -o config.yaml` and review the report.
3. Replace `data/get_data.py` workflows with `exact data pull|verify|status`.
4. Install `exact-om[viz]` and use `exact-inspect` for the alignment viewer.
5. Treat `run_manifest.json` as the artifact index and use `exact run` maintenance commands.

## Compatibility map

| 1.x | 2.0 |
| --- | --- |
| `EvalutionRunner` | `EvaluationRunner` (old alias warns). |
| `exact.utils.paths` | `exact.utils.graph_search` (old module warns). |
| `exact.utils.llm_routing` | `exact.llm.routing` (old module warns). |
| `bioml-eval` | `exact-eval` (both scripts remain). |
| `times.txt` | atomic `timings.json` session ledger. |
| config without `config_version` | config v2; auto-migrated with warning. |
| `study_visualizer_runtime`, `exact-study-viz`, `EXACT_STUDY_*` | `exact_inspect`, `exact-inspect`, `EXACT_INSPECT_*`. |
| Viewer bundle scripts and `exact.analysis.study_visualizer` | `exact-inspect bundle` and `exact_inspect` APIs. |
| `data/get_data.py` | version-pinned `exact data` providers. |
| written `full_explanations.json` | indexed explanation store and `exact run export`. |
| `model/alignment/...` | layout-v2 directories plus `run_manifest.json`. |

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
readable; a compatible resume migrates records into the store. Successful finalization compacts
overlays, writes the manifest, and applies checkpoint retention. Opening an old run never
rewrites it.

The complete command examples, extras table, key-map summary, and compatibility schedule live
in the repository-level
[MIGRATION.md](https://github.com/liseda-lab/Exact-OM/blob/main/MIGRATION.md).
