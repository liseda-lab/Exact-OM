# Architecture

Exact-OM separates stable contracts from replaceable implementations and delivery surfaces.
The matching action resolves data/configuration, builds a `KnowledgeSource`-backed dataset,
runs an ordered model pipeline, writes registered formats, evaluates results, and finalizes a
versioned run.

```mermaid
flowchart LR
    CLI[CLI / Python API] --> Action[Alignment action]
    Track[Track provider] --> Action
    Action --> Source[KnowledgeSource registry]
    Source --> Dataset[Alignment dataset]
    Dataset --> Pipeline[Scorer and selector pipeline]
    Pipeline --> Writers[Writer registry]
    Writers --> Run[RunLayout + manifest]
    Pipeline --> Store[ExplanationStore]
    Store --> Run
    Run --> Reader[RunReader]
    Reader --> Inspect[Analysis / exact-inspect]
```

## Stable seams

- `KnowledgeSource` hides OWL, RDF, and CSV-KG parser details.
- Component registries resolve datasets, models, trainers, and evaluators by stable names.
- Source/writer/track/reasoner entry points allow external plugins without importing them at
  core startup.
- `RunLayout`, `ExplanationStore`, and `RunReader` isolate artifact versions from producers and
  consumers.
- Plain `run_alignment` and `run_evaluation` functions are the action boundary used by CLI and
  Python wrappers.

Import-linter enforces the dependency direction. Core contracts do not import implementations
or delivery code; implementation does not import analysis/delivery; ontology and I/O keep
backend dependencies localized. `exact_inspect` depends on Exact, never the reverse.

## Shared OWL stack

```mermaid
flowchart LR
    Bytes[OWL bytes / resolver] --> Core[pyowl-core snapshot]
    Core --> Facade[OwlOntologySource]
    Core --> Views[Shared structural views]
    Core --> Projector[Shared OWL2Vec* projector]
    Core --> Reasoner[Asserted / optional reasoner adapter]
    Views --> Dataset[Alignment dataset]
    Projector --> Dataset
    Reasoner --> Dataset
    Facade --> Dataset
```

The core snapshot is immutable and owned once by `OwlOntologySource`. All downstream
consumers retain or query that identity; Exact does not reparse a path or materialize a second
ontology representation. Optional process isolation encodes the snapshot once with the
versioned core wire format, and a worker opens and verifies those bytes before reasoning.
Projection and reasoner adapters convert only bounded returned rows into Exact entities.

## Reproducibility

Normalized source hashes, resolved config fingerprints, dataset-track pins, checkpoints,
timing sessions, evaluator provenance, and deliverable checksums meet in `run_manifest.json`
and `stats/run_stats.json`. Optional integrations fail explicitly when unavailable rather than
silently changing the core algorithm.

For OWL sources, `ontology_stack` adds package/API/schema versions, three semantic
fingerprints, closure/resolution and source-document hashes, projector profile/options/backend,
reasoner selection, diagnostics, and verified-wire status. The serializer intentionally omits
machine paths, object IDs, temporary locations, and credentials.
