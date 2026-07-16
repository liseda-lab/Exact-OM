# Extending Exact-OM

Prefer the narrowest stable protocol and keep optional imports inside the integration module.
The [contribution guide](../project/contributing.md) describes quality and compatibility gates.

## Knowledge source

Implement `KnowledgeSource` for labels, annotations, kind-aware entities, hierarchy, projection
edges, and exclusions. Register a local factory with `register_source`, or publish an
`exact.sources` entry point. Run the conformance suite against a small fixture.

```python
from exact.io.sources import register_source

register_source("my-format", create_source)
```

## Alignment writer

Implement `AlignmentWriter` with a stable name/default filename and deterministic `write`
method. Register locally with `register_writer`, or publish an `exact.writers` entry point.
Validate paths, finite scores, relation vocabulary, and ordering at the writer boundary.

## Dataset track

Prefer a declarative HTTP/Hugging Face descriptor when retrieval can be expressed as pinned
files, checksums, safe transforms, and task-layout globs. Use a Python `TrackProvider` plugin
for licensed/manual or specialized materialization. Providers register under `exact.tracks`.

## Reasoner

An `exact.reasoners` entry point receives the ontology store and returns the reasoner protocol.
Only the reasoner backend may depend on its optional library; datasets keep using
`KnowledgeSource` methods.

## Model, dataset, trainer, or evaluator

Subclass the applicable core contract, preserve a stable registry name, add the module to the
central bootstrap path, and test both registry resolution and behavior. Do not make core import
an implementation merely to trigger registration.

Third-party plugins should turn missing dependencies into an actionable installation error and
must not change global state at discovery time.
