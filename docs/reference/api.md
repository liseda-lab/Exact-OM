# Python API

Use Exact's public ontology facade when an integration needs the shared OWL owner:

```python
from exact.ontology import load_ontology

source = load_ontology("source.owl")
owner = source.owl_snapshot()

source.configure_projector(backend="python")
edges = source.projection_edges()

assert source.projector.last_view is owner
assert source.reasoner.ontology is owner
```

`load_ontology` accepts a path, bytes, a supported binary stream, an existing public
`pyowl_core.OntologyView`, or a public snapshot provider. Existing views/providers are
retained rather than serialized or reparsed. Optional reasoners are selected with
`source.configure_reasoner("elk")` or `source.configure_reasoner("hermit")` after installing
`exact-om[reasoning]`.

Do not pass origin paths to consumers, request or decode encoded structural buffers, or import
core/projector/reasoner implementation modules. Exact negotiates capabilities through public
attributes and records the selected owner, ingestion path, schemas, and backend in provenance.

The following pages are rendered from live docstrings.

## Delivery API

::: exact.delivery.api

## Ontology and knowledge sources

::: exact.ontology

::: exact.ontology.projection

::: exact.ontology.reasoning

::: exact.ontology.provenance

::: exact.core.contracts.knowledge

## I/O registries

::: exact.io.sources

::: exact.io.writers

## Core contracts

::: exact.core.contracts.dataset

::: exact.core.contracts.model

::: exact.core.contracts.trainer

## Run artifacts

::: exact.runs
