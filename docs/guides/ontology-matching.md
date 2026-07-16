# Ontology matching (OAEI Bio-ML)

Bio-ML tasks can be selected through a track descriptor or passed as explicit files. A local
candidate file switches Exact-OM from global candidate generation to source-local ranking.

```console
exact data pull bioml_hf/ncit-doid --root data
exact align -o runs/ncit-doid -y config.yaml -e -l
```

Select the materialized task and keep the input adapter automatic:

```yaml
config_version: 2
data:
  track: bioml_hf
  task: ncit-doid
  root: data
io:
  input_format: auto
```

The ontology backend normalizes equivalence components, hierarchy edges, labels, annotations,
restrictions, property metadata, and exclusions behind `KnowledgeSource`.

## Evaluation backends

The built-in evaluator is deterministic and always available. On Python 3.12, add `bioml` to
`evaluation.backends` after installing `exact-om[bioml-eval]` to run the pinned upstream
Bio-ML evaluator as well. Exact writes both results into the canonical `evaluation/` directory
and records evaluator provenance in run statistics.

Training mappings are treated as null mappings where the track requires it. Local evaluation
uses the candidate ranking and reports MRR/Hits@K; global evaluation reports precision,
recall, and F1. Missing optional upstream evaluation fails only that backend unless strict
error behavior is requested.
