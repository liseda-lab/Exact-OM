# Exact-OM

Exact-OM is an explainable matcher for ontologies and knowledge graphs. It combines lexical,
structural, and optional language-model evidence, then records the evidence behind every
scored pair in an inspectable, versioned run.

It supports OWL and RDF sources, descriptor-driven CSV knowledge graphs, class/property/
individual matching, global alignment and local ranking, reproducible dataset tracks, multiple
output formats, and built-in or OAEI Bio-ML evaluation.

## Install

Exact-OM supports Python 3.10–3.12 and runs on CPU. A CUDA-enabled PyTorch installation is
recommended for large embedding workloads.

```console
git clone https://github.com/liseda-lab/Exact-OM.git
cd Exact-OM
poetry install
poetry run exact --help
```

For CUDA 12.8 wheels, create the Poetry environment first, then install the matching PyTorch
build explicitly:

```console
poetry run pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<3"
```

Optional extras enable Java-free EL/DL reasoning (`reasoning`), the viewer (`viz`), Hugging
Face track retrieval (`hf`), the upstream Bio-ML evaluator (`bioml-eval`), and documentation
(`docs`). The base install includes the shared OWL snapshot and projector, but no reasoner,
JDK, Cargo toolchain, or native build step. See the
[installation guide](https://liseda-lab.github.io/Exact-OM/getting-started/installation/).

## CPU quickstart

This fixture run uses a tiny test encoder and makes no hosted LLM request. Its first execution
downloads that small encoder from Hugging Face.

```console
poetry run exact align \
  -s tests/fixtures/ontologies/mini_src.owl \
  -t tests/fixtures/ontologies/mini_tgt.owl \
  -c tests/fixtures/ontologies/mini_test.cands.tsv \
  -f tests/fixtures/ontologies/mini_refs.tsv \
  -y examples/quickstart.yaml \
  -o runs/quickstart -e -l
poetry run exact run info runs/quickstart
```

The run writes `alignment/maps_local.tsv`, evaluation and statistics under their named
directories, source-indexed explanations, `timings.json`, the resolved `config.yaml`, and
`run_manifest.json`. OWL runs also record a path-free `ontology_stack` block with core,
projector, reasoner, import-closure, diagnostic, and semantic-fingerprint provenance.

## Documentation

- [Quickstart and expected outputs](https://liseda-lab.github.io/Exact-OM/getting-started/quickstart/)
- [Datasets and tracks](https://liseda-lab.github.io/Exact-OM/guides/datasets-tracks/)
- [Ontology and KG matching guides](https://liseda-lab.github.io/Exact-OM/guides/ontology-matching/)
- [Configuration reference](https://liseda-lab.github.io/Exact-OM/reference/configuration/)
- [CLI reference](https://liseda-lab.github.io/Exact-OM/reference/cli/)
- [Migration through 2.1](MIGRATION.md) and [contributing](CONTRIBUTING.md)

For Python integrations, use `exact.delivery.api.AlignmentRunner` and `EvaluationRunner`, or
the functional `run_alignment(...)` and `run_evaluation(...)` action APIs.

## Citation

If you use Exact-OM in academic work, cite:

> Cotovio, P. G., Nunes, S., Jiménez-Ruiz, E., & Pesquita, C. (2026).
> *Interpretable Context-Aware Models Improve Expert Validation in Ontology Matching*.
> The Semantic Web: ESWC 2026. Springer.

## License

Exact-OM is distributed under the [MIT License](LICENSE).
