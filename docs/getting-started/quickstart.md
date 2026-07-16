# Quickstart

The repository includes paired mini ontologies, a candidate ranking, a reference alignment,
and a CPU-oriented configuration. The primary scorer uses a tiny test encoder and disables
hosted and local generative models, so no OpenRouter key or CUDA device is required. The first
run downloads the small encoder from Hugging Face.

From the repository root:

```console
poetry install
poetry run exact align \
  -s tests/fixtures/ontologies/mini_src.owl \
  -t tests/fixtures/ontologies/mini_tgt.owl \
  -c tests/fixtures/ontologies/mini_test.cands.tsv \
  -f tests/fixtures/ontologies/mini_refs.tsv \
  -y examples/quickstart.yaml \
  -o runs/quickstart -e -l
poetry run exact run info runs/quickstart
```

Passing a candidate file selects local-ranking mode: every listed target remains in the saved
ranking. Remove `-c` for global candidate generation after choosing a production candidate
encoder in the config.

## Expected output

The final command reports layout v2 and an artifact summary. The run directory contains:

```text
runs/quickstart/
├── alignment/maps_local.tsv
├── evaluation/evaluation_results.csv
├── explanations/index.json
├── explanations/shards/
├── stats/
├── plots/
├── config.yaml
├── timings.json
├── exact.log
└── run_manifest.json
```

Inspect a few results and generate an explanation export:

```console
head runs/quickstart/alignment/maps_local.tsv
poetry run exact run export runs/quickstart --what explanations --format json
```

Next, read [Runs, caching, and resume](../guides/runs-caching-resume.md), choose an
[ontology](../guides/ontology-matching.md) or [KG](../guides/kg-matching.md) workflow, and
replace the tiny encoder with the models appropriate for your task.
