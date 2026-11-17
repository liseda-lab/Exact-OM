# Exact-OM 2.0

Exact matches ontology entities using lexical and structural evidence, optional
gated language-model decisions, and fitted ranking or acceptance models. It
exports alignments, complete local rankings, and inspectable evidence.

The matcher supports Python 3.10–3.12; Exact Explain requires Python 3.11–3.12. Native OWL processing does not require Java.
GPU encoders and Elephant-1 are optional; the bundled demonstration runs offline
on CPU.

## Installation

From a checkout of this release:

```console
python -m pip install .
exact --help
```

For a built distribution, use `pip install exact_om-2.0.0-py3-none-any.whl`.
The package is prepared for PyPI publication; creating this branch does not
publish it. Optional integrations are `.[bioml-eval]`, `.[reasoning]` and `.[viz]`.
Elephant uses a separate, pinned serving environment described in
[the decision guide](docs/decisions.md).

## Offline demonstration

```console
exact config example --output demo
exact align -s demo/mini_src.owl -t demo/mini_tgt.owl \
  -c demo/mini_test.cands.tsv -y demo/config.yaml -o runs/demo
exact run info runs/demo
exact run export runs/demo --format jsonl --output evidence.jsonl
```

The demonstration uses string evidence to exercise the complete input/output
path without downloading models. Its accuracy is not representative of the
encoder-based research configuration.

## Production workflow

The final architecture has explicit stages so expensive pair evidence can be
reused when fitting or changing a selection policy.

| Command | Purpose |
|---|---|
| `exact score` | Generate source-sharded native pair scores and selected evidence |
| `exact decisions` | Freeze the source gate, run/reuse Elephant, and mix pair probabilities |
| `exact acceptance` | Fit a global edge classifier and export full or partial-policy alignments |
| `exact rank` | Fit and apply the local query ranker without changing candidate membership |
| `exact eval` / `exact-eval` | Evaluate saved submissions |
| `exact config` | Copy profiles/examples, print defaults, or migrate configuration |
| `exact data` | Retrieve and verify dataset tracks |
| `exact run` | Inspect, export, and clean standard alignment runs |
| `exact align` | Run the existing configurable matching pipeline and API-compatible workflow |
| `exact-inspect` | Prepare, open and share bundles in the local Exact Explain app |

Start with [the staged workflow](docs/pipeline.md). It distinguishes generated
global candidates from supplied local query pools and explains TRAIN-only
fitting, separate full/partial policies, checkpoints, and output formats.
`exact align` retains the configurable native workflow; it does not implicitly
load the final study's fitted acceptance models or apply an Elephant gate.

Production profiles preserve the selected encoders and numerical settings.
Task-specific models and thresholds must be fitted from designated training
data, qualified on validation data, and frozen before test evaluation.
The BioML full and partial results use different operating policies. NCIT–DOID
also retains a historical full-policy head; these details are explicit in the
[acceptance](docs/acceptance.md), [decisions](docs/decisions.md), and
[local ranking](docs/ranking.md) guides.

## Documentation and development

- [Installation and optional integrations](docs/installation.md)
- [Configuration and inputs](docs/configuration.md)
- [Evaluation](docs/evaluation.md)
- [Artifacts, storage, and resume](docs/runs.md)
- [Install and use Exact Explain locally](docs/local-tool.md)
- [Python API](docs/api.md)
- [Migration from Exact v1](MIGRATION.md)
- [Contributing and release checks](CONTRIBUTING.md)

Exact Explain provides candidate comparison, ontology browsing, evidence graphs,
decision traces, scores and a local bundle library. A release wheel includes the
built app; install it with the `viz` extra and run `exact-inspect serve`.
For a source checkout, build the frontend first as described in the
[local installation guide](docs/local-tool.md). The app accepts prepared bundles
and prepares both standard `exact align` runs and the production Exact 2.0
stages: native scores, applied decisions, global acceptance and local rankings.
Use the stage arguments in the guide to retain their evidence and receipt chain.

Ontology repair, participant/admin study services, private deployment scripts,
and historical experiments remain outside this production release.

## Citation and license

If you use Exact, cite the accompanying Exact ontology-matching paper and record
the package version, configuration, model revisions and evaluation protocol.
Licensed under the [MIT license](LICENSE). Dataset and model licenses apply
separately.
