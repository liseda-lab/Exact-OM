# Installation

Use Python 3.10–3.12 in a virtual environment:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For CPU-only machines, install the CPU PyTorch wheel from PyTorch's official
index before installing Exact. For NVIDIA GPUs, install a PyTorch build
compatible with the local driver. Exact does not change the cluster driver or
CUDA installation.

The base package includes native OWL processing, encoder inference, Parquet
checkpoints, local ranking, acceptance, explanations and standard evaluation.
No JVM is required. Optional integrations:

```console
python -m pip install '.[bioml-eval]'
python -m pip install '.[reasoning]'
python -m pip install '.[viz]'
python -m pip install '.[dev,docs]'
```

The first encoder run downloads the configured pinned weights unless they are
already cached. Set a writable Hugging Face cache on active storage; keep live
models and checkpoints out of archival/cold storage. Local model directories
are supported and their contents are hashed by the staged scorer.

Elephant-1 uses a distinct noncausal decision runtime, not the generic
Hugging Face causal-generation adapter. Follow the pinned separate-environment
instructions in [decisions](decisions.md). Preparation, cache-only replay and
application work without installing that runtime. Its optional GPU backend is
not loaded by ordinary package imports.

For restricted/offline environments, pre-download the pinned encoders and
configure their local directories. `exact config example --output demo`
provides an entirely offline string-evidence example.

For the browser app, use Python 3.11–3.12 and follow [the local explorer guide](local-tool.md). Release
wheels include its built pages; source checkouts require a Node.js build once.
Node.js is not needed to run an installed wheel.
