# Installation

Exact-OM supports Python 3.10–3.12. A CPU is sufficient for development and small
fixtures; a CUDA-enabled PyTorch installation substantially accelerates embedding and scoring
on large tracks.

## Poetry development install

```console
git clone https://github.com/liseda-lab/Exact-OM.git
cd Exact-OM
poetry install
poetry run exact --help
```

Poetry installs the platform-appropriate PyPI build of PyTorch. To use a particular CUDA
wheel, create the environment first and then follow the matching command in the
[PyTorch installation selector](https://pytorch.org/get-started/locally/). For CUDA 12.8:

```console
poetry run pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<3"
```

Do not request a CUDA device on a CPU-only host; omitting `--device` selects CPU.

## Optional features

Release wheels expose independent extras so the matcher does not import service or dataset
dependencies it does not use.

| Extra | Command | Enables |
| --- | --- | --- |
| Java-free reasoning | `pip install "exact-om[reasoning]"` | Optional pyELK and pyHermiT hierarchy adapters. |
| Viewer | `pip install "exact-om[viz]"` | `exact-inspect`, FastAPI, and Uvicorn. |
| Hugging Face data | `pip install "exact-om[hf]"` | Hugging Face track providers. |
| Bio-ML evaluator | `pip install "exact-om[bioml-eval]"` | Compatible 0.2 evaluator plus Java-free official-coherence reasoners. |
| Documentation | `pip install "exact-om[docs]"` | MkDocs and reference generators. |

An integration selected without its extra exits with an installation hint. Hosted LLM access
is optional; all core tests and non-LLM matching paths run without OpenRouter credentials.
The Bio-ML 0.2 evaluator and its reasoner extra support Python 3.10 and newer. The builtin
evaluator remains the lighter option when official coherence is not requested.

The base wheel includes the shared `pyowl-core` snapshot API and OWL2Vec* projector. It does
not require Java, a JDK, Cargo, a compiler, or either optional reasoner. Native accelerators
are selected only when a compatible upstream wheel is already installed; the complete Python
backend remains available on Python 3.10–3.12.

## Verify the installation

```console
poetry run exact data list
poetry run exact config default --format yaml > /tmp/exact-default.yaml
poetry run exact run --help
```

For a source checkout, the standard CPU-only check is:

```console
poetry run pytest \
  -m "not requires_data and not slow and not requires_cuda and not requires_openrouter"
```
