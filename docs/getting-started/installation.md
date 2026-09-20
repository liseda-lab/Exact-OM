# Installation

Exact-OM 2.1 supports Python 3.10–3.12. A CPU is sufficient for development and small
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

## Shared ontology package contract

The base distribution installs the released Java-free shared stack:

- `pyowl-core>=0.2.1,<0.3`;
- `pyowl2vec-star-projector>=0.2.1,<0.3`;
- `pyelk-reasoner>=0.2.1,<0.3` and `pyhermit>=0.2.1,<0.3` only with `reasoning`; and
- `oaei-bioml-eval>=0.2.1,<0.3` only with `bioml-eval`.

The current native package set is recorded under `published_native_stack` in
`release/core-compatibility.json`; the historical baseline and local validation records are
preserved separately. The current set uses PyPI 0.2.1 wheels for all four ontology packages.
Editable installs, source checkouts, development builds, and release candidates are not
published-wheel evidence.

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
The built-in evaluator remains the lighter option when official Bio-ML coherence is not
requested.

The base wheel includes the shared `pyowl-core` snapshot API and OWL2Vec* projector. It does
not require Java, a JDK, Cargo, a compiler, visualization services, or either optional
reasoner. OWL loading and projection require compatible native upstream wheels. Exact
checks native capabilities and execution reports and does not fall back to Python processing.

## Verify the installation

```console
python -m pip check
exact data list
exact config default --format yaml > /tmp/exact-default.yaml
exact run --help
```

For a source checkout, the standard CPU-only check is:

```console
poetry run pytest \
  -m "not requires_data and not slow and not requires_cuda and not requires_openrouter"
```
