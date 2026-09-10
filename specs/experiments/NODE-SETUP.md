# Prepared research node

Observed 2026-09-10: Python 3.12, NVIDIA RTX 5090 (32,607 MiB), driver 570.211.01,
Intel i9-9900K (8 cores / 16 threads), approximately 62 GiB RAM. This is a hardware
inventory, not a campaign throughput forecast.

The working CUDA stack is pinned in
[experiments-rtx5090.constraints.txt](../../deploy/experiments-rtx5090.constraints.txt).
The repository's Poetry lock currently resolves a different Torch version; the constraints
are an explicit node override. A fresh environment avoids the duplicate stale distribution
metadata found in the existing environment:

```console
python3.12 -m venv .venv-experiments
.venv-experiments/bin/python -m pip install --upgrade pip
.venv-experiments/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c deploy/experiments-rtx5090.constraints.txt -e '.[bioml-eval,hf]'
.venv-experiments/bin/python -m pip check
.venv-experiments/bin/python -c 'import torch; from sentence_transformers import SentenceTransformer; print(torch.__version__, torch.cuda.is_available()); print(torch.ones(2, device="cuda").sum().item())'
```

The existing `.venv` passes CUDA tensor execution and SentenceTransformer import. Its
unrelated installed extras still have dependency conflicts, so it is not claimed to pass
`pip check`. The fresh-environment commands above are a reproducible installation recipe;
a fresh installation has not been performed during this implementation pass.

Admit one heavy GPU job. The campaign defaults to two CPU workers and four concurrent
OpenRouter requests; start with two numerical CPU threads per worker to avoid multiplying
BLAS/OpenMP thread pools. LogMap is separately bounded to four Java processors, 24 GiB heap,
and a two-hour timeout. Its forecast must still pass the same node budget admission.

```console
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
```

`tools/prepare_experiment_locks.py` resolves cached immutable model revisions without
loading models or making hosted calls. Keep downloaded corpora, model caches, credentials,
and generated campaign artifacts outside Git. Model and input manifests identify bytes;
paths are relocatable bindings.

All generative roles use the pinned OpenRouter profile. The updated repository `api_key`
file passes actual binary/listwise probability and relocated-cache checks; pass
`--api-key-file api_key` to validation tools. They load the value into the process environment
only. The immutable baseline still has the older key-file locator; do not assume that locator
contains the updated key. Credentials are never written to campaign or source files.

The four-source NCIT–DOID operational proof demonstrates checkpoint interruption, relocation,
and byte-identical replay with no repeated encodings. It does not estimate full-ontology
memory/throughput or replace the prescribed G0 probe and 300-source vertical acceptance.
No long experimental campaign was launched during preparation.
