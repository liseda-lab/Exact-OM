# Ontology backend baselines

The two compressed snapshots are genuine `mowl-borg==1.0.1` captures from the pre-removal
backend (`330044c`). Their environment, input hashes, output hashes, and the two explained ELK
inferences are recorded in `provenance.json` and enforced by `ontology_parity_test.py`.

To regenerate them, create an isolated Python 3.10 environment with a JDK, then install:

```bash
pip install mowl-borg==1.0.1 zstandard 'numpy<2' class-resolver==0.5.4
python tools/capture_backend_baseline.py --backend mowl --output-dir tests/baselines \
  tests/fixtures/ontologies/mini_src.owl tests/fixtures/ontologies/mini_tgt.owl
```

For real-data release checks, use the same command with the materialized Conference/Bio-ML
OWL files and `--output-dir exp/baselines`. That directory is intentionally gitignored because
licensed and large upstream ontologies must not be redistributed.

The reproducible Conference comparison, including input hashes, snapshot parity, fixed config,
global/local alignment hashes, metrics, and timings, is recorded under `benchmarks/evidence/`.
