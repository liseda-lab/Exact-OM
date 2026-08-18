# Correctness and diagnostic measurements

Exact-OM 2.1 makes no performance claim. Fixture and scale timings may be collected for
diagnosis, but timing comparisons are not release gates.

## Fixture diagnostics

`benchmarks/bench.py` measures committed, license-safe fixtures. CI runs it without the
historical reference threshold and uploads the JSON:

```console
poetry run python benchmarks/bench.py \
  --repeat 7 \
  --output benchmark-results/fixture.json
```

The scenarios cover one-shot snapshot loading, hierarchy views, transitive queries, delegated
OWL2Vec* projection, dataset evidence, candidate generation, CSV-KG loading, track
materialization, inference, and the explanation store. A functional failure still needs
investigation; a slower median alone does not block Exact 2.1.

The legacy `--check-reference` mode and `benchmarks/reference.json` remain available for
experiments. Their 25% comparison is not normal CI or release acceptance.

## Exact 2.1 external-data acceptance

The only external-data gate is the frozen NCIT–DOID correctness run. The inputs must match:

| Side | Bytes | SHA-256 |
| --- | ---: | --- |
| NCIT source | 57,163,710 | `379a37f47c0c8e7c30397769358cca955140d16b2797a1cc75da4b1fc2b354eb` |
| DOID target | 6,687,536 | `76f41cce3616ad1a9ba6353f469e96bde7addba5d43e541651a3ab703f9ba2bc` |

Run the released native core/projector path in one cold process:

```console
poetry run python benchmarks/owl_stack_scale.py \
  data/bioml_zenodo/ncit-doid/source.owl \
  data/bioml_zenodo/ncit-doid/target.owl \
  --load-backend native \
  --projector-backend native \
  --require-encoded-consumers \
  --output release/evidence/pyowl-core-0.2-ncit-doid.json
```

Acceptance requires one load per ontology, retained owner identity, encoded-native ingestion,
zero forbidden parse/wire/scalar/copy work, unchanged fingerprints and axiom counts, no second
ontology-sized representation, and identical cache-fill/cache-hit projection digests. The
classified projection result remains 42,103 NCIT edges and 9,388 DOID edges. The historical
NCIT delta is exactly 762 RB-019 additions and eight RB-009 removals, with no unexplained
residual edge.

The command records wall time, CPU time, throughput, cache timing, and RSS as diagnostic
fields. Those values are not compared with Exact 2.0 and do not affect acceptance unless the
ordinary release-job timeout or runner resources are exhausted.

Conference, GO, additional Bio-ML pairs, licensed workflows, CUDA, hosted LLMs, native
reasoner scale, pinned-runner RSS comparisons, Exact Repair, and experiments are outside the
2.1 release gate.
