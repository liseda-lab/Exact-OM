# Benchmarks

`benchmarks/bench.py` measures the committed, license-safe ontology fixtures and checks median
runtime against `benchmarks/reference.json`:

```console
poetry run python benchmarks/bench.py --repeat 7 --check-reference
```

The OWL acceptance scenarios cover one-shot snapshot loading, shared hierarchy views,
transitive queries, delegated OWL2Vec* projection, end-to-end dataset evidence, and candidate
generation. Additional scenarios cover CSV-KG loading, track materialization, inference, and
the explanation store. The 25% fixture tolerance catches coarse regressions without treating
microbenchmark noise as a release failure.

For scale evidence, run selected scenarios with `--output result.json` in the target deployment
and retain the JSON beside the release artifacts. A benchmark implementation must pass the
same `OntologySnapshot` to `OwlOntologySource`, projector, and reasoner. Adding a parsed graph,
path reparse, or unbounded adapter conversion to make a benchmark faster violates the
architecture even if wall time improves.

Peak-memory and throughput claims for optional native upstream engines belong to those
projects' release evidence. Exact's base acceptance proves that no JDK, Cargo invocation,
reasoner distribution, or second ontology representation is needed.

For legally available conference/Bio-ML/GO/NCIT inputs, capture the full WP-M scale record:

```console
poetry run python benchmarks/owl_stack_scale.py source.owl target.owl \
  --output owl-stack-scale.json
```

The record includes exact load count, load wall time, time to first projected edge, projection
throughput, peak/incremental RSS, core document-cache hits, Exact projection-cache fill/hit
timing, spill counters, fingerprint immutability, and source/projector/reasoner snapshot
identity. Input bytes are not redistributed, so licensed corpora can contribute release
evidence without entering the repository.

The current hash-matched NCIT–DOID candidate is committed under
`benchmarks/evidence/wp_m_ncit_doid_candidate.json`. It passes one-load, shared-identity, and
no-second-representation checks, but it is not release acceptance: both loading and projection
exceed the 25% wall-time limit, and the NCIT projection has an unclassified 754-edge delta from
the frozen private-stack result. WP-M M5 and the Exact 2.1.0 version bump remain blocked until
those differences are resolved and a comparable pinned-runner RSS result is reviewed.
