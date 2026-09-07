# 03 — Focused verification and NCIT–DOID acceptance

The release has two verification layers: hermetic compatibility tests and one external-data
NCIT–DOID run. Timing benchmarks and experimental evaluation are explicitly excluded.

## 1. Hermetic compatibility suite

The normal Python 3.10–3.12 CI matrix remains blocking. It must cover these core migration
properties with small committed fixtures.

### Loading and ownership

- path, bytes, and supported stream inputs call core loading exactly once;
- construction from an existing view/provider preserves object identity;
- source, projector, and selected in-process reasoner receive the same retained owner;
- projection/reasoning does not mutate fingerprints, axiom counts, or public lazy views; and
- worker mode uses one verified core wire/mmap boundary and performs zero OWL parses in the worker.

### Schema-2 behavior

- Exact expects the installed public model and encoded schema rather than schema 1;
- the expected descriptor comes from `EncodedStructuralView.DESCRIPTOR_SHA256`;
- a schema-2/v1-digest hybrid contract is rejected;
- anonymous-individual scopes and repeated isomorphic components follow core 0.2 identities;
- strict RDF mapping failures propagate as structured errors; and
- partial diagnostic RDF documents never enter a snapshot, cache, wire payload, projector, or
  reasoner.

### Consumers and fallbacks

- projector Python and published native paths emit identical canonical edges;
- asserted mode works with no reasoner distributions installed;
- ELK and HermiT accept the retained view and return the expected fixture hierarchy;
- mmap workers retain their mapped owner and match in-process results;
- scalar-only/pure-wheel operation remains correct;
- missing optional consumers fail actionably; and
- corrupt or incompatible encoded capabilities propagate a public protocol error without path
  retry, Java fallback, or partial cache write.

### Caches and provenance

- schema-1 ontology-derived caches are rejected before interpretation and rebuilt;
- schema-2 cold and cache-hit results have identical semantic digests;
- cache keys change with the public core/projector/reasoner contract;
- completed historical run artifacts remain readable but cannot resume from schema-1 ontology
  caches; and
- manifests contain public package/schema/backend values and no object IDs or temporary paths.

## 2. Normal engineering commands

Run the repository's blocking quality commands on Python 3.10–3.12:

```bash
poetry install --with prebuild --extras "viz reasoning"
poetry run black --check exact exact_inspect tests tools study_visualizer_runtime
poetry run isort --check-only exact exact_inspect tests tools study_visualizer_runtime
poetry run flake8 exact exact_inspect tests tools study_visualizer_runtime
poetry run mypy
poetry run lint-imports
poetry run interrogate exact/delivery/api exact/ontology exact/io exact/core/contracts \
  exact/utils/timing.py exact/utils/candidate_generation.py exact/utils/graph_search.py
poetry run pytest \
  -m "not requires_data and not slow and not requires_cuda and not requires_openrouter" \
  --cov=exact
```

Also run:

```bash
poetry install --extras docs
poetry run mkdocs build --strict
poetry run linkchecker --no-warnings --ignore-url='^https?://' site/index.html
```

`benchmarks/bench.py --check-reference` is not part of this acceptance suite.

## 3. Frozen NCIT–DOID input

Use exactly the two files identified by the existing content-addressed baseline:

| Side | Bytes | SHA-256 |
|---|---:|---|
| NCIT source | 57,163,710 | `379a37f47c0c8e7c30397769358cca955140d16b2797a1cc75da4b1fc2b354eb` |
| DOID target | 6,687,536 | `76f41cce3616ad1a9ba6353f469e96bde7addba5d43e541651a3ab703f9ba2bc` |

A different digest is a different acceptance input and must not silently replace this gate. The
ontology files remain outside Git; only hashes and the small result record are committed.

## 4. Required NCIT–DOID run

Run the published native core/projector path in one cold process:

```bash
poetry run python benchmarks/owl_stack_scale.py \
  data/bioml_zenodo/ncit-doid/source.owl \
  data/bioml_zenodo/ncit-doid/target.owl \
  --load-backend native \
  --projector-backend native \
  --require-encoded-consumers \
  --output release/evidence/pyowl-core-0.2-ncit-doid.json
```

Run optional ELK/HermiT behavior on committed fixtures, not on NCIT–DOID. The large-data gate is
intended to validate core ownership and projection without turning release into a reasoner or
matcher benchmark.

## 5. NCIT–DOID acceptance assertions

The JSON record passes only when all of the following hold:

1. installed Exact/core/projector versions are final published/candidate release artifacts being
   evaluated, with final acceptance repeated against published dependencies;
2. both input sizes and SHA-256 digests match §3;
3. each ontology has exactly one load/closure resolution;
4. source and projector retain the original owner identity;
5. selected ingestion is `encoded-native` and the fail-closed encoded-consumer acceptance block
   passes;
6. parser, resolver, core-wire, scalar-materialization, base-flattening, structural-copy,
   per-row-FFI, and direct-staging-copy counters remain at their allowed zero values;
7. fingerprint/axiom metadata is unchanged after projection;
8. Exact reports no second ontology-sized representation;
9. cache fill and cache hit produce the same canonical projection digest;
10. projection semantics match the already classified profile result:
    - NCIT: 42,103 canonical unique edges;
    - DOID: 9,388 canonical unique edges;
    - the historical NCIT delta remains exactly 762 RB-019 additions and eight RB-009 removals,
      with zero unexplained residual edges; and
11. the process completes within the ordinary release-job timeout without runner exhaustion.

If core 0.2 intentionally changes an edge set, do not update the expected count automatically.
Capture the complete sorted set difference, classify every edge by a documented semantic rule,
review it, and amend this specification before accepting the new digest.

## 6. Measurements that do not affect acceptance

The evidence may record load time, publication time, projection time, throughput, cache-hit time,
CPU, and RSS. These fields are diagnostic only. Do not compare them with the old Exact 2.0
measurements as a release decision.

The release does not require:

- the fixture timing reference check;
- Conference, GO, another Bio-ML pair, or a licensed workflow;
- a 25% timing threshold;
- a pinned-runner RSS baseline;
- native reasoner performance on NCIT–DOID;
- CUDA or hosted LLM access; or
- any experiment or Exact Repair result.
