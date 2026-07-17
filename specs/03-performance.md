# 03 — Performance Policy (cross-cutting)

Applies to every WP. Goal: the Java-free system must be **at least as fast** as the mowl/JVM
baseline on every stage, with headroom engineered the PyLogMap way — pure-Python correctness
first, compiled kernels for measured hot spots, never speculation. (Reference point: PyLogMap's
C kernels run the Anatomy pair in 10.3 s vs 11.1 s for Java LogMap; its forced-pure-Python mode
is ~11× slower but bit-identical — that dual-mode discipline is what we adopt.)

## Principles

1. **Measure before optimizing.** No compiled code lands without a benchmark showing the
   pure-Python version is a bottleneck on realistic data (conference-scale for latency,
   NCIT/SNOMED-scale for throughput/memory).
2. **Every compiled kernel has a bit-identical pure-Python fallback**, selected automatically at
   import (`exact/*/_kernels.py` dispatcher pattern; `EXACT_FORCE_PYTHON_KERNELS=1` env override
   for testing). CI runs the test suite in both modes for any module that gains a kernel.
3. **Cython is the sanctioned compiled path** (already a dependency; build via Poetry's
   build-script hook, wheels must still install from sdist on machines without a compiler —
   fallback covers that). No hand-written C extensions, no new native deps without discussion.
4. **Vectorize first.** numpy/torch batch operations and int-indexed data structures usually
   beat a kernel: intern IRIs to `int` ids at the `exact/ontology` boundary, store hierarchies
   and adjacency as int arrays/CSR, keep DataFrame ops columnar (no `.apply` row loops in hot
   paths).
5. **Cache once, index once.** No per-entity re-scans of axiom lists (the mowl-era code
   re-queried `EntitySearcher` per class — the audit's known sin; WP-B's store indexes on
   construction). Embedding/LLM caches keep their existing policies.

For WP-M, `pyowl-core` owns parsing, structural indexes, IRI interning, and lazy OWL views;
`pyowl2vec-star-projector` owns its Rust/PyO3 acceleration and complete Python fallback. The
older Cython rule applies only to measured Exact-owned matcher kernels. Exact must not fork those
external engines or convert a snapshot into local int/string tables merely to optimize them.

## Benchmark harness (landed by WP-B, extended by later WPs)

- `benchmarks/bench.py` — a small runner (pytest-benchmark or hand-rolled timer, agent's choice)
  with named scenarios, JSON results to `exp/benchmarks/<git-sha>.json`:
  - `parse_ontology` (fixture / conference / NCIT-scale if available locally)
  - `build_hierarchy` + `transitive_closure`
  - `projection_owl2vecstar`
  - `shared_snapshot_handoff` (WP-M: one load, object identity preserved across source,
    projector, and optional reasoner; parser-call count remains one)
  - `dataset_build_e2e` (conference pair, LLM fakes)
  - `candidate_generation` (lexical index + retrieval)
  - `inference_throughput` (fixture, CPU, fake LLM — examples/s)
  - added later: `csv_kg_load` (WP-G), `track_materialize` (WP-I, network-marked)
- CI runs the fixture-scale scenarios and **fails on >25% regression** against the committed
  reference JSON (`benchmarks/reference.json`, updated deliberately in PRs that change perf).
  Large-scale scenarios are `requires_data`/manual, results pasted into PR descriptions.

## Known hot spots & their owners

| Hot spot | Owner | Plan |
|---|---|---|
| Ontology parse + structural index | WP-M / pyowl-core | one `load_snapshot`, shared lazy views; no Exact structural copy; compare with WP-B baseline |
| OWL hierarchy closure / classification | WP-M / core or optional pyELK/pyHermiT | consume same snapshot identity; never path-reparse; benchmark separately from load |
| OWL2Vec*-style projection | WP-M / shared projector | pinned profile; native/Python parity; bounded-memory streaming owned upstream |
| Lexical candidate index (TF-IDF token/char-grams) | WP-F (touches) | keep sklearn/scipy sparse ops; per-kind indexes avoid inflating the vocab |
| top-k cosine search | — (already `topk_cosine_search_torch`) | leave; benchmark guards it |
| Best-path context algorithms (`utils/graph_search.py`) | WP-D (move only) | legacy path; benchmark, don't optimize unless it shows up |
| Checkpoint/audit shard I/O | WP-D | already zstd streaming; keep incremental writes |
| CSV-KG load (BioKG scale) | WP-G | pandas columnar load → int-id edges; budget: ≤30 s for the largest published BioKG pair on a laptop |

## Memory

- IRI interning and OWL structural storage are shared by the one `OntologySnapshot`; Exact may
  add only feature/candidate indexes. WP-M reports incremental RSS for source facade, projector,
  and reasoner separately and fails if a second ontology-sized representation appears.
- Streaming writers for anything per-candidate (already the audit-shard pattern); no
  materializing all explanations in memory (preserved from current design).

## What NOT to do

- No GPU requirements for non-model code paths.
- No premature parallelism: `num_workers` unfreezing (post-JVM it's safe again) is a measured
  follow-up, not a default flip inside another WP.
- No dependency additions for speed (numba, polars, …) without a benchmark + discussion note in
  the PR.
