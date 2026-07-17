# WP-B — Java-Free Ontology Backend (`exact/ontology/`)

**Depends on**: WP-A. **Blocks**: WP-F, WP-G, final WP-E wiring. **Size**: XL (1 agent, 3 stacked
PRs at the checkpoints marked ✋).
**Behavior**: numerically parity-gated, not byte-identical (asserted hierarchy replaces ELK — see B6).
**Status**: Done (2026-07-16).

> **2.1 supersession:** This file records the completed 2.0 Java-removal implementation and its
> parity evidence. `WP-M-shared-owl-stack.md` supersedes its ownership of structural OWL records,
> parsing/normalization, OWL hierarchy storage, and projection rules. The baselines and observed
> mOWL quirks here remain mandatory migration oracles.

## Context

`mowl-borg` is the **only** Java-pulling dependency (pyproject line 25; brings OWL-API, ELK,
HermiT JARs + JPype). The audit found:

- The default pipeline (`PairAdaptiveSemanticScorer`) consults the reasoner in exactly one place:
  `_direct_superclass_iris` (`exact/impl/datasets/pair_adaptive_context.py:129-142`) —
  `reasoner.getSuperClasses(cls, direct=True)`. The reasoner is built ELK-first with only
  `CLASS_HIERARCHY` precomputed, falling back to a **structural** (asserted-only) reasoner on
  failure (`exact/core/contracts/dataset.py:160-213`), with a 120 s timeout that silently yields
  `None`. So current results already tolerate asserted-only hierarchies.
- Everything else is asserted axioms + annotations: OWL2Vec*/taxonomy projection
  (`exact/core/entities/ontology.py:151-189`), labels/synonyms/attributes via `EntitySearcher`,
  `part_of`/`has_part` via existential-restriction walking
  (`pair_adaptive_context.py:234-282`), `use_in_alignment` filtering (`dataset.py:345-408`,
  `utils/eval.py:37-69`), domains/ranges (`ontology.py:231-276`).
- JVM plumbing pollutes every entry point (`init_jvm` + heap flags in all CLIs/APIs/tools) and
  forces `num_workers: 0` ("avoids fork stalls after JVM init", `default_config.yaml:167`).

Strategy (per PyLogMap): parse with **py-horned-owl** (Rust, no JVM), index into plain-Python
structures, replace "reasoning" with an asserted-hierarchy index, reimplement the OWL2Vec*-style
projector in Python, and leave a `ReasonerProtocol` plugin seam for future DL reasoning packages.
Models, trainer, actions, and `utils/candidate_generation.py` are already Java-free and must not
change.

## Deliverable 1 — baseline capture (do this FIRST, while mowl still works)

1. `tools/capture_backend_baseline.py`: given a dataset pair dir, runs the current (mowl) code
   and dumps, per ontology: sorted class IRIs; labels map; per-class annotation bundles;
   per-class `hierarchy_bundle` (is_a / part_of / has_part); projection edge list (sorted
   3-tuples) for owl2vecstar (±literals) and taxonomy; `excluded_from_alignment` set; property
   domains/ranges — as one zstd JSON per ontology.
2. Run it on: the two **fixture ontologies** (B2), one **conference pair** (`cmt–conference`
   via `data/get_data.py`), and one **Bio-ML task** you have locally (e.g. `omim-ordo`). Commit
   fixture baselines under `tests/baselines/`; conference/Bio-ML baselines go in a gitignored
   `exp/baselines/` (document how to regenerate).
3. Also record an end-to-end run: alignment outputs + `evaluation_results.csv` for the
   conference pair with a fixed config/seed (this is the parity target for B8).

Prerequisites for this step: a machine with a JDK (mowl must still run to capture baselines)
and a config that makes the e2e run deterministic — LLM routing set to the `none`/local-fake
backends and fixed seed, so parity diffs measure the backend swap, not LLM nondeterminism.
Record the exact config used next to the baselines. **Decision (owner, 2026-07-16): the
implementing agent runs the capture itself** — install a JDK and download Conference data as
needed; escalate to the owner only if the environment genuinely cannot provide Java or network.

✋ **PR-B1**: baseline tool + fixtures + captured fixture baselines. No production changes.

## Deliverable 2 — fixture ontologies (`tests/fixtures/ontologies/`)

Hand-write `mini_src.owl` and `mini_tgt.owl` (RDF/XML, ~30 classes each) that together exercise
every capability: rdfs:label (multi-lang), `oboInOwl:hasExactSynonym`, `IAO:0000115` definitions,
multi-parent SubClassOf chains, `EquivalentClasses` with `ObjectIntersectionOf(named,
ObjectSomeValuesFrom(BFO:0000050, named))`, direct `ObjectSomeValuesFrom` subclass restrictions,
2 object properties (with domain/range + subPropertyOf), 1 data property, 3 named individuals
(types + object/data assertions — needed by WP-F), literal annotations, one class with
`use_in_alignment=false` (`exact/core/values.py:7` IRI), one `owl:deprecated=true` class, one
class with no label (short-form fallback). Plus `mini_refs.tsv` and `mini_test.cands.tsv` in
Bio-ML format. Document the inventory in `tests/fixtures/ontologies/README.md`.

## Deliverable 3 — `exact/ontology/` package

Add dependency `py-horned-owl` (pin the latest stable; verify the real API of the pinned version
before coding — do not code against remembered APIs). `parser.py` is the **only** module allowed
to import `pyhornedowl`; if its API shifts, only that file changes.

- `parser.py` — `parse(path: Path) -> ParsedOntology`: loads RDF/XML and OWL/XML (+ functional
  syntax if the pinned version supports it; `.obo` is out of scope — Bio-ML ships RDF/XML) and
  normalizes into backend-neutral records: axiom lists (SubClassOf, EquivalentClasses,
  AnnotationAssertion, Domain/Range, SubObjectPropertyOf, ClassAssertion,
  ObjectPropertyAssertion, DataPropertyAssertion), signature per `EntityKind`, ontology IRI.
- `expressions.py` — walkers over parsed class expressions: `named_class_iri(expr)`,
  `existential_targets(expr, property_iris)`, `intersection_operands(expr)` — mirrors
  `pair_adaptive_context.py:163-232` semantics.
- `store.py` — `OwlOntologySource(KnowledgeSource)` (see `02-shared-contracts.md` §4): builds
  indexes once (annotations by subject, subclass axioms by subclass, equivalences by member,
  domains/ranges by property, label map with configurable label properties defaulting to
  rdfs:label). Implements `excluded_from_alignment` (use_in_alignment=false ∪ deprecated),
  `attributes`, `hierarchy_bundle` (is_a via `hierarchy.py`; other families via
  `expressions.existential_targets` over SubClassOf + EquivalentClasses conjuncts — exact port
  of `_hierarchy_axiom_targets`).
- `hierarchy.py` — asserted hierarchy index: named-to-named SubClassOf edges, **plus** parents
  harvested from EquivalentClasses conjuncts (named operands of top-level intersections), with
  equivalence-cycle normalization (SCC-collapse; members of a cycle share parents/children,
  self-edges dropped) and `owl:Thing`/`owl:Nothing` filtering — mirroring what the structural
  reasoner returned. Provide `direct_parents/direct_children/ancestors/descendants` with memoized
  transitive closures (bitset or array-of-int adjacency; SNOMED-scale = ~400k classes must load
  in seconds).
- `projection.py` — `project(parsed, method, include_literals) -> list[Edge]`.
  **Replicate mowl's projectors by observed behavior, not by reading mowl docs**: the parity
  baselines from Deliverable 1 are the spec. Expected rule set (verify each against baselines):
  - taxonomy: `SubClassOf(A,B)` named→named ⇒ `(A, "http://subclassof", B)`.
  - owl2vecstar adds: `SubClassOf(A, some(r, B))` ⇒ `(A, r, B)`; `EquivalentClasses`
    intersections contribute their named + existential conjuncts; inverse/symmetric handling as
    observed; with `include_literals`, annotation/data literals as `(A, prop, literal)` edges.
  - Where mowl behavior is clearly buggy, matching it still wins (parity first); log deviations
    in the spec's Deviations section only if parity is impossible.
- `reasoning.py` — `ReasonerProtocol`, `AssertedHierarchyReasoner` (delegates to `hierarchy.py`),
  `load_reasoner()` with the `exact.reasoners` entry-point group (contracts §5).
- Performance guardrails (`03-performance.md` applies; this WP also **lands the benchmark
  harness** — `benchmarks/bench.py` with the `parse_ontology`, `build_hierarchy`,
  `projection_owl2vecstar`, `dataset_build_e2e`, `candidate_generation`,
  `inference_throughput` scenarios, `benchmarks/reference.json`, and the CI regression job):
  construction is single-pass over axioms; no per-IRI re-scans (the current code re-queries
  `EntitySearcher` per class; the new store must index once); IRIs interned to int ids at the
  boundary, hierarchy/adjacency as int arrays. If profiling on NCIT/SNOMED shows a hot kernel
  (e.g. transitive closure), it may move to a Cython module under
  `exact/ontology/_kernels.pyx` **with a bit-identical pure-Python fallback selected
  automatically** (contracts §15; `cython` is already a dependency) — do not add C code
  speculatively.

✋ **PR-B2**: `exact/ontology/` + unit tests against fixtures + conformance suite
(`tests/knowledge_source_conformance.py`) + baseline-diff test (`tests/ontology_parity_test.py`,
fixture baselines must match exactly).

## Deliverable 4 — rewire the pipeline (and delete Java)

Blast radius (complete list from the audit; anything not listed must not change):

1. `exact/core/contracts/dataset.py` — remove `OWLDataset`/reasoner imports and machinery
   (`_make_reasoner`, `_get_reasoner`, timeout executor, `source_reasoner`/`target_reasoner`).
   `load_ontologies` now builds `OwlOntologySource` via `exact.ontology.load_ontology(path)`.
   Move the concrete candidate-generation/exact-match/cache logic that lives in this "contract"
   into `exact/impl/datasets/base.py` (new `BaseAlignmentDataset(IDataset)`), leaving `IDataset`
   as a true interface + thin template methods. (This is WP-B's job, not WP-D's, because the
   move and the de-Java rewrite touch the same lines.)
2. `exact/core/entities/ontology.py` — `OntologyGraph` consumes `KnowledgeSource`: projection
   edges from `source.projection_edges(...)`; labels from `source.labels`; delete JPype
   string-coercion; keep the pure-Python parts untouched (IC weights, BFS neighborhoods,
   best-path context, bridges). `Entity.subclasses/superclasses` now delegate to
   `source.direct_children/direct_parents`. Rename file to `exact/core/entities/graph_view.py`
   if convenient (optional; shim if done).
3. `exact/impl/datasets/pair_adaptive_context.py` — `_direct_superclass_iris` →
   `source.direct_parents`; `_hierarchy_axiom_targets` + `_iter_java_items` deleted in favor of
   `source.hierarchy_bundle`; `_annotation_bundle` → `source.attributes`.
4. `exact/impl/datasets/contextgraph.py` — same treatment for the legacy path (labels,
   projection, relations, domains/ranges via the protocol).
5. `exact/utils/eval.py` — `MetricUtils.get_ignored_class_index` → `source.excluded_from_alignment`;
   drop `OWLOntology` typing; evaluator entry points load `OwlOntologySource` instead of mowl.
   (`impl/evaluator.py`, `core/contracts/evaluator.py` type-only imports updated.)
6. `exact/analysis/study_visualizer.py` — `direct_parents`/`direct_children`/annotation lookups
   via the protocol; drop the JVM fallback path.
7. `exact/analysis/user_study.py` — remove `init_jvm` usage in `_load_configs_for_rationale`
   (`:730-732`) and heap plumbing (`:953,1078,2601,2676,2686`).
8. **Delivery layer** (owned here, not WP-D): remove `init_jvm` from
   `delivery/cli/{align,eval,llm_debug,study_visualizer}.py` and `delivery/api/{align,eval}.py`.
   `-m/--jvm_heap_size` stays as a hidden, accepted-and-ignored argument emitting
   `DeprecationWarning` (so existing sbatch scripts don't crash); same for
   `EXACT_STUDY_JVM_HEAP_SIZE`. Update `deploy/sbatch/*.sh` to stop passing it.
9. `tools/run_cardinality_threshold_tests.py`, `tools/run_candidate_recall_experiment.py`,
   `tools/prepare_study_visualizer_bundle.py` — drop `init_jvm`/`JVMNotFoundException`.
10. `exact/__init__.py` — remove the `init_jvm` re-export; leave a stub raising a helpful
    `RuntimeError("Exact-OM no longer needs Java; remove init_jvm calls")` until 2.1.
11. `pyproject.toml` — remove `mowl-borg`; remove `gensim` (only mowl wanted it — verify with
    `git grep`); add `py-horned-owl`. `default_config.yaml:167` comment about JVM fork stalls:
    update comment; keep `num_workers: 0` default (changing it is a perf experiment, not this WP).
12. `utils/logs.py:27` — drop the `"jvm"` setup stage label.

There is **no dual-backend switch**: parity is proven by comparing against the *captured
baselines* from Deliverable 1, so mowl never needs to be importable after this PR.

## Deliverable 5 — parity gates (merge blockers for PR-B3)

On fixture + conference baselines (and Bio-ML if available locally):
1. Class sets, label maps, `excluded_from_alignment`: **exact match**.
2. Annotation/attribute bundles: exact match up to ordering.
3. Projection edge sets: exact match (taxonomy) / ≥99.9% Jaccard with every diff explained
   (owl2vecstar).
4. `hierarchy_bundle`: exact on fixtures; on conference/Bio-ML ≥99.9% of entities identical —
   divergences must be ELK-inferred (non-asserted) superclasses only; list them in the PR.
5. End-to-end conference run (fixed seed, LLM disabled or fixed fakes): global P/R/F1 within
   ±0.001 of the captured baseline; local Hits@k/MRR within ±0.001.
6. `git grep -iE "jpype|mowl|org\.semanticweb|init_jvm"` over `exact/ tools/ tests/ deploy/`
   returns nothing except deprecation shims, the isolated baseline-capture mode and its
   provenance, and docs; add this grep as a CI step.
7. Dataset build wall-time on conference pair ≤ the mowl path's captured time (JVM startup gone;
   expect a win) and NCIT-class ontology loads without >2× regression (soft gate; profile note
   in PR).

✋ **PR-B3**: rewiring + Java removal + parity evidence attached to the PR description.

## Out of scope

Property/instance signatures beyond exposing them in `entities(kind)` (WP-F consumes);
RDF/CSV sources (WP-G); timing (WP-C); README/docs beyond deleting the Java prerequisites
paragraph (`README.md:63`, `:101`) — full docs are WP-H.

## Risks

- **py-horned-owl API drift / gaps** (e.g. anonymous-individual annotations): isolate in
  `parser.py`; if a Bio-ML file exposes a parser gap, fall back to an rdflib-based parse for
  that construct inside `parser.py` (rdflib is already a dependency) rather than leaking it up.
- **ELK-vs-asserted divergence** larger than expected on some ontology: gate 4 will catch it;
  the escape hatch is a one-off `precomputed_hierarchy` option on `OwlOntologySource` fed from a
  file, and a note to build the future EL-reasoner plugin. Do not reintroduce Java.
- **Memory** on SNOMED-scale: intern IRIs (single str instances), use arrays/dicts of ints for
  hierarchy; verify RSS in the profile note.

## Deviations

No design deviations. The fixture snapshots are genuine `mowl-borg==1.0.1` captures. A
CPU-only `cmt-conference` release check against pre-removal revision `330044c` produced
byte-identical complete backend snapshots, global alignments, and local ranking alignments;
local metrics were identical and the legacy-denominator global metrics were identical. The
canonical 2.0 global recall differs only because WP-F deliberately excludes the three property
references from a class-only run. Inputs, hashes, metrics, and timings are recorded in
`benchmarks/evidence/wp_b_conference.json`.

No licensed Bio-ML ontology was available locally, so that conditional release check and the
NCIT/SNOMED-scale RSS soft gate remain external. The checked-in fixture and unredistributed
Conference captures exercise the complete parity gate without adding Java to the release
environment.
