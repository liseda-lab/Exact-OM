# 01 — Target Architecture

## Where we are (commit `9e72ecf`)

The `exact` package already follows a sound **contracts → impl → delivery** layering with a
self-registration registry (`exact/core/entities/registry/`) wiring names to classes and a
dependency map binding each model to its dataset+trainer (`exact/impl/__init__.py:8-11`).
The pipeline: CLI/API → `AlignmentAction.run` → dataset build (candidate generation + per-entity
feature bundles) → `SemanticAlignmentRunner.predict` (pair-adaptive scorer, gated LLM decision) →
selector/reranker → prefilter → outputs → optional evaluation.

What has eroded through iteration:

- **Java everywhere at the base**: `mowl-borg` (OWL-API + JPype) is the single ontology backend;
  every entry point starts a JVM. Yet the default pipeline consults the reasoner in exactly one
  place — direct named superclasses (`exact/impl/datasets/pair_adaptive_context.py:129-142`) —
  and everything else is asserted axioms + annotations.
- **Contracts doing concrete work**: `exact/core/contracts/dataset.py` (1204 lines) builds
  reasoners, generates candidates, and manages caches inside an "interface".
- **Monoliths**: `candidate_set_selector.py` (3294), `semantic_runner.py` (3055),
  `analysis/user_study.py` (2716), `semantic_scorer.py` (2340), `pair_adaptive_scorer.py` (2171).
- **Timing that lies on resume** (see WP-C), a metrics-registry bug, dead code, misnamed modules
  (`utils/paths.py` is graph search), a misspelled public class (`EvalutionRunner`), broken
  packaging (`study_visualizer_runtime` imported but not shipped), stale Matcha-DL artifacts.

## Where we're going

```
exact/
  core/                    # pure: contracts, entities, orchestration. No heavy deps, no impl imports.
    contracts/             #   IModel, IDataset (slim), ITrainer, IMetric, IEvaluator, ISeedSetter,
                           #   knowledge.py (KnowledgeSource protocol — see 02-shared-contracts.md)
    entities/              #   configs, mappings, EntityKind, evaluation, registry
    actions/               #   run_alignment(), run_evaluation() as plain functions (not Protocols)
    values.py
  ontology/                # NEW (WP-B) — the only OWL-aware package
    parser.py              #   sole importer of py-horned-owl
    store.py               #   OwlOntologySource: indexed signature, annotations, axioms
    hierarchy.py           #   asserted class/property hierarchy (direct parents/children, equiv-normalized)
    expressions.py         #   class-expression walking (named, someValuesFrom, intersections)
    projection.py          #   owl2vecstar + taxonomy projectors → Edge lists
    reasoning.py           #   ReasonerProtocol + AssertedHierarchyReasoner + plugin loading
  io/                      # NEW (WP-G) — sources and writers, format-dispatching
    sources/               #   owl.py (wraps exact.ontology), rdf.py (rdflib), csv_kg.py (BioKG), datalog.py
    writers/               #   tsv.py, oaei_rdf.py, typed_tsv.py, json.py + registry
  runs/                    # NEW (WP-L) — run-dir layout (v1+v2), run_manifest.json, sharded
                           #   indexed explanation store, RunReader, retention/GC
  tracks/                  # NEW (WP-I) — dataset/track retrieval: TrackProvider protocol,
                           #   declarative YAML descriptors (bio-ml HF, diso-oaei, conference,
                           #   anatomy, OAEI-KG, biokg stub), lockfile w/ revision pinning
                           #   [extra: hf for HuggingFace-backed tracks]
  impl/
    datasets/              #   contextgraph.py, pair_adaptive_context.py (consume KnowledgeSource)
    models/                #   scorer/ and selector/ subpackages (WP-D split)
    trainer/               #   runner.py + checkpointing.py + audit_io.py + rationales.py (WP-D split)
    evaluators/            #   builtin.py (current Evaluator) + bioml.py (WP-E adapter)
    metrics/
    seed.py
  llm/                     # routing.py (moved from utils/llm_routing.py, WP-D)
  utils/                   # timing.py (ledger), logs.py, graph_search.py (ex paths.py),
                           # formatting.py, data.py, mappings.py, candidate_generation.py
  delivery/
    common.py              #   shared arg validation/assembly (WP-D)
    cli/  api/             #   thin wrappers over actions
  analysis/                # post-run tools (user_study/ package, diagnostics)
exact_inspect/             # alignment-inspection service/CLI (ex study_visualizer_runtime;
                           #   WP-A packages it as-is, WP-K renames + repackages)
                           #   [extra: viz — fastapi/uvicorn leave the core deps]
```

**Core vs plugins.** The core (`exact` matcher + builtin evaluator + OWL backend) installs and
runs with zero optional deps, no network services, no Java. Everything situational is an
in-repo plugin behind an extra (contracts §13: `viz`, `hf`, `bioml-eval`, `docs`) with
import-linter-enforced boundaries, so any of them can graduate to a separate PyPI package later
without code changes. Config is schema v2 (`config_version: 2`, WP-J) with a migrator for v1
files.

## Layering rules (enforced with `import-linter` from WP-A on)

1. `exact.core` imports only stdlib, pydantic, torch-typing, and `exact.utils` leaf helpers —
   never `exact.impl`, `exact.delivery`, `exact.ontology`, `exact.io`.
   (Registry wiring stays lazy/by-import-path, as today.)
2. `exact.ontology` and `exact.io` import `exact.core` (for contracts/entities) and nothing from
   `exact.impl`/`exact.delivery`. `exact.ontology` is the **only** package importing `pyhornedowl`;
   `exact.io` is the only one importing `rdflib`.
3. `exact.impl` may import `core`, `ontology`, `io`, `llm`, `utils`.
4. `exact.delivery` may import everything; nothing imports `exact.delivery`.
5. `exact.analysis` may import `core`/`impl`/`utils`; nothing inside `exact` imports `analysis`
   except `delivery` CLIs.

## Extension points

- **ComponentRegistry** (existing): models, datasets, trainers, metrics — extended with
  `EVALUATOR` (WP-E). Registration stays via `SelfRegisteringComponent.__init_subclass__`.
- **Entry-point plugin groups** (new, WP-B/WP-G/WP-I): `exact.reasoners`, `exact.sources`,
  `exact.writers`, `exact.tracks`. The main library ships no DL reasoner; a future
  `exact-reasoner-el` package (Python + C kernels, PyLogMap-style) can register through
  `exact.reasoners` without any change here. This satisfies "avoid reasoners in the main lib,
  keep the door open". New datasets are addable via YAML descriptor or an `exact.tracks` plugin.
- **Performance**: cross-cutting policy in `03-performance.md` — benchmark harness with CI
  regression gates, and Cython kernels (bit-identical pure-Python fallbacks) for measured hot
  spots only.

## Non-goals (explicitly out of scope for this overhaul)

- No changes to the matching algorithms, scoring math, LLM prompting, or calibration logic.
- No build-system migration (Poetry stays), no rename of the `exact` package or repo.
- No monorepo split. Candidate future extractions — `exact/ontology` as a standalone package,
  an EL reasoner plugin — are prepared by the boundaries above but not executed now.
- No rewrite of `explanations_visualizer` (Next.js app) beyond packaging/docs touch-ups.
