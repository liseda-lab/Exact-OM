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
  ontology/                # WP-M — thin Exact adapter over the shared Java-free OWL stack
    store.py               #   OwlOntologySource owns one pyowl_core.OntologySnapshot
    projection.py          #   compatibility shim delegating to pyowl2vec_star_projector
    reasoning.py           #   asserted core views + optional pyELK/pyHermiT adapters
    provenance.py          #   shared fingerprints/version/options in Exact run manifests
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

**Core vs plugins.** The core (`exact` matcher + builtin evaluator + shared OWL core/projector)
installs and runs with zero optional deps, no network services, no Java, and a complete
pure-Python projector fallback. Native projection and pyELK/pyHermiT reasoning are optional.
Everything situational is an
in-repo plugin behind an extra (contracts §13: `viz`, `hf`, `bioml-eval`, `docs`) with
import-linter-enforced boundaries, so any of them can graduate to a separate PyPI package later
without code changes. Config is schema v2 (`config_version: 2`, WP-J) with a migrator for v1
files.

## Layering rules (enforced with `import-linter` from WP-A on)

1. `exact.core` imports only stdlib, pydantic, torch-typing, and `exact.utils` leaf helpers —
   never `exact.impl`, `exact.delivery`, `exact.ontology`, `exact.io`.
   (Registry wiring stays lazy/by-import-path, as today.)
2. `exact.ontology` and `exact.io` import `exact.core` (for contracts/entities) and nothing from
   `exact.impl`/`exact.delivery`. `exact.ontology` imports the public `pyowl_core` and projector
   APIs only; it owns no structural OWL records or parser. `exact.io` is the only package
   importing `rdflib` for non-OWL RDF/KG sources.
3. `exact.impl` may import `core`, `ontology`, `io`, `llm`, `utils`.
4. `exact.delivery` may import everything; nothing imports `exact.delivery`.
5. `exact.analysis` may import `core`/`impl`/`utils`; nothing inside `exact` imports `analysis`
   except `delivery` CLIs.

## Extension points

- **ComponentRegistry** (existing): models, datasets, trainers, metrics — extended with
  `EVALUATOR` (WP-E). Registration stays via `SelfRegisteringComponent.__init_subclass__`.
- **Entry-point plugin groups** (WP-B/WP-G/WP-I): `exact.reasoners`, `exact.sources`,
  `exact.writers`, `exact.tracks`. WP-M gives the reasoner seam first-party adapters for optional
  pyELK and pyHermiT, both consuming `OwlOntologySource.owl_snapshot()` by identity. Third-party
  reasoners may still register, but path handoff/reparse and private OWL models are forbidden.
  New datasets remain addable via YAML descriptor or an `exact.tracks` plugin.
- **Performance**: cross-cutting policy in `03-performance.md` — benchmark harness with CI
  regression gates, and Cython kernels (bit-identical pure-Python fallbacks) for measured hot
  spots only.

## Non-goals (explicitly out of scope for this overhaul)

- No changes to the matching algorithms, scoring math, LLM prompting, or calibration logic.
- No build-system migration (Poetry stays), no rename of the `exact` package or repo.
- No repository merger or dependency cycle. The shared core, projector, and reasoners remain
  independently versioned packages; Exact consumes them through public contracts (WP-M).
- No rewrite of `explanations_visualizer` (Next.js app) beyond packaging/docs touch-ups.

## WP-M ownership correction for 2.1

WP-B correctly removed Java for 2.0, but its local `records.py`, parser/normalizer,
expression walkers, hierarchy indexes, and projector duplicate capabilities are now owned by
the shared stack. WP-M removes those OWL-specific duplicates. The generic hierarchy utility needed
by CSV/RDF knowledge sources remains inside Exact under `exact.io`; it must not become an OWL
model. The exact snapshot instance is the only in-process handoff to projector/reasoners, and
cross-process work uses the versioned core wire format rather than pickle or source-path reparse.
