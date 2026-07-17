# 02 — Shared Contracts (frozen interfaces for parallel work)

These interfaces let WPs proceed in parallel: WP-B implements them for OWL, WP-C implements the
ledger, WP-D/E/F/G consume them. **Changing a signature here requires updating this file in the
same PR** and checking the consumers listed next to each contract.

**2.1 ownership note:** WP-M supersedes WP-B wherever this document assigns structural OWL
records, parsing, projection rules, or OWL hierarchy storage to Exact. Exact keeps its
`KnowledgeSource` facade, but the authoritative object is the concrete
`pyowl_core.OntologySnapshot`; projector/reasoners receive that same instance.

Status legend: `[B]` implemented by WP-B, `[C]` by WP-C, `[G]` by WP-G, etc.

## 1. `EntityKind` `[B, consumed by F/G]`

`exact/core/entities/kinds.py`

```python
class EntityKind(str, Enum):
    CLASS = "class"
    OBJECT_PROPERTY = "object_property"
    DATA_PROPERTY = "data_property"
    ANNOTATION_PROPERTY = "annotation_property"
    INDIVIDUAL = "individual"
```

**WP-M mapping to `pyowl_core.EntityKind`** (explicit at the `exact.ontology` boundary;
never string-compared across packages): `CLASS↔CLASS`, `OBJECT_PROPERTY↔OBJECT_PROPERTY`,
`DATA_PROPERTY↔DATA_PROPERTY`, `ANNOTATION_PROPERTY↔ANNOTATION_PROPERTY`, and
`INDIVIDUAL↔NAMED_INDIVIDUAL` (note the differing value strings `"individual"` vs
`"named_individual"`). Core's `DATATYPE` kind has no Exact counterpart and is never a
matching kind; the adapter surfaces datatype entities explicitly where a signature needs
them and must not coerce them to another kind. Anonymous individuals are not named
entities and appear in neither enum.

## 2. `Edge` `[B]`

`exact/core/entities/graph.py` — replaces `mowl.projection.Edge` everywhere
(consumers: `exact/utils/paths.py` best-path functions, `OntologyGraph`, datasets).

```python
@dataclass(frozen=True, slots=True)
class Edge:
    src: str   # IRI
    rel: str   # IRI or synthetic relation id (e.g. "http://subclassof")
    dst: str   # IRI or literal (when include_literals)

    def astuple(self) -> tuple[str, str, str]: ...
```

Keep `.astuple()` and attribute names `src/rel/dst` — the existing graph code depends on them.

## 3. `AnnotationValue` `[B]`

```python
@dataclass(frozen=True, slots=True)
class AnnotationValue:
    property_iri: str
    value: str                 # lexical form for literals, IRI string otherwise
    is_literal: bool
    lang: str | None = None
    datatype: str | None = None
```

## 4. `KnowledgeSource` protocol `[B for OWL, G for RDF/CSV]`

`exact/core/contracts/knowledge.py`. This is **the** seam that removes both OWL-API and, later,
OWL itself from the pipeline: datasets/models consume only this protocol. Each method notes the
current call it replaces (from the Java audit).

```python
class KnowledgeSource(Protocol):
    @property
    def origin(self) -> Path | None: ...
    def entities(self, kind: EntityKind = EntityKind.CLASS) -> Sequence[str]:
        """IRIs in signature. Replaces getClassesInSignature (ontology.py:766, dataset.py:364)."""
    def labels(self, iri: str) -> list[str]:
        """rdfs:label (+ configured label properties). Replaces EntitySearcher label lookups
        (ontology.py:351-373)."""
    def annotations(self, iri: str, properties: Sequence[str] | None = None) -> list[AnnotationValue]:
        """All annotation assertions on iri, optionally filtered by property IRIs.
        Replaces EntitySearcher.getAnnotations (dataset.py:833, pair_adaptive_context.py:373,
        study_visualizer.py:201, utils/eval.py:63)."""
    def attributes(self, iri: str) -> list[AnnotationValue]:
        """Non-label literal annotations (+ data-property values for individuals).
        Feeds the attribute channel (pair_adaptive_context._annotation_bundle)."""
    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Direct named parents in the (equivalence-normalized) asserted hierarchy.
        Replaces reasoner.getSuperClasses(direct=True) (pair_adaptive_context.py:129-142) —
        the ONLY reasoner call on the default pipeline. For properties: subPropertyOf.
        For individuals: asserted rdf:type classes."""
    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Inverse of direct_parents. Replaces getSubClasses (ontology.py:839,
        study_visualizer.py:259)."""
    def hierarchy_bundle(self, iri: str, families: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        """families maps family name -> property IRIs, e.g. {"is_a": [], "part_of": [BFO:0000050]}.
        "is_a" comes from direct_parents; other families from existential restrictions on OWL
        (replaces pair_adaptive_context._hierarchy_axiom_targets :234-282) or configured edge
        predicates on CSV/RDF sources."""
    def projection_edges(self, *, method: str = "owl2vecstar", include_literals: bool = False) -> list[Edge]:
        """Graph projection. Replaces mowl OWL2VecStarProjector/TaxonomyProjector
        (ontology.py:151-189). method in {"owl2vecstar", "taxonomy"}; CSV/RDF sources return
        their triples directly and ignore method."""
    def property_domains(self, prop_iri: str) -> list[str]: ...
    def property_ranges(self, prop_iri: str) -> list[str]:
        """Replace getObjectPropertyDomain/RangeAxioms (ontology.py:248,256). Empty for pure KGs."""
    def excluded_from_alignment(self) -> frozenset[str]:
        """IRIs annotated use_in_alignment=false (values.py:7) plus owl:deprecated=true.
        Replaces dataset._ignored_alignment_class_iris (:345-408) and
        MetricUtils.get_ignored_class_index (utils/eval.py:37-69)."""
    def short_form(self, iri: str) -> str:
        """Replaces IRI.getShortForm()."""
```

For an OWL source, `OwlOntologySource` additionally implements the shared adapter protocol:

```python
from pyowl_core import OntologySnapshot

class OwlOntologySource(KnowledgeSource):
    def owl_snapshot(self) -> OntologySnapshot:
        """Return the exact snapshot instance owned by this source; never rebuild or reparse."""
```

`pyowl_core.coerce_snapshot(source)` calls `owl_snapshot()` once and preserves object identity
and the snapshot's shared lazy-view cache. This method is deliberately OWL-specific and is not
added to generic RDF/CSV `KnowledgeSource` implementations.

Notes:
- `OwlOntologySource` (WP-B, migrated by WP-M in `exact/ontology/store.py`) implements this and
  may expose the snapshot-provider method, but generic pipeline code must not use OWL extras.
- Expensive results (labels map, hierarchy, projection) are computed once and cached on the
  shared snapshot/lazy views or on the source; all methods are read-only and thread-safe after
  construction.
- `KnowledgeSourceConformance` (WP-B delivers, WP-G reuses): a parametrized pytest suite any
  implementation must pass (`tests/knowledge_source_conformance.py`).

## 5. `ReasonerProtocol` + shared reasoner adapters `[B, superseded by M for OWL ownership]`

`exact/ontology/reasoning.py`

```python
class ReasonerProtocol(Protocol):
    def direct_parents(self, iri: str) -> list[str]: ...
    def direct_children(self, iri: str) -> list[str]: ...
    def ancestors(self, iri: str) -> set[str]: ...
    def descendants(self, iri: str) -> set[str]: ...

def load_reasoner(name: str, store: "OwlOntologySource") -> ReasonerProtocol:
    """Use store.owl_snapshot() by identity. "asserted" uses core structural views;
    "elk" and "hermit" adapt optional native packages; other names use entry points."""
```

The base installation preserves `"asserted"` as the default and remains independently usable.
The `reasoning` extra installs pyELK/pyHermiT. Adapters call their snapshot constructors, never
pass a path, translate into Exact's narrow protocol without copying the ontology, and include
core/reasoner versions plus structural/logical/signature fingerprints in provenance. Process
workers exchange only `pyowl_core.encode_snapshot`/`decode_snapshot`/`open_snapshot` artifacts;
pickle and original-OWL-path handoff are prohibited.

## 5a. Projection delegation `[M]`

`KnowledgeSource.projection_edges(...)` remains source-neutral. The OWL implementation delegates
to `pyowl2vec_star_projector` with `store.owl_snapshot()` as the strict snapshot argument.
`owl2vecstar` uses the pinned mOWL compatibility profile and stable unique-edge mode to preserve
Exact 2.0 behavior; `taxonomy` uses the projector's dedicated asserted-taxonomy API, not mOWL's
defective `only_taxonomy` flag. A narrow conversion to Exact's public `Edge(src, rel, dst)` is
allowed; duplicating compiler rules or structural axioms is not.

## 6. Candidate/feature table schema `[additive changes only]`

The processed dataset DataFrame and alignment outputs are keyed by `Src`, `Tgt` (IRIs) with
`Score` and mask columns (`exact/core/entities/configs/dataset.py: DatasetMask`). Rules:

- Existing columns are never renamed or re-typed.
- WP-F adds `SrcKind`/`TgtKind` (values from `EntityKind`, default `"class"` when absent).
- WP-G adds `Relation` (default `"="`) for typed outputs.
- Readers must treat missing new columns as their defaults (old caches/checkpoints stay loadable;
  cache fingerprints — `IDataset.cache_fingerprint` — must incorporate the new params so changed
  configs never silently reuse stale caches).

## 7. Timing ledger `[C]`

File: `<output_dir>/timings.json`, schema:

```jsonc
{
  "schema_version": 1,
  "sessions": [
    {
      "run_id": "uuid4",
      "command": "align" | "eval",
      "started_at": "2026-07-15T10:00:00+00:00",
      "ended_at": "...",                      // null if crashed
      "config_fingerprint": "sha1",           // from ConfigModel dump (stable serialization)
      "dataset_signature": "sha1|null",       // IDataset.dataset_signature
      "exact_version": "2.0.0",
      "stages": [
        {
          "stage": "Alignment.Inference",     // names from exact/core/values.py:TIMING_STEP_ORDER
          "seconds": 812.4,                   // monotonic (perf_counter) span for THIS session
          "cache_status": "fresh|resumed|cache_hit|skipped",
          "work_done": 1200, "work_total": 5000, "unit": "examples"  // optional
        }
      ]
    }
  ]
}
```

API (`exact/utils/timing.py`):

```python
class CacheStatus(str, Enum): FRESH; RESUMED; CACHE_HIT; SKIPPED

class TimingLedger:
    @classmethod
    def open(cls, run_dir: Path) -> "TimingLedger": ...
    @contextmanager
    def session(self, *, command: str, config_fingerprint: str,
                dataset_signature: str | None = None) -> Iterator["RunSession"]: ...
    def stage_totals(self, *, config_fingerprint: str | None = None) -> dict[str, "StageTotal"]:
        """StageTotal(compute_seconds, overhead_seconds, sessions, work_done, work_total).
        compute_seconds = sum over FRESH+RESUMED records; CACHE_HIT/SKIPPED go to overhead."""
    def estimates(self, *, config_fingerprint: str | None = None) -> dict[str, float]:
        """Per-stage expected seconds for a fresh run (feeds RunProgressLogger)."""

class RunSession:
    @contextmanager
    def stage(self, name: str, *, cache_status: CacheStatus = CacheStatus.FRESH,
              work_total: int | None = None) -> Iterator["StageSpan"]: ...
    def record(self, name: str, *, seconds: float, cache_status: CacheStatus,
               work_done: int | None = None, work_total: int | None = None,
               unit: str | None = None) -> None: ...
```

Semantics (the fixes — rationale in WP-C):
- **Append-only.** A session never mutates prior sessions. No merge-overwrite of `times.txt`-style.
- `0.0`-second skipped stages are recorded as `SKIPPED`/`CACHE_HIT`, never as `FRESH` — so a fully
  resumed run can no longer erase real inference time.
- Concurrency-safe: atomic write (tmp + `os.replace`) under an advisory lockfile
  (`timings.json.lock`, `O_CREAT|O_EXCL` with stale-lock timeout) — two runs sharing a dir
  interleave without corruption.
- `times.txt` remains as a **derived, human-readable render** of `stage_totals()` (deprecated;
  removed in a later release).

## 8. Checkpoint manifest additions `[C writes, D preserves]`

`SemanticAlignmentRunner` inference checkpoints gain a `timing` block:

```jsonc
"timing": { "inference_seconds_cumulative": 812.4, "examples_per_second_ema": 1.48 }
```

On resume the trainer seeds its accumulator from this, so `Alignment.Inference` reported at the
end of a resumed session equals cumulative true cost (session record still stores only this
session's span with `cache_status="resumed"`; both views stay reconstructible).

## 9. Config surface additions

Naming note: **WP-J restructures the config into schema v2** (`WP-J-config-v2.md`) early in
wave 2. Wave-1 WPs (B/C/E/I) add keys under the v1 section names below and WP-J folds them into
v2; wave-2 WPs (F/G) land v2-native names directly (right column).

| Key (v1 home → v2 home) | Default | WP | Meaning |
|-----|---------|----|---------|
| `dataset_params.reasoner` → `dataset.reasoner` | `"asserted"` | B | Reasoner plugin name |
| `dataset.projector.backend` (v2) | `"auto"` | M | native projector when available, complete Python fallback otherwise |
| `dataset.projector.profile` (v2) | pinned compatibility profile | M | versioned OWL2Vec* edge semantics |
| `dataset_track.*` → `data.*` (`track`, `task`, `root`, `revision`) | — | I | Track-provider dataset resolution |
| `evaluation.backends` (same in v2) | `["builtin"]` | E | Ordered evaluator names (`builtin`, `bioml`) |
| `evaluation.bioml.*` (same) | — | E | Track/task options passed to OAEI-Bio-ML-eval |
| `matching.entity_kinds` (v2-native) | `["class"]` | F | Kinds to match |
| `io.input_format` (v2-native) | `"auto"` | G | `auto\|owl\|rdf\|csv-kg` |
| `io.source_options` / `io.target_options` (v2-native) | `{}` | G | Per-source options (label predicates, hierarchy predicates, datalog files) |
| `io.output_formats` (v2-native) | `["tsv-global","tsv-local"]` | G | Writer names: + `oaei-rdf`, `typed-tsv`, `json` |
| `matching.relation_prediction` (v2-native) | `"none"` | G | `none\|hierarchy_heuristic` |
| `output.explanations.shard_mb` (v2-native) | `32` | L | Explanation-store shard size |
| `output.save.full_explanations_json` (v2-native) | `false` | L | Emit legacy monolithic JSON (deprecated) |
| `output.retention.checkpoints` (v2-native) | `"latest"` | L | Checkpoint pruning at run finalization |

Removed/deprecated: `-m/--jvm_heap_size` and `EXACT_STUDY_JVM_HEAP_SIZE` become accepted-but-ignored
with a `DeprecationWarning` (WP-B), deleted in a later release.

## 10. Output format registry names `[G]`

`tsv-global` (`src2tgt.maps_global.tsv`), `tsv-local` (`src2tgt.maps_local.tsv`),
`oaei-rdf` (Alignment-Format `.rdf`), `typed-tsv` (`SrcEntity\tTgtEntity\tRelation\tScore`,
BioKG-submission-compatible), `json`.

## 11. Public API compatibility shims

| Old | New | Shim until |
|-----|-----|-----------|
| `exact.delivery.api.EvalutionRunner` | `EvaluationRunner` | 2.1 |
| `exact.utils.paths` (graph search) | `exact.utils.graph_search` | 2.1 |
| `bioml-eval` console script | `exact-eval` (both installed) | keep both |
| `times.txt` | `timings.json` | 2.1 |
| `exact.init_jvm` re-export | removed (no-op stub raising helpful error) | 2.1 |
| v1 config files (no `config_version`) | v2 schema, auto-migrated + `exact config migrate` | 2.1 |
| `study_visualizer_runtime` pkg, `exact-study-viz` script, `EXACT_STUDY_*` env | `exact_inspect` pkg, `exact-inspect` script, `EXACT_INSPECT_*` env | 2.1 |
| `data/get_data.py` | `exact data pull/verify/status` CLI | removed in 2.0 (WP-I) |
| `full_explanations.json` (written during run) | explanation store + `exact run export` (WP-L); legacy file still emitted when `output.save.full_explanations_json: true` | 2.1 |
| run layout v1 (`model/alignment/...`) | layout v2 + `run_manifest.json`; v1 dirs stay readable via `RunReader` | read support kept |
| `exact.ontology` structural record/parser/projector imports | `pyowl_core` and `pyowl2vec_star_projector` public APIs | 2.2 for documented import shims; no duplicate engine |

## 12. `TrackProvider` protocol `[I]`

`exact/tracks/provider.py` — dataset/track retrieval seam (details: `WP-I-datasets-tracks.md`).

```python
@dataclass(frozen=True)
class TaskLayout:
    source: Path            # owl file or csv-kg dir
    target: Path
    refs: dict[str, Path]   # e.g. {"train": ..., "test": ...}; may be empty
    candidates: Path | None
    extras: dict[str, Any]  # e.g. {"repaired_refs": Path, "nil_candidates": True}
    provenance: dict[str, Any]  # provider, upstream id, revision, hashes

class TrackProvider(Protocol):
    name: str
    def tasks(self) -> list[str]: ...
    def materialize(self, task: str, data_root: Path, *,
                    revision: str | None = None, update: bool = False) -> TaskLayout: ...
    def verify(self, task: str, data_root: Path) -> "VerificationReport": ...
    def status(self, task: str, data_root: Path) -> Literal[
        "ok", "local-drift", "upstream-moved", "not-materialized"]: ...
```

Built-in providers are **declarative YAML descriptors** interpreted by generic HTTP/HF engines;
third-party providers register via the `exact.tracks` entry-point group. Materializations are
revision-pinned in `<data_root>/datasets.lock.json`.

## 13. Optional-dependency extras (plugin model: in-repo + extras)

| Extra | Enables | Pulls |
|-------|---------|-------|
| `viz` | `exact_inspect` service/CLI | fastapi, uvicorn |
| `hf` | Hugging Face track providers (bio-ml, diso-oaei, biokg) | huggingface_hub |
| `bioml-eval` | `bioml` evaluation backend | oaei-bioml-eval (git pin) |
| `docs` | docs build | mkdocs-material, mkdocstrings, … |

Core `exact` must import none of these at module import time; each integration module
import-guards with an actionable `pip install "exact-om[<extra>]"` error. Boundaries are
enforced by import-linter so any extra can later be extracted to its own PyPI package without
code changes.

## 14. `RunReader` + run-layout contract `[L implements; K, analysis consume]`

`exact/runs/reader.py` — the read seam for run artifacts (details: `WP-L-run-artifacts.md`).
Consumers (`exact_inspect`, `analysis/user_study`, diagnostics) must access run artifacts only
through this API — never by hardcoding paths like `model/alignment/...`.

```python
class RunReader(Protocol):
    @classmethod
    def open(cls, run_dir: Path) -> "RunReader": ...   # auto-detects layout v1 (pre-WP-L) or v2
    def mappings(self, kind: Literal["global", "local"]) -> pd.DataFrame: ...
    def explanations_for(self, src_iri: str) -> list[dict]: ...   # lazy, one shard max (v2)
    def iter_explanations(self) -> Iterator[dict]: ...
    def manifest(self) -> dict: ...                    # run_manifest.json (v2) or synthesized (v1)
    def stats(self) -> dict: ...                       # run_stats.json
```

Layout v2 directories (`alignment/`, `explanations/` store, `stats/`, `plots/`,
`checkpoints/`, `run_manifest.json`) and the explanation-record schema are owned by WP-L;
explanation records carry the producing session `run_id`, joining them to `timings.json`
sessions (§7). `full_explanations.json` becomes a derived export (see §11).

## 15. Compiled-kernel dispatch contract `[any WP adding a kernel]`

Per `03-performance.md`: a module gaining a Cython kernel ships `_kernels.py` exposing the pure
API, auto-selecting the compiled variant when importable; `EXACT_FORCE_PYTHON_KERNELS=1` forces
Python; both variants are bit-identical (same tests run against both in CI).
