# WP-M — Shared Java-free OWL stack migration

**Target release:** Exact-OM `2.1.0`. **Depends on:** completed WP-B plus released compatible
`pyowl-core`, `pyowl2vec-star-projector`, pyELK, and pyHermiT contracts. **Status:** repository
implementation M0–M4 complete; M5/release blocked by the performance, pinned-runner RSS, and
published-dependency gates recorded in §12.

**Audit baseline:** Exact-OM commit `c51b2b56f42bdd2cf6d27787bd607462d88d222b`
(version `2.0.0`, 2026-07-16). Re-locate files by symbol if line numbers drift.

## 1. Why this WP exists

WP-B successfully removed the JVM for Exact `2.0.0`, but necessarily created a private OWL
stack while no shared Python API was available. The current path parses through py-horned-owl,
serializes/re-normalizes with RDFLib, stores local records, builds local structural indexes, and
compiles OWL2Vec* edges locally. OAEI coherence and future repair reasoners would otherwise parse
the same ontology again into unrelated models.

WP-M changes ownership, not matching methodology. Exact holds one concrete
`pyowl_core.OntologySnapshot` and shares it by identity with projection and optional reasoners.
The committed WP-B/mOWL captures remain the behavior oracle.

## 2. Required dependency graph

```text
pyowl-core <--- pyowl2vec-star-projector
    ^                  ^
    |                  |
 pyELK / pyHermiT      |
    ^                  |
    +-------- Exact-OM-+
                 |
          OAEI-Bio-ML-eval (optional evaluation extra)
```

Exact may depend on each leaf, but no shared package imports Exact or OAEI. OAEI does not import
Exact. CI enforces the DAG and independent install/import smoke tests.

## 3. Snapshot ownership and loading

`OwlOntologySource` owns exactly one snapshot:

```python
from pyowl_core import OntologySnapshot

class OwlOntologySource(KnowledgeSource):
    @classmethod
    def load(cls, source, *, options=None, resolver=None) -> "OwlOntologySource":
        snapshot = pyowl_core.load_snapshot(source, options=options, resolver=resolver)
        return cls(snapshot)

    def __init__(self, snapshot: OntologySnapshot) -> None: ...

    def owl_snapshot(self) -> OntologySnapshot:
        return self._snapshot
```

Construction from an existing snapshot preserves `is` identity. Construction from a path,
bytes, or binary stream calls `load_snapshot(...)` once. `pyowl_core.coerce_snapshot(source)`
recognizes the `SnapshotProvider.owl_snapshot()` method and returns the same instance and lazy
view cache. No consumer receives `source.origin` as a substitute for the snapshot.

Process boundaries use only the core's versioned wire:

- producer: `encode_snapshot(snapshot)`;
- consumer: `decode_snapshot(buffer)`; or
- durable/memory-mapped worker input: `open_snapshot(path, mmap=True, verify=True)`.

Pickle, a Python object graph, a temporary RDF/XML export, and original-path handoff/reparse are
forbidden. The wire closure/resolution manifest and fingerprint are verified before use.

## 4. File-level migration

| Current ownership | 2.1 action |
|---|---|
| `exact/ontology/records.py` | delete structural dataclasses; internal imports use/re-export exact `pyowl_core` classes only where a one-minor compatibility shim is documented |
| `exact/ontology/parser.py` | replace parser/serialization pipeline with thin core loading/coercion; remove direct py-horned-owl use |
| `exact/io/sources/_owl_rdf.py` | remove from the OWL path and delete OWL normalization records; retain independent RDF/KG parsing only for the generic RDF source |
| `exact/ontology/expressions.py` | delete duplicate OWL expression records/walkers; use public core expression visitors/views |
| `exact/ontology/hierarchy.py` | OWL queries use core asserted views or selected reasoner; relocate the generic int-graph hierarchy needed by CSV/RDF sources under `exact/io` |
| `exact/ontology/projection.py` | reduce to a compatibility adapter over the shared projector; no rules/hash emulation remain in Exact |
| `exact/ontology/reasoning.py` | adapt core asserted views and optional pyELK/pyHermiT snapshot APIs to Exact's narrow protocol |
| `exact/ontology/store.py` | retain as the `KnowledgeSource` facade and snapshot/provider owner; indexes are feature-specific only |

Search gates reject new structural axiom/class-expression dataclasses under `exact/`, direct
imports of parser backends, and copied projector/reasoner rule tables. Generic `Edge` and
`AnnotationValue` facade types are allowed only where they are part of Exact's source-neutral
public contract; conversions must be bounded to result rows, not ontology structures.

## 5. Projection behavior

The OWL source delegates its exact snapshot to the shared projector. Effective defaults preserve
Exact 2.0:

- OWL2Vec*: `profile="mowl-d993536-v1"`, isolated fresh-invocation state, canonical order, and
  `duplicates="unique"`;
- taxonomy: the projector's dedicated asserted-taxonomy operation, not the pinned Scala
  `only_taxonomy` flag;
- backend: `auto`, with complete Python fallback and the projector's one-time warning;
- `dataset.projection_include_literals` maps to the projector's `include_literals`; and
- `io.source_options.include_abox` remains an **Exact-side** schema-only source view
  (`exact/io/sources/owl.py`): when `false`, individuals are dropped from signatures and
  edges touching them are filtered from the projector's returned edge list. It is not, and
  must not become, a projector `ProjectionOptions` field — the pinned profile emits ABox
  edges unconditionally, and the filter operates on the returned `Edge` rows.

Exact's current `exact/ontology/projection.py` and committed WP-B captures are secondary
executable comparators for migration. They encode observed mOWL details such as Java/OWLAPI 4.5
hash ordering and the sibling-subproperty overwrite defect. The pinned fresh-instance golden
output wins whenever prose or the public mOWL table differs. The upstream Scala cross-call role
map leakage is undesirable and excluded from production defaults; the projector's explicit
forensic compatibility-state option tests it separately. Every baseline difference is classified
and documented before approval.

Projection caches key on snapshot structural fingerprint, core model/wire versions, projector
package/API/profile/compiler schema, and normalized options. They never key only on a source path
or mtime.

## 6. Hierarchy and reasoning

`dataset.reasoner="asserted"` remains the default so Exact works with only core dependencies and
preserves current matching behavior. It uses core structural views over the shared snapshot.

The `reasoning` extra offers:

- `"elk"`: pyELK for OWL 2 EL classification and fast hierarchy queries; and
- `"hermit"`: pyHermiT for DL reasoning/coherence and future repair workflows.

Both packages accept `pyowl_core` ontology inputs and call `coerce_snapshot`; Exact passes the
source/provider or exact snapshot, never a path. The adapter exposes only `direct_parents`,
`direct_children`, `ancestors`, and `descendants` to existing matching code. Private reasoner IR
does not leak into Exact caches or public records.

Switching from asserted to inferred hierarchy can change model features and is explicit in
config/provenance; it is not silently enabled by installing an extra. Repair itself remains out
of scope for WP-M, but the snapshot/reasoner result seam must support later repair without another
parse.

## 7. Dependencies and packaging

For `2.1.0`:

- remove Exact's direct `py-horned-owl` dependency and every parser-backend import;
- require compatible `pyowl-core>=0.1,<0.2` and the released projector line;
- retain RDFLib only for Exact's generic RDF/OAEI serialization paths, not OWL normalization;
- keep pyELK and pyHermiT in an optional `reasoning` extra; and
- keep OAEI evaluation optional so Exact installs and aligns independently.

The base wheel and sdist are tested on Python 3.10–3.12 at minimum with Java, Cargo, native
projector, and reasoners absent. Pure-Python projection must be complete. Platform-wheel tests
also exercise native projection; reasoning-extra tests exercise both native reasoners without a
JVM.

## 8. Provenance and caches

Run manifests add one `ontology_stack` block per source/target containing:

- core package/API, `MODEL_SCHEMA_VERSION`, and `WIRE_FORMAT_VERSION`;
- structural, logical, and signature fingerprints;
- closure/resolution manifest and source-document fingerprints;
- projector package/backend/profile/options/compiler-cache schema;
- asserted/ELK/HermiT reasoner selection and package/backend version; and
- loader/import diagnostics and whether a verified wire snapshot was used.

No credential, machine-specific temporary path, or Python object ID is persisted. Existing
cache compatibility logic invalidates old `ParsedOntology`/projection caches with an actionable
message; it does not attempt unsafe unpickling/conversion.

## 9. Verification

### Hermetic gates

- instrumented load proves exactly one parse/closure resolution for each source and target;
- source, projector, and reasoner observe the exact snapshot object identity in-process;
- no snapshot fingerprint, axiom count, or shared lazy view changes after projection/reasoning;
- WP-B fixture class sets, labels, exclusions, annotations, domains/ranges, hierarchy bundles,
  and projection baselines match under documented ordering rules;
- conference end-to-end outputs/metrics remain within WP-B's `0.001` gates;
- projector Python/native outputs match, and Python-only Exact runs end-to-end;
- asserted behavior is unchanged when reasoner extras are absent;
- core wire workers match in-process results without invoking a parser;
- Python 3.10, 3.11, and 3.12 unit/integration matrices pass; and
- dependency/import scans find no Java, ROBOT, JPype, DeepOnto, OWLAPI, mOWL runtime, duplicate
  OWL records, or dependency cycle.

### Scale gates

Conference plus legally available Bio-ML/GO/NCIT inputs record load count, wall time, time to
first edge, throughput, peak/incremental RSS, cache hit behavior, and spill. WP-M must not add a
second ontology-sized representation. Large external data tests are release evidence even when
licensing prevents ordinary CI redistribution.

### Documentation and release audit

Update architecture, config reference, dependency extras, API docs, migration guide, CLI help,
benchmark docs, changelog, SBOM, and troubleshooting. Remove claims that Exact owns a parser or
requires parser-specific formats. Verify every example on Python 3.10 and 3.12. Version bump to
`2.1.0` occurs only after implementation and all gates; this spec does not change runtime
metadata early.

## 10. Work breakdown

1. **M0 — freeze baselines:** snapshot current tests/caches and validate projector goldens.
2. **M1 — source adapter:** introduce snapshot ownership/provider and dual-run read-only parity
   instrumentation.
3. **M2 — projection:** delegate OWL2Vec*/taxonomy, invalidate old caches, delete compiler rules.
4. **M3 — structural views:** migrate annotations/expressions/hierarchy and remove records/parser
   normalization.
5. **M4 — reasoner adapters:** asserted, optional pyELK/pyHermiT, provenance, worker wire.
6. **M5 — cleanup/release:** dependency removals, searches, scale/perf, docs, packages, `2.1.0`.

Each step keeps Exact runnable and has a rollback at the adapter boundary. Dual execution is
allowed only in tests/migration telemetry before local implementations are deleted; release code
has one shared path.

## 11. Out of scope

- modifying the shared core model for Exact-specific labels/features;
- fixing pinned projector semantics inside a snapshot;
- enabling inferred hierarchies by default;
- implementing ontology repair;
- changing typed/equivalence evaluation metrics; and
- merging repositories or releasing all packages in one distribution.

## 12. Implementation status and deviations (2026-07-18)

The repository implementation landed incrementally on `dev`: baseline `4a3cbc5`, source adapter
`b70cd16`, shared projection `87134b9`, structural views `50719f8`, reasoner adapters `c8d7de3`,
release-candidate cleanup `e943881`, and lazy-index scale correction `08b859d`. Hermetic tests
prove one load, exact in-process snapshot
identity across source/projector/reasoner, immutable shared fingerprints, verified-wire workers,
fixture semantic parity, Python-only operation, cache versioning, and forbidden-dependency scans.
Exact's optional OAEI 0.2 coherence adapter also passes these same providers directly rather than
their origin paths.

M5 is not accepted. The content-addressed result in
`benchmarks/evidence/wp_m_ncit_doid_candidate.json` uses the exact frozen NCIT–DOID inputs and
proves one load and no second ontology representation, but records:

- NCIT load at 301.92 s versus 68.75 s and projection at 7.056 s versus 0.218 s;
- DOID load at 23.10 s versus 5.44 s and projection at 0.947 s versus 0.0528 s;
- 1.678 GB NCIT peak process RSS, still lacking the comparable pinned-runner baseline required
  for an RSS acceptance decision.

The 42,103-versus-41,349 NCIT edge difference is now completely classified. Exact set comparison
records 762 additions from pinned projector rule RB-019 (restriction expansion through historical
inverse/subrole maps) and eight removals from RB-009 (the old private Exact path incorrectly
expanded top-level intersection superclass shapes), with zero residual edges. The shared result
therefore passes pinned mOWL/projector semantic parity even though it does not reproduce the old
private approximation's count.

The load and projection measurements still exceed the permitted 25% wall-time regression.
Compatible final core/projector/reasoner distributions and hosted Python/native-wheel matrices
also remain release prerequisites. Exact therefore stays at `2.0.0`; changing metadata to `2.1.0`
before these gates pass would contradict §9. Lazy Exact feature indexes and the current shared-core
optimizations materially improve the first diagnostic run, but do not close the shared-stack
performance gap.
