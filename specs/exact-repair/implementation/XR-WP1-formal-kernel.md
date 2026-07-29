# XR-WP1 — Formal kernel, state contracts, and certificates

**Status:** proposed<br>
**Depends on:** XR G0; current shared OWL snapshot and run-manifest contracts<br>
**Unlocks:** XR-WP2, XR-E00<br>
**Research boundary:** creates evidence infrastructure for RQ1–RQ3 and RQ10; it does not answer
those questions or change alignment results by default.

## Objective

Implement the deterministic, reasoner-independent kernel on which every safety and optimality
claim depends: immutable domain records, the base class-state compiler, policy/query records,
canonical provenance, statuses, certificates, and replay scaffolding.

This WP deliberately stops before a production solver/reasoner loop. It makes the formal objects
executable and reviewable without conflating them with backend behavior.

## In scope

- `exact.repair` package scaffold with lazy optional imports;
- `RevisionObjectV1`, `ClassStateV1`, assignment, policy, violation, explanation, state-conflict,
  utility/bound, capability, result, and certificate records;
- Exact-OM mapping/evidence normalization into class revision objects;
- base state generation and compilation for off/equivalence/forward/backward;
- canonical axiom/state serialization, hashing, deduplication, and provenance indexes;
- status semantics from `01-architecture-and-contracts.md`;
- candidate inventory freeze/read APIs;
- certificate writer/reader and replay validation up to the oracle call;
- logical regression fixtures and an exhaustive semantic truth-table harness interface;
- additive layout-v2 `repair/` artifact paths and `RunReader` accessors.

## Out of scope

- marking a real ontology assignment safe;
- selecting a solver or extending upstream reasoner packages;
- guards, qualifications, endpoint replacements, learned utility, component solving, or Exact-OM
  automatic integration;
- changing config defaults or existing matching artifacts.

## Required semantics

### Base class compiler

For normalized classes `s` and `t`, compile exactly:

| Family | Forward mode | Backward mode | Axioms |
|---|---|---|---|
| off | off | off | none |
| forward | plain | off | `s SubClassOf t` |
| backward | off | plain | `t SubClassOf s` |
| equivalence | plain | plain | both subclass axioms in canonical order |

The compiler constructs shared-core axiom objects. If `pyowl-core` lacks the necessary public
axiom construction/canonicalization seam, record an upstream dependency and do not introduce a
private parallel record type.

The off state is always present for a mutable object and compiles to an empty tuple. A pinned
mapping is either moved into immutable trusted axioms or governed by one legal state; the adapter
records which choice was made.

### Identity and canonicalization

- Normalize relation aliases through the existing typed relation vocabulary before state
  generation.
- State IDs hash compiler ID/version, mapping type, endpoints, modes, guards/qualifications, and
  canonical axioms. They exclude utility/model values.
- Revision IDs prefer existing stable mapping IDs. Hash-derived fallbacks include a deterministic
  duplicate index so duplicate mapping rows are not accidentally merged.
- State ordering is semantic and stable; no set/dict iteration order reaches an artifact.
- Every finite numeric value is validated before objective scaling or serialization.

### Provenance index

Maintain both directions:

```text
(revision_object_id, state_id) -> canonical generated axiom IDs
canonical generated axiom ID   -> exactly one (revision_object_id, state_id)
```

Equal axioms emitted by two distinct selected states cannot use axiom text alone as the key; use
an assertion occurrence/module ID while separately recording semantic axiom equality. The replay
path must reconstruct the selected modules without ambiguity.

### Policy and conflict records

`SafetyPolicyV1` is immutable and content-addressed. XR-WP1 implements query descriptions and
baseline record schemas, not their reasoning. A `StateConflictV1` contains nonempty selected state
literals, violation ID/family, explanation references, conditional literals, shrink provenance,
and a replay status. It rejects two states from the same one-hot revision object unless the
constraint is explicitly normalized as tautological/invalid.

### Certificate lifecycle

Implement an atomic builder with states `CREATED -> INVENTORY_FROZEN -> SEARCHED -> VERIFIED ->
FINALIZED`. Only XR-WP2 can attach an authorizing final oracle report. The writer validates status
field consistency; for example, `OPTIMAL_SAFE` requires `safe=true`, `optimal=true`, and
`objective==best_bound`; `SAFE_WITH_GAP` requires a non-null valid bound and derived gap; and
`SAFE_FALLBACK` requires an absent or invalid bound, `gap=null`, and the mandatory
no-optimality-claim warning (plus an information-loss warning when the incumbent is all-off).

Large inventories/conflicts are content-addressed JSONL artifacts. The certificate embeds their
schema/version/hash/count. Replay rejects missing, reordered-without-canonical-hash, or altered
artifacts.

## Exact-OM adapter contract

The adapter consumes a captured output, never reruns matching. Preserve:

- mapping identifiers, entity kinds, endpoints, relation, and score;
- per-channel evidence, uncertainty/margins, selector decisions, and rationales by reference;
- top-k alternatives with source artifact/rank/score;
- matcher/config/run/ontology provenance;
- user review and pinning information.

Missing optional evidence becomes an explicit availability flag, not zero. Unsupported relation or
entity types produce `INVALID_INPUT` or an explicitly excluded record according to a strict config;
they are never coerced to class equivalence.

## File changes

- add `exact/repair/domain.py`, `contracts.py`, `adapter.py`, `policy.py`, `certificates.py`;
- add `exact/repair/state_compilers/base.py` and `class_mapping.py`;
- add additive repair paths/accessors to `exact/runs/layout.py`, `manifest.py`, and `reader.py`;
- add `tests/repair/` fixtures/tests;
- add versioned JSON Schemas under `schemas/exact-repair/v1/` if the repository adopts checked-in
  schemas; otherwise generate them deterministically in tests and package them as resources;
- update public docs/changelog only when a callable public surface lands.

Do not edit matching explanation-store records or overload their schema.

## Verification

### Unit tests

- exact axiom output for every base family and both direction orientations;
- all-off emits no axioms;
- duplicate semantic states deduplicate deterministically;
- invalid IRIs, NaN/Inf utilities, unknown relations, wrong entity kinds, and illegal governance
  fail validation;
- stable hashes across process/hash seeds and JSON round-trip;
- complete bidirectional provenance for multi-axiom equivalence;
- status truth-table validation;
- manifest path safety, atomic writes, and historical run reading.

### Property-based tests

Generate bounded revision inventories and prove structural properties:

- exactly one off state for each mutable object;
- no two retained state IDs have equal normalized modules;
- reconstruct/compile is idempotent;
- one-hot assignment serialization selects exactly one known state per object;
- mutating any semantically relevant field changes the appropriate content hash.

### Review fixtures

Create minimal shared-core ontologies for every family listed in the common logical mechanism
suite. At this stage fixtures declare expected policy outcomes in a backend-neutral manifest; they
are not used to claim reasoner coverage.

## Acceptance criteria

XR-WP1 is done when:

1. the base state compiler and schemas are independently reviewed against the proposal table;
2. deterministic/hashing/property tests pass on supported Python versions;
3. every generated axiom occurrence has one state origin and every state round-trips;
4. certificate validation makes impossible safe/optimal status combinations unrepresentable;
5. repair-disabled Exact-OM outputs and imports remain unchanged;
6. layout-v1/v2 reader compatibility tests pass;
7. the logical fixture manifest and XR-E00 record schemas exist;
8. no Java, source-path reparse, duplicate ontology model, solver, or model dependency enters the
   base package.

## Experiment handoff

XR-E00 consumes schema IDs, fixture manifests, state inventory hashes, and the captured-alignment
adapter. XR-E01 later uses certificate replay and provenance audits. Passing this WP authorizes no
claim that the reasoner is complete, the loop is safe, or richer states improve preservation.
