# XR architecture and shared contracts

**Status:** proposed; interface freeze at G0<br>
**Implementation namespace:** `exact.repair`<br>
**Schema namespace:** `exact-repair/*/v1`

This document turns the proposal's formal objects into repository-specific interfaces. It is
normative for implementation; the LaTeX proposal remains the motivation and mathematical source.
Where they differ operationally, this file targets the current Java-free Exact-OM architecture.

Where restatements differ, [README.md §Cross-cutting invariants](README.md#cross-cutting-invariants)
is normative.

## Architectural placement

Exact-Repair is a post-processing boundary. It consumes a completed provisional alignment and the
same source/target `pyowl_core.OntologySnapshot` objects used by Exact-OM. It returns a repaired
alignment plus a certificate. It must not be embedded in pair scoring or mutate matcher evidence.

```text
Exact-OM alignment + evidence + shared snapshots
                    |
             ExactOmRepairAdapter
                    |
   revision objects + frozen candidate-state inventory
                    |
       utility --> master solver <--> safety oracle
                              lazy state conflicts
                    |
       verified assignment + certificate + repaired alignment
```

The standalone entry point is implemented before pipeline integration. This makes every comparison
able to repair an immutable captured matcher output and prevents accidental rematching between
baselines.

## Proposed package ownership

```text
exact/repair/
  __init__.py                 # no eager optional imports
  contracts.py                # Protocols and capability records
  domain.py                   # immutable versioned value objects
  adapter.py                  # Exact-OM -> revision-object conversion
  policy.py                   # baseline and violation-query definitions
  certificates.py             # canonical serialization, hashing, replay
  orchestrator.py             # fail-closed counterexample-guided loop
  state_compilers/
    base.py
    class_mapping.py
  candidates/
    base.py
    guards.py
    replacements.py
  solvers/
    base.py
    exhaustive.py             # test/reference backend only
    <selected exact backend>.py
  reasoners/
    base.py
    shared_stack.py
  learning/
    features.py
    utility.py
    proposals.py
    structured.py
  evaluation/
    corruption.py
    queries.py
    metrics.py
```

Tests live under `tests/repair/`; research runners and frozen analysis code live under
`tools/repair/`. Training data and experiment outputs do not ship in the base wheel.

The base Exact-OM install remains usable without repair optimizer/reasoner/model extras. Optional
imports occur only after an explicit repair command or config enables them, and missing
dependencies produce actionable non-safe statuses or CLI errors.

## Domain model

All public records are immutable Pydantic models or frozen dataclasses with an explicit
`schema_version`. Canonical JSON uses UTF-8, sorted keys, normalized IRIs/OWL expressions, finite
numbers only, and SHA-256 content hashes.

### `RevisionObjectV1`

Required fields:

- `revision_object_id`: stable ID derived from mapping ID when available, otherwise from the
  normalized original mapping and occurrence index;
- `mapping_type`: exactly `class_mapping` in XR-1;
- original `source_iri`, `target_iri`, and normalized provisional relation;
- provisional confidence plus decomposed evidence and matcher/validator provenance;
- alternative endpoints with scores/provenance, never only an opaque top-k index;
- governance: `mutable`, `pinned`, or `proposal_only`;
- `compiler_id` and `compiler_version`;
- a nonempty ordered list of candidate-state IDs.

Pinned mappings become trusted core axioms or one-state governed objects according to a recorded
policy. The choice must be explicit; pinning must not be silently discarded by the all-off
fallback.

### `ClassStateV1`

Required fields:

- state/revision IDs and state family;
- selected source and target IRIs;
- forward/backward mode (`off`, `plain`, or a normalized guard reference);
- independently listed qualification components;
- candidate source and prefilter decisions;
- canonical generated axioms;
- profile requirements and compiler content hash;
- integer utility, uncertainty diagnostics, and complexity features.

State IDs are content-derived from semantic inputs and compiler version; learned utility is not
part of the state ID. The mandatory all-off state emits zero axioms. Two states with equal
normalized axiom sets for one revision object are deduplicated with provenance merged.

### `SafetyPolicyV1`

Fields:

- `require_consistency`;
- monitored named-class signature and relative baseline unsatisfiable-set semantics;
- finite forbidden-entailment query inventory and baseline-entailment hash;
- active-guard non-vacuity requirement;
- policy/query compiler versions;
- reasoner completeness requirements and per-family budgets, plus per-query budgets only for the
  explicitly finite, budget-capped forbidden-entailment inventory.

Relative named-class coherence is evaluated with one complete classification or equivalent batch
operation, followed by an unsatisfiable-set difference against the baseline. It is not implemented
as one satisfiability call per monitored class. Per-query execution applies only to the explicit
finite forbidden-entailment list.

XR-1 permits only finite monotone violation families. Non-redundant guards may be a generation
filter or utility penalty, but if made a hard conditional policy its state-literal semantics must
be versioned and tested. SHACL, closed-world constraints, and nonmonotonic rules require a new
policy version and proof review.

### Status contract

| Status | Alignment usable? | `safe` | `optimal` | Required evidence |
|---|---:|---:|---:|---|
| `OPTIMAL_SAFE` | yes | true | true | complete final verification and objective bound equals value |
| `SAFE_WITH_GAP` | yes | true | false | complete verification of returned incumbent plus valid bound/gap |
| `SAFE_FALLBACK` | yes, with mandatory no-optimality-claim warning | true | false | verified incumbent with no valid retained optimality bound; all-off is the minimal case and additionally requires an information-loss warning |
| `UNSAFE_CORE` | no repaired alignment | false | false | baseline violation in immutable core or pinned trusted axioms |
| `NO_GOVERNED_SOLUTION` | no | false | false | logical fallback exists, but no safe assignment satisfies explicit coverage/pinning governance constraints |
| `ORACLE_UNKNOWN` | no | false | false | no verified incumbent is available after timeout, unsupported feature, incomplete adapter, or missing valid conflict; a verified incumbent routes to `SAFE_WITH_GAP` when a valid bound is retained, otherwise to `SAFE_FALLBACK` |
| `INVALID_INPUT` | no | false | false | schema, provenance, profile, or governance validation failure |

No other status may set `safe=true`. `SAFE_WITH_GAP` never contains an unverified final master
candidate. If a verified incumbent exists when a later oracle call fails, the certificate records
the failure and returns that incumbent with a safe status and correct bound semantics.

## Protocols

### State compiler

```python
class StateCompiler(Protocol):
    compiler_id: str
    compiler_version: int

    def compile(self, revision: RevisionObjectV1, state: StateSpecV1) -> CompiledStateV1: ...
    def validate(self, state: CompiledStateV1, capabilities: OracleCapabilities) -> ValidationReport: ...
    def reconstruct(self, state: CompiledStateV1) -> tuple[OWLAxiom, ...]: ...
```

Compilation must be pure and deterministic. XR-1 class axioms are constructed in the shared core
model, not serialized to a path and reparsed. Provenance is maintained outside OWL annotations so
it cannot affect Direct Semantics.

### Repair reasoner

The current `exact.ontology.reasoning.ReasonerProtocol` is hierarchy-only and is **not** a repair
oracle. XR-WP2 adds a separate capability-checked protocol:

```python
class RepairReasoner(Protocol):
    @property
    def capabilities(self) -> OracleCapabilities: ...
    def inspect_profile(self, core, states, policy) -> ProfileReport: ...
    def check_baseline(self, core, policy) -> BaselineReport: ...
    def check_assignment(self, core, assignment, policy) -> SafetyReport: ...
    def explain(self, violation, core, assignment, limit) -> tuple[AxiomExplanation, ...]: ...
    def entails(self, core, assignment, query) -> EntailmentResult: ...
    def satisfiable(self, core, assignment, expression) -> SatisfiabilityResult: ...
    def close(self) -> None: ...
```

Capabilities declare supported OWL profiles, policy query families, explanation method,
incrementality, worker-wire compatibility, and whether results are complete. An asserted
hierarchy fallback is never a complete safety oracle. pyELK may authorize only policy/profile
combinations for which its adapter declares completeness; expressive cases require pyHermiT or
another complete shared-snapshot adapter.

Upstream pyELK and pyHermiT are committed to complete reasoning and full justification generation
at parity with the original Java reasoners, over the shared snapshot model without reparsing.
Adapters may assume those capabilities exist upstream but must still verify them through capability
negotiation and the XR-WP2 conformance/throughput audit; a failed verification is an upstream
regression that blocks the affected path until fixed. Exact-Repair must not infer capabilities from
the package name alone.

### Master solver

```python
class RepairSolver(Protocol):
    @property
    def capabilities(self) -> SolverCapabilities: ...
    def add_revision_object(self, object_id, states) -> None: ...
    def add_hard_constraint(self, constraint) -> None: ...
    def set_lexicographic_objective(self, tiers) -> None: ...
    def add_conflict(self, conflict) -> None: ...
    def solve(self, budget) -> SolverResult: ...
    def incumbent(self) -> Assignment | None: ...
    def bound(self) -> IntegerBound | None: ...
```

The adapter must enforce one-hot selection, integer objectives, deterministic tie-breaking, and
incremental no-goods. XR-WP2 includes an exhaustive reference solver for small tests and one
selected exact backend. A backend can claim exactness only when it exposes a valid optimality
result/bound. Floating-point model scores are frozen and scaled before solving.

A backend adapter may realize the enumerated one-hot space through an equivalent factorized
encoding (per-component Booleans plus compatibility constraints) when the enumerated cross-product
of composite states is large. The bijection between factorized solutions and enumerated states must
be tested, and certificates always record the selected composite state, never raw factor literals.

Endpoint replacement can let two distinct revision objects select states whose canonical generated
axiom sets are semantically identical. The engine must detect such cross-object collisions in the
frozen inventory and either add a hard constraint forbidding simultaneous selection or apply a
recorded canonical-axiom-level utility correction. The additive objective must never count
identical semantic content twice.

### Utility and proposal models

`StateUtility` returns integer utility plus uncertainty/provenance. `CandidateProposer` returns a
ranked finite inventory from a closed entity and constructor grammar. Both have deterministic
hand-coded implementations. Learned models are optional plugins/checkpoints and never receive an
API capable of marking an assignment safe.

## Counterexample-guided orchestration

The normative loop is:

1. validate input contracts, snapshot identity/provenance, state compilation, and capabilities;
2. check the immutable core and freeze baseline policy data;
3. when configured, extract a locality-based module of the immutable core for the frozen
   inventory signature and run module-restricted checks under the per-family rules of
   [`03-module-soundness.md`](03-module-soundness.md): module results may find violations for
   every family, but they verify incumbents only for families with a proven completeness lemma —
   under the default full-signature coherence policy, incumbent verification always runs on the
   full integration;
4. generate/freeze states, integer utilities, governance constraints, and their inventory hash;
5. build one-hot master constraints, optionally seed replay-valid eager conflicts from
   deterministic structural patterns, and verify the all-off assignment;
6. request the best not-yet-excluded master assignment;
7. check all enabled policy families with a complete oracle;
8. on `SAFE`, retain the assignment as a verified incumbent and stop only with zero gap or budget;
9. on `UNSAFE`, project a sufficient explanation for every reported violation to selected state
   literals — batch cut extraction, not one cut per iteration — then validate each projected
   conflict, optionally shrink it, persist it, and add its no-good;
10. on `UNKNOWN`, switch only to a configured compatible complete path; otherwise return an existing
    verified incumbent as `SAFE_WITH_GAP` when a valid bound is retained or `SAFE_FALLBACK` when it
    is not, and use `ORACLE_UNKNOWN` only when no verified incumbent is available; never add a
    speculative cut;
11. run final rebuild-mode verification and write the certificate atomically.

A cut is valid only if replaying its immutable axioms, generated state modules, and conditional
state literals reproduces a policy violation. Minimality is optional. Missing explanation support
may use a black-box state-deletion procedure, but failure to produce a sufficient conflict is
`ORACLE_UNKNOWN`.

## Configuration contract

Add an optional strict `RepairConfig` under config v2 without changing behavior when absent:

```yaml
repair:
  enabled: false
  mode: exact                 # exact | anytime
  policy:
    require_consistency: true
    relative_named_class_coherence: true
    forbidden_entailments: null
    guard_non_vacuity: true
  states:
    families: [off, equivalence, forward, backward]
    max_states_per_object: 4
    guard_depth: 0
    guard_constructors: 0
    guards_per_direction: 0
    replacements_per_side: 0
  oracle:
    reasoner: auto
    require_complete: true
    timeout_seconds: null
  solver:
    backend: auto
    timeout_seconds: null
  utility:
    kind: hand_coded
    checkpoint: null
    integer_scale: 10000
  budgets:
    wall_seconds: null
    reasoner_calls: null
    memory_mb: null
```

Defaults expose only base states and keep repair off. Rich states, finite conservativity queries,
learned utilities, and production budgets are enabled explicitly and captured in the certificate.
Config migration must reject ambiguous legacy/unknown keys; it must not guess safety settings.

## CLI and API

The standalone command is the first public delivery seam:

```text
exact repair --source SOURCE --target TARGET --alignment ALIGNMENT \
  --config CONFIG --output RUN_DIR
exact repair replay RUN_DIR/repair/certificate.json
```

The Python API is a lazily imported `RepairRunner` accepting existing snapshots/providers,
mapping records, evidence references, policy, and budgets. Pipeline integration later adds an
explicit opt-in to `AlignmentRunner`; it must pass in-memory snapshot identity rather than source
paths.

Exit codes distinguish successful safe output, safe fallback, unsafe core, unknown, and invalid
input. Logs are diagnostics; the certificate is the authoritative result.

## Run artifacts

Extend layout-v2 additively with a `repair/` subtree registered in `run_manifest.json`:

```text
repair/
  certificate.json            # exact-repair/certificate/v1
  candidate_states.jsonl      # exact-repair/state/v1
  conflicts.jsonl             # exact-repair/conflict/v1
  solver_trace.jsonl          # bounded, versioned, optional
  alignment/                  # selected repaired mappings/axioms in requested formats
  metrics.json                # descriptive run metrics, not confirmatory aggregation
```

The certificate contains input/snapshot/policy/inventory hashes; compiler, oracle, solver, model,
and software versions; selected state per object; generated axioms; final verification results;
all safe/optimal fields; objective/bound/gap; failure/fallback events; seeds; and artifact hashes.
Large state/conflict inventories may be referenced by manifest hash rather than embedded.

Repair explanations are not written into the matching explanation shards as though they were model
rationales. `RunReader` gains explicit repair accessors and remains able to read historical runs.

## Certificate replay

Replay must work without the learned model or master solver:

1. verify schema and all referenced hashes;
2. load/coerce the recorded ontology snapshots under the recorded import policy;
3. reconstruct each selected state using the recorded compiler version;
4. assert exact equality with the recorded canonical generated axioms;
5. reconstruct the policy and baseline data;
6. run a configured complete verifier in rebuild mode;
7. compare per-query results and emit a signed/hashed replay report.

Replaying the same adapter is necessary but not “independent verification.” Experiments label a
check independent only when it uses a separately implemented complete path or reasoner and the
same Direct Semantics/profile coverage.

## Compatibility and security requirements

- No credentials, absolute machine paths, object IDs, prompts containing protected ontology text,
  or unrestricted model output enter certificates.
- Imports are resolved according to the shared snapshot manifest; replay never fetches mutable
  network resources implicitly.
- Candidate OWL expressions are parsed/constructed from a closed grammar and canonical inventory.
  Free-form LLM text never becomes an axiom.
- Artifact writes are atomic and manifest paths cannot escape the run directory.
- New schemas follow additive/minor and breaking/major versioning. Readers reject newer unknown
  major versions with an actionable error.
- Existing alignment outputs remain byte-identical when `repair.enabled=false`.

## Contract acceptance

The architecture is accepted only when tests demonstrate:

- unchanged Exact-OM outputs and optional-dependency behavior with repair disabled;
- exact in-process snapshot identity or a verified content-addressed shared-core integration view;
- deterministic state inventory and objective hashes across processes;
- all-off feasibility when and only when the immutable core satisfies the policy;
- no safe status from injected unknowns/timeouts/unsupported features;
- state-to-axiom and explanation-to-conflict provenance completeness;
- exhaustive solver parity on tractable instances;
- certificate round-trip and rebuild-mode replay;
- historical `RunReader` compatibility;
- no Java, second ontology model, path reparse, or unrestricted generated OWL.
