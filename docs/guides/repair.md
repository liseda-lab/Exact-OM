# Exact-Repair

Exact-Repair is an opt-in implementation of the [XR-2 contracts](https://github.com/liseda-lab/Exact-OM/tree/dev/specs/exact-repair). It consumes an alignment from any matcher and returns complete replacement OWL axioms, a separate ontology patch, and verification/search evidence. Ordinary matching commands do not enable repair.

Install its optional dependencies:

```bash
poetry install --extras repair
```

The qualified component versions are PySAT 1.8.dev24 (RC2/Glucose3), PySDD 1.0.6, torch-geometric 2.6.1, and the shared pyowl-core/pyHermiT/pyELK 0.2 interfaces. Neural modules and external solvers are imported only when used.

## Standalone and sequential use

Run repair directly on a supplied alignment:

```bash
exact repair --source source.owl --target target.owl \
  --alignment alignment.json --output repair.json \
  --seconds 60 --stage-seconds 10
```

Alignment JSON accepts the existing `Src`, `Tgt`, `Score`, `Relation`, and `Kind` columns. Scores may be absent, but supplied scores must be finite. TSV, CSV and OAEI RDF inputs also use the existing readers. Relations are `=`, `<`, or `>`; property and individual mappings retain their entity types.

After an Exact matching run, use its saved output:

```bash
exact repair --source source.owl --target target.owl \
  --matching-run output/run --output output/repair.json
```

This reuses the captured alignment, full explanations, local alternatives, source decisions, and available inference checkpoint/audit/candidate channels. `--evidence evidence.json` adds optional deployment features. In-memory callers can supply every available matching channel without first writing a run:

```python
from exact.repair import prepare_repair, make_objective, repair

problem = prepare_repair(
    source_snapshot, target_snapshot, provisional_mappings,
    matcher_identity="my-matcher", evidence=matching_features,
)
objective = make_objective(
    problem.objects,
    profile=(("edit", 1.0), ("delete", 1.0)),
)
result = repair(problem, objective)
```

Use a `if __name__ == "__main__":` guard for scripts calling repair, because native operations run in spawned processes. Existing shared snapshots are retained at the preparation boundary; repair does not require a second OWL parser or a matching invocation.

Soft features remain separate from asserted axioms. Scores are neither assumed calibrated nor used as semantic truth. Reference answers, corruption traces, teacher outcomes and future explanations are excluded from deployment evidence, with omissions recorded. The default CLI objective is an explicit edit-cost control; it does not claim to be a trained repair model.

## Actions and ontology occurrences

`exact.repair.candidates` supplies keep/delete, relation-aware directional weakening, endpoint replacement, subclass specialisation, necessary conditions, complex equivalence, and compatible composites. Eligible ontology objects support disjointness removal, superclass-conjunct removal, subclass specialisation, and superclass/domain/range/existential-filler generalisation. Direct fixed subclass premises justify the initial generalisation menu.

`ontology_occurrences(source_snapshot, target_snapshot)` exposes exact side/document/occurrence identities. To enable an ontology edit, construct a `RevisionObjectV2` for that occurrence and pass it through `ontology_objects` to `prepare_repair`. `ontology_candidates` supplies its alternatives. An unchanged import or another candidate emitting the same axiom preserves that axiom. Ontology files are never overwritten by applying a repair view.

Every candidate emits its complete replacement. Specialised antecedents activate satisfiability checks of the whole expression. Non-class mappings initially have typed keep/delete menus. Fixed unsupported constructs remain in the logical input and can prevent complete verification.

## Learned proposals and training

`build_observable_graph` retains typed entities, axiom/expression roles, revision objects, diagnosis supports and optional evidence channels. `RepairModel` provides HGT, matched R-GCN and no-graph controls, shared object/candidate attention, syntax-aware candidate values and optional signed pair factors.

`freeze_neural_round(problem, model, ...)` connects shared proposal logits to PySDD conditioning, samples candidates, preserves elementary controls and action-family representatives, computes completed-candidate benefits, and freezes the integer objective before exact selection:

```python
from exact.repair.pipeline import freeze_neural_round

round = freeze_neural_round(problem, model, profile=profile, seed=13)
result = repair(round.problem, round.objective)
```

The current circuit route compiles a bounded-enumerated canonical language. Its support, normalisers, mixture posterior, likelihood and gradients are checked against tiny exhaustive references. It does not establish scalability of compiling a large grammar. Empty distributions and insufficient candidate budgets fail explicitly.

`exact.repair.learning` supplies typed symbolic probes, complete/partial teacher caches, exact marginals only for complete caches, feasible-anchor benefit differences and ranking losses, grouped splits, and nonnegative preference fitting. `tools/repair` contains the generated-corpus, training and comparison entry points. Training and study runs are explicit actions; importing repair or invoking the default CLI never starts them.

## Verification, bounds and artifacts

The default policy monitors the complete original named-class signature, consistency, hard required/prohibited queries and every activated expression. Source-only exceptions require frozen source evidence and are proved again during verification/replay. An already verified input is returned unchanged by default.

| Field | Meaning |
| --- | --- |
| `logical_status` | Whether the returned assignment is completely verified |
| `verification_scope` | Full supported OWL, a complete supported fragment, or partial detection |
| `search_status` | Finite-pool optimum, incumbent with gap, exhausted infeasible pool, or unresolved |
| `candidate_coverage` | The declared generated or sampled universe |
| `lower_bound`, `upper_bound` | Exact integer objective bounds, including unresolved alternatives |
| `pending` | Assignments with unknown verification; these are never logical cuts |

The supervisor bounds solver/reasoner calls and retains completed incumbent evidence in the parent. Stage timeouts and crashes remain visible. Cleanup and artifact persistence have their own bounded allowance. A timeout never proves infeasibility, and absence of an incumbent produces no repaired alignment.

The `repair.json` artifact contains versioned, content-hashed input, objective and result records. Complex output is the canonical shared-core OWL bundle in the record; it is not silently projected to a simple mapping table. Loading v1 repair records fails explicitly. `--problem` accepts the input/objective bundle for a recorded inventory. `replay_safety` reconstructs and checks the selected theory without the neural model or optimiser. `replay_optimality` performs a fresh bounded exact search rather than trusting a copied zero gap.

## Small conformance checks

```bash
poetry run pytest tests/repair_*_test.py -q
poetry run python -m unittest discover -s specs/exact-repair/reference -q
poetry run python specs/exact-repair/reference/validate_protocol.py
```

These tests exercise small semantic fixtures, brute-force finite comparisons, circuit distributions/gradients, feature isolation, typed learning, CLI replay and process failures. They are not Conference/Bio-ML evaluation, throughput measurements or evidence of learned model quality. Full-scale experiments and checkpoint selection remain separate, explicitly scheduled work.

See the [dated validation record](../project/repair-validation.md) for executed checks and existing environment limitations.
