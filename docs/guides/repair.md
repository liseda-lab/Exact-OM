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

Soft features remain separate from asserted axioms. Scores are neither assumed calibrated nor used as semantic truth. Reference answers, corruption traces, teacher outcomes and future explanations are excluded from deployment evidence, with omissions recorded. Without `--model`, the CLI objective is an explicit edit-cost control. Bounded retrieval uses ontology labels/definitions, structural neighborhoods, typed matcher alternatives and pre-decision diagnosis supports. Its exact menus, omissions and evidence identity are captured for replay.

## Actions and ontology occurrences

`exact.repair.candidates` supplies keep/delete, relation-aware directional weakening, endpoint replacement, subclass specialisation, necessary conditions, complex equivalence, and compatible composites. Eligible ontology objects support disjointness removal, superclass-conjunct removal, subclass specialisation, and superclass/domain/range/existential-filler generalisation. Direct fixed subclass premises justify the initial generalisation menu.

`ontology_occurrences(source_snapshot, target_snapshot)` exposes exact side/document/occurrence identities. To enable an ontology edit, construct a `RevisionObjectV2` for that occurrence and pass it through `ontology_objects` to `prepare_repair`. `ontology_candidates` supplies its alternatives. An unchanged import or another candidate emitting the same axiom preserves that axiom. Ontology files are never overwritten by applying a repair view.

Every candidate emits its complete replacement. Specialised antecedents activate satisfiability checks of the whole expression. Non-class mappings initially have typed keep/delete menus. Fixed unsupported constructs remain in the logical input and can prevent complete verification.

## Learned proposals and training

`build_observable_graph` retains typed entities, axiom/expression roles, revision objects, diagnosis supports and optional evidence channels. `RepairModel` provides HGT, matched R-GCN and no-graph controls, shared object/candidate attention, syntax-aware candidate values and optional signed pair factors.

Direct grammar generation represents template, retained-direction and canonical expression slots. PySDD compiles their Boolean constraints without first enumerating complete expressions or candidate bundles. The conditioned mixture uses exact normalisers and posterior component weights. If several encodings emit the same bundle and activations, their probability masses are summed. Only sampled replacements and mandatory controls need completed-candidate scoring.

Use a frozen checkpoint through either standalone or sequential input:

```bash
exact repair --source source.owl --target target.owl \
  --matching-run output/run --model training/model.pt \
  --output output/repair.json --proposal-arm grammar_mixture \
  --proposal-seconds 30 --compile-seconds 10 --candidate-cap 64
```

In memory, call `bounded_freeze_neural_round(problem, model, ...)`; a successful result contains `.value.problem` and `.value.objective` for `repair`. The model snapshot is prepared in the caller; the worker deadline covers retrieval, graph encoding, compilation, sampling and scoring. The checkpoint CLI loads and snapshots inside the supervised worker as well. `freeze_neural_round` is the local variant for an already supervised training/campaign stage.

Every keep/delete/direction/endpoint control remains deterministic, with at least one representative of each available action family. Insufficient caps fail explicitly. If checkpoint loading or proposal generation fails, the CLI records the failure and attempts the captured pool with its frozen objective within the remaining budget. It never relaxes grammar constraints or reports the failed learned arm as successful.

The comparison controls use the same menus/templates/bounds: bounded enumeration, uniform constrained sampling, one conditioned product, conditioned mixtures, and unconstrained generation with rejection. Enumeration and rejection do not pay for circuit compilation. Circuit construction still has exponential worst cases; these implementations do not establish which approach is fastest on a benchmark.

`exact.repair.learning` supplies typed symbolic probes, masked partial teachers, complete-cache marginals, whole-repair benefit/ranking losses and nonnegative preference fitting. Generated parents vary connected paths, branching, explanation overlap, expression depth, redundant/complementary interactions, mixed compositions and observable evidence. Corruptions, renamings and controls share their clean parent's split.

Preparation and training are separate explicit commands:

```bash
python -m tools.repair.train --protocol specs/exact-repair/protocol/smoke.json \
  --prepare-only --output preparation
python -m tools.repair.train --protocol specs/exact-repair/protocol/smoke.json \
  --prepared preparation/preparation.json --output training
```

`--case-limit` permits smaller conformance preparation; use it when checking the workflow without preparing a whole smoke schedule. `--real-manifest` accepts explicit local source/target files, observed/clean alignments, permitted probes and declared splits; no dataset is downloaded. A separate adaptation run requires a generated checkpoint via `--warm-start`. Teacher labels and clean intentions remain outside deployment features.

Checkpoints use periodic decoded development regret on a complete common teacher inventory, with generated useful-candidate availability recorded separately. Novel generated bundles without cached policy labels remain unknown. Training/development groups, model/config hashes, grammar/retrieval schemas and adaptation provenance accompany the checkpoint. Held-out cases cannot select it.

`tools.repair.compare_proposals` compares generation controls; `tools.repair.run_study --matrix` declares matched action, scoring and model controls over captured inputs. Missing models and datasets remain unavailable rows. `tools.repair.report_study` computes paired effects over structural/pair groups and includes unresolved-case counts. A study schedule records its candidate/objective artifacts and declared splits and resumes only when their identities match. `tools.repair.prepare_study` validates local Conference/Bio-ML capture identities and release declarations, preserves whole-ontology holdouts, and reports missing scheduled assets without downloading them. Paired reports can read protocol confidence levels, group resampling and Holm-adjusted contrasts.

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

The supervisor bounds compiler/solver/reasoner calls and retains completed incumbent evidence in the parent. Nested workers share a process group for cancellation. Optional `--memory-mb` monitors Linux worker-tree RSS; its sampled cap excludes parent-owned inputs and device memory. Graph node/edge/explanation/text limits record omissions. Stage timeouts, memory limits and crashes remain visible. Cleanup and artifact persistence have their own bounded allowance. A timeout never proves infeasibility, and absence of an incumbent produces no repaired alignment.

The `repair.json` artifact contains versioned, content-hashed input, objective and result records. Complex output is the canonical shared-core OWL bundle in the record; it is not silently projected to a simple mapping table. Loading v1 repair records fails explicitly. `--problem` accepts the input/objective bundle for a recorded inventory. `replay_safety` reconstructs and checks the selected theory without the neural model or optimiser. `replay_optimality` performs a fresh bounded exact search rather than trusting a copied zero gap.

## Small conformance checks

```bash
poetry run pytest tests/repair_*_test.py -q
poetry run python -m unittest discover -s specs/exact-repair/reference -q
poetry run python specs/exact-repair/reference/validate_protocol.py
```

These tests exercise small semantic fixtures, brute-force finite comparisons, circuit distributions/gradients, feature isolation, typed learning, CLI replay and process failures. They are not Conference/Bio-ML evaluation, throughput measurements or evidence of learned model quality. Full-scale experiments and model-quality claims require separate, explicitly scheduled work.

See the [dated validation record](../project/repair-validation.md) for executed checks and the local optimized dependency revisions used.
