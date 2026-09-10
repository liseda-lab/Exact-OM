# Historical implementation inventory and ordered work

This is the pre-implementation audit. See [PREPARATION-STATUS.md](PREPARATION-STATUS.md)
for current implementation and readiness evidence.

**Snapshot: 655f599e714e13d592f702f326ca5a36f6b50b2f, inspected 2026-09-07 and rechecked 2026-09-09.**
The code revision is unchanged at this specification update. No v2 runtime implementation has
been performed. The previous suite marked E00/E17 ready, 18 deferred and seven
deferred_unavailable; those were v1 declarations, not proof of v2 readiness.

The user now confirms all relevant OAEI/BioKG-Align datasets are available. Treat old missing
dataset/publication flags as stale, resolve the actual inventory, and distinguish a missing
local descriptor/path from unavailable data. The new outstanding data task is binding and
capability verification. Do not invent typed/NIL/property labels merely because a file exists.

## Shared code evidence

- exact/experiments/schema.py and harness.py implement strict v1 declarations, screen/confirm,
  provenance, existing selection, and same-fingerprint reuse. New case/budget/gate/recovery
  semantics require implementation in this existing workflow.
- exact/experiments/paper_metrics.py reconstructs paired source-level F1. Preserve this and add
  the frozen contrast/result-set/cluster handling rather than averaging per-source F1.
- exact/impl/trainer/checkpointing.py, audit_io.py, rationales.py and exact/runs/store.py provide
  reusable checkpoint/shard primitives. They are not a cross-version recovery planner.
- exact/core/actions/alignment.py infers local mode from candidate-file presence; the selector
  skips that mode, and non-greedy extraction refuses it. This is the first execution correction.
- exact/impl/models/pair_adaptive_scorer.py has a legacy gate when experimental gating is off,
  pair-ID quantile membership, and a changed aggregation for analytic_fitted.
- exact/impl/extraction.py optimizes raw assignment scores before post-thresholding.
- exact/impl/models/pair_adaptive_channels.py contains the quality/difference primitives and
  their singleton/empty-side behavior; these need the reviewed semantic controls.
- exact/impl/models/selector/listwise_llm.py and llm_gate.py provide useful primitives;
  grouped backend execution and actual fitting are still separate missing work.
- exact/tracks/builtin/biokg.yaml is a stale stub. Bind the user-available BioKG data explicitly.

Relevant existing tests include experiment_harness_test.py, experiment_replay_test.py,
experiment_paper_metrics_integration_test.py, semantic_runner_checkpoint_test.py,
selector_experiment_primitives_test.py, pair_adaptive_experiments_test.py, and
retrieval_experiments_test.py. The prior review ran 39 targeted primitive tests, not the full
campaign. Reconcile current test behavior and extend only where the new contracts need proof.

## Per-family inventory

| Family | Implemented or reusable at inspected baseline | Still to implement/integrate |
| --- | --- | --- |
| [E00](E00-harness.md) | Strict screen/confirm schemas, task/arm expansion, provenance, source-level metrics, same-fingerprint resume, inference/additional-model checkpoints, atomic explanation storage, and E17 composition helpers exist. | Explicit task-mode routing; v2 per-stage identities and readiness; cross-directory import; selective repair/reuse; budget and broad-search planning; reporting access ledger; per-role OpenRouter cost admission; relocated and interrupted end-to-end proof. |
| [E01](E01-global-extraction.md) | Greedy, mutual-best, stable marriage, component assignment, protected-match checks, and component-cap fallback primitives exist. | True global frozen-pool execution and threshold-aware accepted-edge assignment; current assignment applies threshold after optimization. |
| [E02](E02-anchor-rescoring.md) | Exact-anchor and graph evidence helpers exist; the bounded second pass is not integrated for every declared arm. | Soft-anchor competition, stage-level second-pass runner integration, explicit anchor provenance, self-confirmation exclusion, noise diagnostics, and checkpointed rescore. |
| [E03](E03-calibration-thresholds.md) | Platt/isotonic application and distribution-threshold primitives exist. | Grouped OOF fitting/export, correct train-pool features, donor-to-recipient application, risk/coverage reporting, and replayable threshold selection. |
| [E04](E04-nil-abstention.md) | Joint candidate/NIL probability primitives exist; full NIL writer/evaluator and fitted calibration remain incomplete. | Natural-NIL task binding, grouped fit, output/evaluator integration, three-way absence semantics, and optional listwise-none integration. |
| [E05](E05-retrieval-upgrades.md) | Bounded encoder/fusion/adaptive-k controls, pool diagnostics, and matched-pool selection primitives exist. | Bounded progressive orchestration, reusable embedding/index artifacts, per-role leakage checks, and an explicit G1 policy freeze; optional grounded query-rescue path. |
| [E06](E06-string-similarity-channel.md) | Bounded string metrics, conservative abbreviation matching, and provenance-bearing channel controls exist. | V2 case selection, stage replay, full feature attribution, and effective-cost/duplication diagnostics. |
| [E07](E07-listwise-llm.md) | Listwise prompt, categorical probability, order, and aggregation primitives exist; scorer grouped execution still rejects listwise modes. | OpenRouter grouped runtime, source-level eligibility, score-blind direct packets, no-summary comparison, bounded evidence requests, complete response/cost ledger, and source-level integration. |
| [E08](E08-attribute-polarity.md) | Attribute-bank controls and allowlisted signed-identifier primitives exist. | Task descriptors with real property/namespace semantics, cross-bank deduplication validation, and bounded conditional interaction assembly. |
| [E09](E09-hierarchy-semantics.md) | IC-weighted hierarchy overlap and sibling evidence primitives exist. | Feature-appropriate case binding, controlled representation/noise checks, and complete hierarchy-versus-lexical contribution reporting. |
| [E10](E10-fusion-selector-ablations.md) | Fusion controls and the existing listwise-linear/acceptance baseline exist; the earlier 45-cell/two-phase training orchestration is incomplete. | Cached-score constant replay, bounded two-phase selection, winner-plus-runner-up fitting, and immutable training artifacts. |
| [E11](E11-property-equivalence.md) | Entity-kind plumbing and typed I/O exist; property-specific evidence switches are not fully integrated. | Property evidence bundles/signatures, per-kind retrieval/fitting, actual reference inventory, and case-scoped evaluation. |
| [E12](E12-instance-equivalence.md) | Instance-kind plumbing exists; complete evidence bundles, shuffle and anchor crosses are incomplete. | Type/literal/relation bundles, kind-specific multi-view retrieval, grouped research partitions, deterministic degree/predicate-preserving shuffle, and instance evaluation. |
| [E13](E13-representation-robustness.md) | OWL/RDF/CSV adapters and closure interfaces exist; paired parity/enrichment harness is incomplete. | Lossless matched snapshots, feature/evidence parity comparison, explicit information-loss inventory, and independent materialization artifacts. |
| [E14](E14-typed-relations.md) | Heuristic/graph relation primitives and typed writers exist; trained typer, full bridge integration and coherence audit are incomplete. | Replace stale BioKG stub with the available data descriptor; grouped typed fitting, leave-query-bridge-out semantics, oracle-pair/full-pipeline evaluation and supported-profile reasoning audit. |
| [E15](E15-label-free-selection.md) | Mode resolution and several heuristic selector controls exist; mixed-capability matrix handling remains incomplete. | Strict per-component label-free execution with present train files, Otsu/margin and reciprocal controls in the true global path, and capability-correct comparison. |
| [E16](E16-cross-pair-transfer.md) | Strict artifact-consuming paths exist; donor training/export and transfer orchestration are incomplete. | Source-pair-only fit, held-out recipient application without silent refit, metadata for ontology/label overlap, and limited transfer reporting. |
| [E17](E17-promoted-stack-integration.md) | Composition, leave-one-out/interaction generation and paired global-F1 reporting helpers exist. | Development-only stack selection, new bounded final panel, explicit task mode across pool ablation, compatible refitting, fresh-role enforcement, and repair-aware result-set aggregation. |
| [E18](E18-supervised-reranking.md) | Current listwise-linear training/ranking and analytic controls exist; new objective/model artifact trainers are incomplete. | Grouped trainer/export for pairwise and channel-gating variants, fixed-feature comparisons, additive diagnostic, and acceptance interaction. |
| [E19](E19-supervised-fusion.md) | Artifact-backed analytic_fitted/learned_global consumers exist; fitting and adaptive providers are incomplete. Analytic_fitted currently changes the outer aggregation. | Neutral-parameter identity correction, grouped fitting/export, explicit family names, optional bounded adaptive weights, and raw quality-component handling. |
| [E20](E20-supervised-retrieval.md) | Fitted retrieval loaders and bounded cross-reranking exist; safe mining and immutable trainers are incomplete. | Source-group-safe negative mining, encoder/cross-encoder fitting and resumable state, matched-pool comparison, and early G1 orchestration. |
| [E21](E21-supervised-llm.md) | Canonical controls/guards exist; exemplars, counterfactual router labels, student training and fitted trust are missing. | Training-only exemplar retrieval, actual forced-call counterfactual data, net-benefit router fitting, gold-only/student comparison, and immutable teacher bindings. |
| [E22](E22-label-efficiency-and-mode-resolution.md) | Supervision modes/policy fields exist; grouped budget subsampling, active selection and fitted policy artifacts are missing. | Nested deterministic label budgets, component-specific effective-unit counts, one active-versus-passive comparison, and frozen crossover/policy fitting. |
| [E23](E23-graph-structure-supervision.md) | Graph-channel schema/guards exist; fitted graph-head production and controlled TBox input orchestration are missing. | One inductive structural head, degree/predicate-preserving control, controlled hierarchy removal, graph/profile provenance, and supported explanation integration. |
| [E24](E24-contrastive-channel-degeneracy.md) | Normalized, absolute, asymmetric and off difference primitives with diagnostics exist. | Supported/contradicted/unobserved semantics, zero contradiction authority for unjustified one-sided absence, controlled perturbations and valid typed asymmetry binding. |
| [E25](E25-llm-gate-viability.md) | Analytic/quantile/forced/oracle primitives and gate instrumentation exist; off control and development-ID transfer are defective. | Actual decision-off wiring, source-top-fraction and transferred-threshold policies, stratified forced judgments, decoupled trust, and budget-matched real-response oracle diagnostics. |
| [E26](E26-quality-proxy-validity.md) | Five sigma decomposition modes and lexical quality variants exist; component dumps and fitted reliability path remain incomplete. | Complete quality-component artifacts, duplicate/singleton/count-aware checks, candidate-level ambiguity quality, controlled entropy temperature and conditional reliability fitting. |

## Ordered implementation packages

1. **X0 — bindings and corrected vertical path.** Node profile, all-OpenRouter role routing,
   dataset/capability registry, explicit global/local execution, real training-pool features,
   off-control fix, quantile policy, same-family fusion identity, threshold-aware assignment.
   Deliver one correct NCIT–DOID train/development global/local run with full traces.
2. **X1 — checkpoint and repair reliability.** Component dependency graph and relevant code
   hashes; content-addressed artifacts; durable training/API state; interruption/relocation,
   selective repair, immutable attempts and current-result selection. Deliver the recovery
   acceptance tests before the long campaign.
3. **X2 — common fitting and evidence replay.** Source-group trainer, OOF artifacts, safe-negative
   inventory, resumable optimizers, quality/raw-evidence dumps, calibration/reranking/fusion
   providers. This unlocks several families without duplicate training infrastructure.
4. **X3 — feature-specific minimum treatments.** Property/instance signatures and pools,
   representation parity, typed BioKG relation fitting/closure, natural NIL, and bounded
   structural/anchor controls. Select cases from the available inventory; do not force D0.
5. **X4 — OpenRouter comparison and learning.** Per-source direct/listwise runtime, exact request
   ledger, bounded evidence acquisition, source routing and trust separation, counterfactual
   datasets, and the minimal E21/student/E22 recipes when their prerequisites pass.
6. **X5 — bounded campaign execution.** Per-arm/stage readiness, resource forecasts/admission,
   case-role locks, sequential screen gates, baseline/default dependency outputs, automatic
   mechanical G4 freeze, final-role access control, repair-aware aggregate reporting.

The ordering gives dependencies, not permission to delay testing until the end. Build one
integrated path early and add small working increments. Dataset binding and independent
primitives may proceed without waiting for every larger optional model.

## Readiness ledger required from the agent

For each family/arm/stage record: inspected commit, implemented paths, tests/operational proof,
missing implementation, required inputs/model/provider bindings, resource forecast, and status.
Statuses are planned, implementing, fixture_ready, screen_ready, confirm_ready, complete,
screened_out, inapplicable, blocked_input_resolution, deferred_budget, interrupted, failed,
superseded. Missing fit/runtime is implementing, not screened_out. Give exact reasons.

An arm can become screen_ready while its final reporting calibration or broader case binding
is still pending. Confirm_ready requires the final frozen design, compatible artifacts, data
roles and budget. Current v1 declarations are marked blocked to prevent accidental execution
of superseded treatments. Re-enable each migrated path only after evidence meets v2 acceptance.
