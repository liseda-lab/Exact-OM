# Explanation framework implementation programme

**Revision: 2026-09-20. Status: approved design direction; implementation and operational verification pending.** Audited baseline: `e81865aceed0cd1257580097bc788329cd9084ae`. Reconcile this inventory with the implementing checkout. Specifications, source inspection, raw-file profiles and a small library probe are not completed backend features.

Build two products sharing one explanation engine: an exploration app (local bundle import and a fixed-bundle online demo), and a complete Render-hosted study application. The study uses partial candidate ranking with explicit none/insufficient-information responses and both answer-present and answer-absent cases. Ontology context, recorded matcher evidence, decision history and generated explanation are distinct components, not four separate products. Neither fluent prose nor agreement with Exact establishes human benefit.

## Execution sequence and authority

1. One independent backend review informs this suite; its findings are captured in [evidence](evidence/independent-backend-review.md). The accepted product/study expansion also has a [design and persistence review](evidence/study-amendment-review.md).
2. **B0: shared contracts, input bindings and real fixtures.** No frontend-specialist dependency.
3. **B1–B4: Exact integration, ontology context, local import, backend/study APIs and generated explanations.** Independent work may overlap after B0; deliver one working class-pair path early.
4. **B5: verify backend readiness and publish its handoff.** Pass the objective gates in [07](07-validation-and-handoff.md).
5. **F1: specialized frontend agent performs full design and implementation**, for both products using [08](08-frontend-implementation.md) and 10–13. Do not engage that agent for an earlier design/review phase. Integrate and validate the whole tool after F1.

The user’s instructions take precedence. This suite governs the new explanation/context and user-interface work. It extends WP-K/WP-L and supersedes their selected-graph/limited-context behavior where explicitly described; it does not undo shared OWL-stack ownership, run storage or behavior-preserving matching contracts. [Experiment specifications](../experiments/README.md) continue to own result-changing matching methods and campaign budgets. [Exact-Repair](../exact-repair/README.md) remains separate and is not a prerequisite. API/schema semantics in 01 and recovery/redaction semantics in 06 govern all packages; implementation-status text cannot weaken them. The accepted 2026-09-20 product/study amendment in 10–13 governs study ranking, anonymous resumption, deployment boundaries and questionnaires, superseding the earlier accept/reject-first study and external-survey assumptions. Historical evidence reports describe the earlier review, not overriding requirements.

## Read and implement

| File | Purpose / owner |
|---|---|
| [00 — Status and scope](00-status-and-scope.md) | Existing foundations, missing work, package ledger |
| [01 — Shared contracts](01-contracts.md) | B0: identities, facts, decisions, collection and generation semantics |
| [02 — Exact integration](02-exact-integration.md) | B1: versioned exports, decision stages and provenance |
| [03 — Ontology context](03-ontology-context.md) | B2: snapshots, index, hierarchy, restrictions and annotations |
| [04 — Backend API](04-backend-api.md) | B3: paginated services, separation from generation, packaging |
| [05 — Generated explanations](05-generated-explanations.md) | B4: independent profiles and comparative assessments through OpenRouter |
| [06 — Recovery and study controls](06-recovery-and-study.md) | All backend packages: checkpoints, selective invalidation, visibility |
| [07 — Validation and handoff](07-validation-and-handoff.md) | B5: executable evidence, resource checks, frontend admission |
| [08 — Frontend implementation](08-frontend-implementation.md) | F1: complete specialist assignment after B5 |
| [09 — Data and case plan](09-data-and-cases.md) | Dataset pins, bounded development, wider validation |
| [10 — Products and bundles](10-products-and-bundles.md) | Two products, local import, fixed demo, portable packages |
| [11 — Ranking study](11-study-design-and-ranking.md) | Conditions, mixed cases, assignment, response and metric semantics |
| [12 — Study service](12-study-service-and-render.md) | Reusable invitations, durable state, telemetry, administration and Render |
| [13 — Questionnaires](13-study-questionnaires.md) | Setup, participant questions, per-case resource use and final feedback |
| [AGENT-HANDOFF](AGENT-HANDOFF.md) | Assignment and completion rules |

[protocol/development.json](protocol/development.json) is a **design blueprint**, not a configuration accepted by the current runtime. B0 defines strict stage-scoped execution locks: resolve every dependency before its consuming stage runs. Future run/model bindings may remain pending without blocking independent context preparation; no stage may execute with an unresolved required input. [contract.schema.json](protocol/contract.schema.json) defines the target shared wire primitives, not every future route response. Runtime response models/OpenAPI must extend these primitives and cover all resources in 01/04; extra route fields do not waive invariants. Fixtures marked illustrative or raw-profile evidence must never be presented as fresh matcher output.

## Minimum deliverable and boundaries

Required first delivery: NCIT–DOID classes, candidate-independent ontology context, usable hierarchy/axiom information, current Exact score/evidence and final-decision trace, independent entity profiles and balanced comparison, resumable preparation, portable local bundle import, fixed-demo enforcement, complete durable study APIs (invitations, forms, allocation, rankings, events and exports), tested deployment boundaries, and a reproducible frontend handoff. All generative calls use OpenRouter. Viewing an artifact must not load models, allocate a GPU, call a provider or regenerate results.

Optional inference, full OWL justification, exhaustive OWL prose templates, complete KGA/property studies, counterfactual rescoring and production logical repair do not block B5. Unsupported expressions retain lossless structured/original form. Unsupported features have honest status; no fake probabilities, relations, context or model output.

Implementation/preparation is separate from the 336–504 node-hour matching campaign. Target one node with 64 GB RAM (128 GB alternate) and one RTX 5090; serving is CPU-only. Use small development sets, measure full-snapshot indexing once, and reuse it. Final human-study recruitment/execution is separate; do not contact participants or submit external messages through this assignment.
