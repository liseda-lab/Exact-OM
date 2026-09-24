# Implementation inventory and package boundaries

**Observed status at `e81865a`, 2026-09-19. All B/F acceptance gates are pending.** Verify symbols at current HEAD before work; do not assume this inventory is timeless.

Current implementation evidence is maintained in the [status ledger](../../docs/verification/explanation-framework-status.json) and [backend verification report](../../docs/verification/explanation-backend-report.md). The dated inventory below remains the historical baseline.

| Foundation | Implemented and reusable | Required work |
|---|---|---|
| Pair records | `exact/impl/models/pair_adaptive_scorer.py`: explanation schema v3, kind/IRI, scores, quality, contributions, selected evidence | Typed versioned inspection adapter; preserve decision stages and original-fact links |
| Selected evidence | `pair_adaptive_channels.py`, `pair_adaptive_evidence.py`: endpoint IRIs, many predicate IRIs, comparison links | Semantic IDs; axiom/projection provenance; no label-only identity recovery |
| Selection | `exact/impl/trainer/overlays.py`: pair/selector values, threshold/member status, winner/reason | Candidate export retains retrieval, extraction and NIL fields; precise stage semantics and competitors |
| Relation typing | `exact/io/relations.py`: optional named class/property graph closure and directional evidence | Expose anchor assumptions; no default `=`/1.0 certainty claim; individuals unsupported |
| Artifacts | `exact/runs/store.py`, `reader.py`; trainer rationale checkpoints | Reuse sharding/readers, add context/text dependency keys, migration and restart verification |
| Ontology | `exact/ontology/store.py`: retained pyowl-core snapshot | Rich context adapter over snapshot indexes; no second parser or matcher feature facade as full context |
| Inspection API | `exact_inspect/app.py`, `bundles.py`; graph/source/node routes | Independent entity routes, bounded paging/caches, per-pair loading, robust completeness and redaction |
| Current context export | Limited definitions/parents/children and selected-IRI cache | Exact predicate registry, complete scope/index, metadata/provenance, no partial-cache masking |
| LLMs | OpenRouter routing, pair brief, decision-conditioned rationale | Separate candidate-blind profiles and score-blind comparison; grounding and explicit regeneration |
| Frontend | Candidate list, metrics, graph, partial responsive controls | F1 replaces primary reading/navigation flow; specialist starts after B5 |
| Data retrieval | `exact/tracks/builtin/bioml_hf.yaml` | Update hosted files, checksum names/keys, validation references and removed pool paths |
| Tests | `tests/exact_inspect_test.py`, `user_study_analysis_test.py`, ontology/run/rationale suites | Extend meaningful semantic, recovery, API and scale checks; avoid duplicate test frameworks |

The old shipped OMIM–ORDO bundle lacks identifiers/context that current producers can emit. Keep it as a legacy/sparsity fixture. It is not the schema to copy. The sibling-source pyOWLCore probe demonstrates possible APIs, not installed-release/native/full-file acceptance. Verify the [published native stack](../native-stack.md) (`0.2.1` in the current lock) and record wheel/version/hash.

## Owners and order

- **B0 contract owner:** shared schema, execution lock, annotation/visibility policies, capabilities, reference semantics, fixtures and stage plan. Freeze v1 semantics before B1/B2 producers diverge.
- **B1 Exact integration owner:** instrument existing producers/adapters. Do not alter scores, thresholds, candidate ordering or committed alignments merely to explain them better. Cross-result changes belong to experiment scope.
- **B2 context owner:** acquire/pin imports; index one ontology version independently of any matching run; produce hierarchy and original axiom context.
- **B3 service owner:** package and serve validated B1/B2 products; bounded read APIs, explicit local import, transactional study services, isolation and error behavior.
- **B4 explanation owner:** OpenRouter jobs and deterministic fallback text, policy-aware grounded outputs and regeneration.
- **B5 integration owner:** verifies installed packages, real NCIT–DOID path, kill/relocation/bug repair, performance, redaction and handoff bundle.
- **F1 specialist:** full frontend design/implementation only after B5 pass. No early specialist consultation is required.

B1 and B2 can overlap after B0. B3 can use fixtures while B1/B2 develop; B4 starts when fact identity, context/visibility and provenance are stable. Implement a small vertical case before bulk preparation. Do not make frontend admission depend on every optional provider, final model research result, full-ontology reasoner run or human-study outcomes.

## Required status ledger

For each B0–B5/F1 item record owner, inspected revision, implemented paths, tests/evidence artifacts, remaining inputs, resource measurements and status: `planned`, `implementing`, `fixture_ready`, `operational_ready`, `passed`, `blocked_input`, `failed`, `superseded`. Source review is not `passed`; stubs are not implemented; absent runtime/data cannot count as a negative experiment result. A blocked optional provider does not block an otherwise complete class-pair backend. The frontend gate is passed only by the full checklist in 07.

## Accepted product/study extension — 2026-09-20

All new features below remain **planned**, not implemented by these specifications. Existing viewer/study-export code does not constitute a complete survey application.

| New deliverable | Existing foundation | Missing implementation / owner |
|---|---|---|
| Local app bundle import | RunReader, preparation/export and viewer | Portable manifest/archive, validation/staging/library and import API (B0/B3); full UI (F1) |
| Fixed public demo | Existing deployment/viewer | Server-enforced profile, no import/study/admin surfaces, separate study assets (B3/F1) |
| Anonymous study | Restricted view and exported study cases | Durable definitions/invitations/sessions, setup/forms, allocation, mixed-case ranking, consultation and admin/export (B0/B3) |
| Resume and timing | Artifact checkpoints | Transactional participant persistence, idempotency/revisions, timing/event reconciliation and database recovery (B3/B5) |
| Study interface | Reusable candidate/explanation components | Entire first-party tutorial/questionnaire/ranking/feedback/resume workflow (F1) |
| Render study deployment | Existing demo blueprint | Database binding, migrations, secrets, backup/recovery, isolated service and launch runbook (B3/B5/F1) |

B0 defines 10–13 and study schema/locks; B3 implements all new service surfaces; B5 verifies APIs and database restart/recovery before frontend admission. F1 implements both products. Actual participant launch and subgroup-effect claims remain outside backend readiness. Authenticated researcher controls are required, but participant accounts and a generic survey builder are not.
