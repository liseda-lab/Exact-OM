# Integration corrective programme and agent protocol

**2026-10-03. Status: S1, S2 and S3 run; R01, R02 and R04 joint-verified; R03 blocked only on
the manual J10 audit. Reviewed baseline: `2d9715b`.**
The [backend handoff](../../docs/verification/explanation-integration-backend-handoff.md)
records implementation `96b76fe`; the [frontend handoff](../../docs/verification/explanation-integration-frontend-handoff.md)
records `6895bc6`; the [joint acceptance record](../../docs/verification/explanation-integration-acceptance.md)
gives the J01–J12 results, a new fixed finding (R05) and the remaining human and release gates.
The follow-up is not fully accepted until a person completes the J10 screen-reader/keyboard audit.
Start here for the next correction cycle. The [review record](evidence/integration-review-20261003.md)
separates direct observations, a diagnostic reproduction and checks that remain outstanding.
The implementation assignments are [18 — Backend](18-backend-integration-corrections.md)
and [19 — Frontend](19-frontend-integration-corrections.md). The
[machine-readable acceptance contract](protocol/integration-followup.json) is a design
artifact, not runtime configuration or proof of a deployed service.

## Authority and preserved decisions

This amendment extends 14–16 and overrides their **implementation order for this follow-up**:
**backend contracts and verification first, frontend integration second, joint acceptance last**.
The frontend design/inventory phase from 14 has already happened; do not repeat it as a
dependency or redesign the product. Earlier dated handoffs remain historical evidence,
not proof that the defects below are closed. This document governs acceptance across agents;
18 owns the wire contracts, and 19 owns their UI consumption. If an assignment appears to
contradict a wire rule, resolve it in the spec/contract record before implementation, rather
than implementing different assumptions on each side.

Preserve the following requirements from [14](14-corrective-programme.md):

- One shared ontology/explanation workspace for the main app, tutorial and explanation
  condition. The study adds its workflow and ranking controls. The baseline deliberately
  does not expose integrated explanations. Shared component imports alone do not prove parity.
- Protégé is recommended, never mandatory. Participants can use any permitted inspection
  method, combine methods, change between cases, or use none. Do not introduce a study-wide
  tool commitment or require use of a named application.
- One consultation report **per source plus five-candidate case**, after the ranking. Record
  the actual combination used, including none; do not infer per-candidate use, carry previous
  methods forward automatically, or equate download/visibility events with reported use.
- Interactive synthetic training and five server-graded comprehension items, durable
  drafts/attempts, specific feedback and unlimited pedagogical retries. Training resources
  remain disjoint from scored cases; no scoring allocation before server-authorized completion.
- Server-owned condition, assignment, policy and session isolation; typed facts, original
  axioms, exact citations and honest missingness; no generation on read or matching changes.
- Preserve frozen publications, hashes, existing answers, idempotency receipts and saved
  exports. No reset or automatic migration of active human sessions to make tests pass.

This is the complete disposition of the **identified review findings and their regression
risks**, not a claim that all possible defects are known. Implementation is authorized;
recruitment, participant contact, production deployment and matching campaigns are not
authorized by this assignment. Existing real-data, recovery and launch gates remain binding.

## Reopened findings and ownership

| ID | Priority and old coverage | Defect and required outcome | Primary owner |
|---|---|---|---|
| R01 | P1; C01/C02 | Current-case responses do not advertise their workspace, so v2 silently uses prepared excerpts. Return the authorized scope and require the real scoped adapter for v2 explanation cases. | Backend, then frontend |
| R02 | P1; C16/C23 | The API adapter's readiness path waits for the source only. A target failure can coexist with an enabled submission. Readiness must cover a bounded, explicit set of required case content and actual shared-workspace state. | Frontend; backend contract/tests |
| R03 | P2; C14 | Navigation saves the old lesson; assessment position is ambiguous; omitted positions can clear persisted state. Persist explicit lesson/assessment position and restore it without an extra interaction. Move keyboard focus on deliberate navigation. | Backend state contract, then frontend |
| R04 | P2; C16/C27 | Summed observations from independent page clocks are subtracted from elapsed time as if they were unique coverage. Preserve raw page observations and export unknown coverage honestly. | Backend export, frontend consumer/regression |

The existing browser test's “hierarchy region is visible” assertion misses R01; an action
after changing lesson hides R03. Test strengthening is part of each correction, not a fifth
product defect. All four findings must be implemented and independently verified before this
follow-up is accepted. A new issue must receive its own ID, reproduction, severity, owner and
disposition; a required behavior cannot be removed to obtain a green result.

## Default sequential execution

### S0 — Record the starting point

Each agent reads 17–19, the review record, current source and Git status. Record its starting
commit and any intervening changes since `2d9715b`. Reproduce the assigned defect or point to
an executable regression that fails on that baseline. Preserve unrelated work, including
untracked archives. Do not reset, clean or bulk-stage the repository. An earlier summary or
agent's “done” message is not a substitute for inspecting the checkout.

### S1 — Backend implements and hands off first

The backend agent owns 18, generated runtime schemas/OpenAPI, fixture tooling and backend
tests. Implement R01/R03/R04 and the backend portions of R02. Check any new fields end to end
through response projection, strict serialization and stored state, not just model definitions.
Regenerate runtime snapshots from implemented models; do not hand-edit them to suggest support.

Produce `docs/verification/explanation-integration-backend-handoff.md` and
`docs/verification/explanation-integration-backend-receipt.json`. Include:

1. Exact backend commit, start commit, protocol-extension identifier and generated-contract
   hashes. Use a subsequent documentation commit to record the implementation SHA; do not
   create a self-referential commit hash.
2. A reproducible synthetic fixture builder and launch instructions, dependency versions,
   package/publication hashes and explicit synthetic provenance. Store no tokens, invitation
   URLs, participant data or machine-specific secrets in Git.
3. Actual serialized success/error examples for the current case, capabilities, tutorial
   position mutations/receipts and analysis export; name omission/null/error behavior.
4. Test commands, results, database used, skips and unresolved checks. Include authorization,
   restart, conflict and compatibility checks, plus a contract-level integration test.
5. An R01–R04 acceptance table: `implemented`, `backend_verified`, `frontend_pending`,
   `joint_verified` or `blocked`, with evidence for every claimed transition. Do not mark
   frontend or joint acceptance passed from backend-only tests.

Backend handoff is ready when J01/J02/J06/J08/J09 below pass at the API/database boundary
and frontend-facing fixtures/contracts are reproducible. PostgreSQL failures block that
handoff; absent real-release data or Docker/Render can remain clearly separate release gates.
Fix backend test failures before handing implementation to the frontend agent. Missing
external inputs do not prevent independent work, but cannot be described as verified.

### S2 — Frontend consumes that exact backend handoff

The frontend agent first confirms the backend commit is present, the extension is advertised,
the fixture builds, and real current-case responses carry the expected descriptor. If a
backend contract is broken, produce a minimal failing request/test and return it to the
backend owner; continue independent frontend work without inserting a production shim.

Implement 19 against the published contract. Produce
`docs/verification/explanation-integration-frontend-handoff.md` and
`docs/verification/explanation-integration-frontend-receipt.json`, identifying both implementation
SHAs and contract hashes, the adapter used, tested routes, browser/viewport dimensions,
network assertions, screenshots where useful, keyboard results and remaining gates. UI
previews, mocks and screenshots without request assertions cannot establish integrated closure.

### S3 — Verify the combined system and close findings

Run the joint matrix on the combined commits and the production frontend build served by the
actual study service. At least one complete browser journey starts with a new synthetic
invitation and performs consent, setup, background, all tutorial interactions, all five
assessment items, both conditions, consultation, final form and completion. Do not seed that
journey with API-completed training. Separate seeded tests are useful for fault scenarios.

Record `docs/verification/explanation-integration-acceptance.md` and a machine-readable
receipt listing case/test IDs, commits, fixture hashes, results and unverified gates. Either
agent can run this phase when assigned, but the receipt must verify both sides. The second
agent normally performs it in a sequential run. A failure is returned to its owner and the
affected cross-boundary checks are repeated after the fix. Neither agent can waive a P1/P2
requirement or declare another agent's work verified solely by reading their handoff.

## Optional parallel execution

Use separate branches/checkouts. Do not share an index, generated output directory, mutable
fixture/database or service port. Record a branch/file ownership table before parallel edits.

| Backend owns | Frontend owns | Coordinated files |
|---|---|---|
| `exact_inspect/study/`, backend tests, fixture generators, generated runtime contracts, backend receipt | `explanations_visualizer/`, frontend tests, browser tests, frontend receipt | These specs, protocol design JSON, status ledger, shared guides and combined acceptance |

Only one named owner edits each coordinated file in a given phase. Backend produces the
wire contract; frontend may develop independent interaction/readiness tests against labeled
fixtures meanwhile. Freeze the contract examples and hashes before integrating either side.
Merge backend first, update the frontend branch onto that commit, regenerate any client
types, then run the same S2/S3 acceptance. A changed contract reopens affected tests and
requires an updated handoff; do not resolve a mismatch by accepting arbitrary unknown data.
No parallel mode may bypass S1's handoff or S3's combined verification.

## Joint acceptance matrix

| ID | Required proof, using the actual service unless explicitly a unit test | Owners |
|---|---|---|
| J01 | V2 explanation current-case descriptor survives HTTP serialization; capabilities match current revision/scope/ontologies; browser uses scoped routes and never the prepared-excerpt scored adapter. | Both |
| J02 | Baseline, stale invitation generation/session binding, other presentation/scope, closed/revoked session, wrong cursor/query and unauthenticated reads are denied under existing rules; synthetic help grants no scored access. | Backend + browser subset |
| J03 | Search an entity outside the focal excerpt; follow parents/children; exhaust multi-page context and hierarchy without duplicates; open an exact citation/original axiom. Compare against the same prepared data in the main app, allowing only declared condition/policy restrictions. | Both |
| J04 | Delay/fail every required content class, especially target context and admitted comparison. No premature ready event or submission; explicit genuine absence remains usable. Retry recovers without losing draft/rank. Delayed previous-case/candidate/session responses cannot mark the new case ready. | Both |
| J05 | Check initial/reloaded/multiple-page readiness, rapid candidate switches, lazy panel failure, offline recovery, 409 conflict, pause/resume and stage transition. No duplicate logical case start, pre-ready duration, baseline fallback or implicit hidden-tab pause. | Both |
| J06 | Navigate lesson 1 → 2, wait for acknowledgement without interacting, reload/restart and remain on 2. Enter assessment without answering and reload there; save draft, reload, retry an incorrect answer and retain receipts. Omitted position never resets it. Keyboard focus follows deliberate navigation. | Both |
| J07 | Complete fresh browser training and the full scored journey in both conditions; training remains disjoint, server grading/assignment authoritative, methods optional/combinable/changeable and reports per case. | Frontend + backend |
| J08 | Single-page, multiple-page, empty, unavailable/late and retried timing exports match the explicit oracles in 18. JSON and CSV agree; no raw page sum becomes unique elapsed coverage or active duration. | Backend + admin browser |
| J09 | V1 canonical publication/resume/export regressions; historical v2 resume normalization, unchanged saved exports/hashes, explicit derived export version, new frozen export version and database restart/restore. | Backend |
| J10 | Keyboard and manual screen-reader spot checks for changed controls, contrast/focus, 320px and desktop layouts, 200% text, dark theme and persistent ranking. Record actual viewport values; test override success, not just requested width. | Frontend |
| J11 | Main-app import/browse/compare/trace/citations, fixed demo boundaries, study CSP, ranking states, ordered/branching forms, consultation draft clearing and researcher export controls still work. | Both |
| J12 | Build/typecheck, relevant unit/API/browser suites, spec validator, clean scoped diff; combined receipt explicitly preserves real-package, image/deployment, usability/pilot and study-owner launch gates. | Both |

Fault-injection tests may intercept or delay a real endpoint, but must identify the injected
failure. A replacement success response that invents the missing workspace is diagnostic
evidence only. Include at least one no-interception happy path. No unhandled browser error,
unexplained failed request, unexpected CSP violation or unexplained test skip can be ignored.

## Dispatch instructions

For the backend agent: “Implement specs 17 and 18 from the current checkout. Preserve 14–16's
study decisions. Deliver S1 with reproducible contracts, tests and the backend handoff; mark
frontend/joint verification pending. Do not implement substitute frontend behavior.”

For the frontend agent, after S1: “Read specs 17 and 19 and the backend handoff. Verify its
commit and actual HTTP contract, integrate all four findings, then perform S2/S3 on the
combined system. Return any backend defect with a reproduction; do not hide it with a fallback.”

Each agent's final handoff must list what changed, why, exact verification and failures,
remaining issues and the next owner. Commit only its assigned work when authorized by the
implementation request. Do not push, deploy or contact participants by inference.
