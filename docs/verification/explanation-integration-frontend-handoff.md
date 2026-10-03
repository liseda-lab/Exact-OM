# Study integration frontend handoff — 2026-10-03 (S2)

**S2 frontend implemented and verified against the S1 backend.** This implements
[spec 19](../../specs/explanation-framework/19-frontend-integration-corrections.md) under
[17](../../specs/explanation-framework/17-integration-corrective-programme.md), consuming the
[S1 backend handoff](explanation-integration-backend-handoff.md) without changing backend code.
The combined S3 result, including what remains unverified, is in the
[joint acceptance record](explanation-integration-acceptance.md).

| | Commit |
|---|---|
| Start (S1 documentation) | `4b02ee8d451ebf5fa7c37460ae61ba21077e162d` |
| Backend implementation consumed | `96b76fe74afab6d2fdb6437e2ff0b1c28d9b5a3f` |
| Frontend implementation | `6895bc64a04bf43a68deb51266cea4fa797c9610` |
| Test addition (scoped evidence views in the journey; no product change) | `10fea04611f628329be732067cf22e57370620d6` |
| This handoff | the following documentation commit (not self-referential) |

The [machine-readable receipt](explanation-integration-frontend-receipt.json) lists hashes,
tests and environments. No push, deployment, recruitment or participant contact was done.

## S2 preconditions, checked before implementation

- The backend commits `6aa4a9c`, `acdb252`, `8e76aa6`, `96b76fe` and handoff `4b02ee8` are in this checkout.
  The four generated contract files have the exact SHA-256 values in the backend receipt, and
  `python -m tools.freeze_explanation_contracts --check` passes.
- A fresh fixture built with `python -m tools.build_explanation_v2_fixture <dir> --navigation-size 65`
  has 59 files, like the S1 inventory. Its own hashes differ because the revision IDs are new (see the receipt).
- The real service (HTTPS, PostgreSQL 16.2, compiled export, normal CSP) advertises
  `integration_contract: "study-integration/1"`. A real `GET /api/v1/study/cases/current` for a v2 explanation
  case carries `workspace.scope_id`; baseline carries `workspace: null` and no explanation refs.
- No backend defect was found, so nothing was returned to the backend owner.

**Baseline reproduction.** The new browser regressions were run first against the starting
commit's frontend served by the real service. Four failed, one passed:

| Regression | At `4b02ee8` |
|---|---|
| Lesson 1 → 2, wait for Saved, reload without a lesson-2 action | Failed: the server position stayed on lesson 1 |
| Enter the assessment, wait, reload without an answer | Failed: the server position was `identity` (lesson) |
| v2 explanation case without its descriptor | Failed: no blocking error; the excerpt adapter path remained |
| Target context 503 | Failed: no blocking error; ranking enabled; `case_ready` sent |
| Scoped reads when the descriptor is present | Passed (the backend commit had already wired the API adapter) |

## What changed and why

| Finding | Change |
|---|---|
| **R01 / F11** adapter selection | v2 sessions run only when `state.integration_contract` is `study-integration/1`; otherwise a page says the service needs an update and nothing is collected or mutated (pause is hidden). A replayed pre-extension receipt keeps the confirmed extension and triggers a fresh state read instead of flashing that page. A v2 explanation case requires `workspace.scope_id`. Its `/capabilities` are validated against the scope, the study revision, the case's ontologies, its six focal entities and the components the frozen condition shows. The case then reads only `/api/v1/study/workspace/{scope}/…` through `createApiSource`. A missing descriptor, failed read or incompatible capabilities block the case with a retry; there is no fallback to `createResourceSource`. The v1 legacy adapter and the tutorial's synthetic adapter remain, selected explicitly. A response for another presentation, revision or protocol, or with malformed candidates, is rejected and not rendered. Cache keys include the session, revision, presentation, scope and policy hash. |
| **R02 / F12** readiness | A per-presentation state: `loading` → `rendering` → `usable`, or `blocked` (with `submitting` from the ranking panel). The bounded minimum for every case is its identity, five distinct candidates in display positions 1–5, and listed download metadata with size and hash. Explanation cases also need capabilities; the source and all five candidates' contexts, each checked against its entity; the admitted descriptions (6); and the admitted comparisons (5). These reads run four at a time and are shared with the rendered workspace (`sharedReads.ts`), so the cards render exactly what was validated. Recorded absence (`not_exported`, `not_requested`, `unverified`) is usable; a failed, mismatched or unknown response is not. Readiness becomes `usable` only after the cards and comparison render without error (an error boundary plus a render probe); a stalled render blocks after 10 s. While not usable, ranking and submission are disabled, the saved answer stays visible, and the failed items are listed with a retry. A retry re-reads only what failed and keeps the answer and its order. A lazy panel failure stays local. A 401/403/409 from any scoped read (other than a stale cursor) blocks submission. `case_ready` and case timing need the page's own usable presentation: the case timer also stays closed after pause/resume until the content is usable again. Submission rechecks readiness and the presentation/revision binding. |
| **R03 / F13** tutorial position | Navigation builds the destination `position` first and saves exactly that, position only, with a coalescing key that a newer destination may replace; evidence, drafts and attempts never carry a position. Reload, a new device and conflict reconciliation restore the server's acknowledged position; an acknowledgement never pulls a newer local navigation back. The deprecated `current_lesson_id` is no longer sent. Deliberate navigation focuses and scrolls to the new heading (a question heading for question navigation); restore and autosave never move focus. The assessment gains a "Questions" navigation saving `question_id`. An incomplete answer focuses its announced error. After a reload with unsent work, a note says the page shows the last saved position until the server confirms. |
| **R04 / F14** exports | The admin page finds the revision in the authenticated listing (paging as needed). For exact-study/2.0 it requests `analysis_schema=exact-study-analysis/3` in both formats, and for v1 it requests nothing extra (frozen analysis/1). The returned manifest is checked before saving: JSON `manifest.schema`, CSV `manifest.json` `schema` and `archive_schema`. A mismatch or a service without analysis 3 is reported and not saved; it never retries with 2. The page shows the source protocol, the frozen source export and the derived schema separately, and states that analysis 3 leaves unique elapsed coverage unknown. No timing is computed in the browser. |
| **F15** tests | `e2e/study-v2-backend.spec.ts` is the unseeded journey, and `e2e/study-v2-integration.spec.ts` holds 25 seeded fault/recovery tests with labelled injections. `e2e/prepare_main_app_package.py` wraps the same frozen contexts for the main-app comparison. Unit tests: `caseReadiness.test.ts` (on the backend's recorded HTTP examples) and `tutorialPosition.test.ts`. |

**Also changed:**
- The shared entity card now pages fact categories by continuing the context's own cursor through `/entity-facts`, in both the study and the main app. Spec 19 lists bounded fact pagination as a permitted operation, and before this change the card showed only the first 20.
- The development-only v2 preview (`/preview/participate/`, its in-browser fixture service and its tests) is removed. A second implementation of the contract was a source of divergence, and the real service now covers those flows.

**New issue R05 (P3, frontend, fixed).** Under the study CSP (`style-src 'self'`), Cytoscape's injected `<style>`
element was blocked whenever the evidence graph opened; the browser reported the violation in tutorial lesson 3.
The graph still rendered, but its container rule was lost. The rule now lives in `app.css`, and Cytoscape's
stylesheet id is reserved before initialization, so nothing inline is injected. The acceptance journey now
fails on any CSP violation and names its source.

## Tested routes and assertions

**Participant routes exercised in the browser:**
- `state`, `session`, `consent`, `setup`, `questionnaires/{background,final}`;
- `tutorial/{progress,assessment,complete}`;
- `cases/current`, `cases/{id}/{draft,submit,consultation/draft,consultation}`;
- `events`, `timing`, `pause`, `resume`, `complete`, `resources/{asset}`;
- under `workspace/{scope}/`: `capabilities`, `entity-context`, `entity-facts`, `entities`, `hierarchy`, `labels`, `axioms/{id}`, `explanations`, `explanations/{id}` and `evidence`.

**Admin routes:** `GET studies` and `POST studies/{rev}/exports` (JSON and CSV).

**Network assertions:**
- The descriptor is present and every focal context is fetched from its scope, each once. Rapid candidate switching refetches nothing.
- No v2 request goes to `study/resources/case-*`.
- Baseline cases make no workspace request, even with tutorial help open. A previous scope returns 403 after advancing.
- The participant journey makes no request outside `/api/v1/study/`, and no participant response contains grading rules or case keys.
- No `case_ready` or case timing segment is sent before usable. One `case_ready` per page and presentation; a second tab sends its own.
- Every export request carries `analysis_schema=exact-study-analysis%2F3`.

**Browser and viewports:** Chrome 154, with the requested widths confirmed by `innerWidth`: 320, 390, 768,
1280 and 2560 px at 100% and 200% text (root font 32 px), light and dark, plus 1280×720 and 1280×900.

**Keyboard (automated assertions):**
- focus moves to the new lesson, assessment or question heading after navigation, and not on restore;
- an incomplete answer focuses its alert;
- the blocked case's retry follows the toolbar in Tab order;
- tutorial help returns focus to its trigger.

A manual keyboard and screen-reader pass was **not** performed (see the acceptance record).

## Remaining issues and next owner

S3 joint verification is in the acceptance record. Still open:
- a manual screen-reader and keyboard pass over the changed controls (J10);
- the real `backend-release` package;
- a built Docker image, Render deployment and restore;
- formative usability and the duration pilot;
- study-owner consent/information approval, real case adjudication, and the historical DOID/import-closure gates.

The next owner for J10's manual audit is a human accessibility reviewer. Release gates stay with the study owner.
