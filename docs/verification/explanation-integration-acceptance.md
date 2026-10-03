# Study integration joint acceptance — 2026-10-03 (S3)

**Result: R01, R02 and R04 are joint-verified. R03 is verified except for the manual
screen-reader and keyboard audit in J10, which was not performed; the follow-up is therefore
not yet fully accepted.** A new finding, R05, was found and fixed. This record follows
[17](../../specs/explanation-framework/17-integration-corrective-programme.md) S3. It was run by
the frontend agent on the combined system and verifies both sides. It does not certify the
study for launch. The machine-readable [acceptance receipt](explanation-integration-acceptance-receipt.json)
lists every test ID and result.

## System under test

| | |
|---|---|
| Commits | backend `96b76fe` (+ S1 docs `4b02ee8`); frontend `6895bc6` and test addition `10fea04` |
| Service | `tools.serve_explanation_v2_e2e`: real study app, compiled `out/` export, normal CSP, HTTPS on 127.0.0.1 (local self-signed certificate) |
| Database | PostgreSQL 16.2, dedicated synthetic clusters (one for the harness, one marked cluster for backend restart/restore tests) |
| Fixture | Built fresh with `--navigation-size 65`: 59 files; publication file SHA-256 `386477263ac48895df25d1036f9f8a7a029393413352857c46f3c338cb3676cc`; admitted frozen hash `68180971f0e92528a57f5b881ebd9276cd332204a956a3d9a5b37d9f89ef7c44`; revision `synthetic-v2-4f937dc78fd0444ea8ec765371c7bf19`; source export `exact-study-analysis/3` |
| Main app (J03) | `exact-inspect serve --profile local_app` on the same frozen contexts, wrapped by `e2e/prepare_main_app_package.py` under the publication's own visibility policy (package `sha256:e69e4604…`) |
| Contracts | The four generated files match the S1 receipt hashes; `tools.freeze_explanation_contracts --check` passes |
| Browser | Chrome 154 via Playwright 1.63 (Node 24.21); Next 15.5.26, React 19.1 |
| Python | 3.12.3, FastAPI 0.116.2, Pydantic 2.13.4, uvicorn 0.35.0, psycopg 3.3.6 |

All data is synthetic. No invitation URL, token or session record is stored in Git; the harness
config is a mode-0600 file outside the repository.

## Findings

| ID | Status | Basis |
|---|---|---|
| R01 | **joint_verified** | J01, J02 (browser subset plus backend), J03 below |
| R02 | **joint_verified** | J04, J05 below |
| R03 | **blocked on J10 manual audit** | J06, J07 and J09 joint-verified; J10's automated parts pass; the manual screen-reader/keyboard pass was not performed |
| R04 | **joint_verified** | J08 (backend oracles plus admin browser), J09 |
| R05 (new, P3, frontend) | **fixed and verified** | Cytoscape's injected `<style>` violated the study CSP whenever the graph opened; reproduced in the journey (violation in tutorial lesson 3), fixed in `6895bc6`; the journey fails on any CSP violation and now has none |

## Joint matrix

| ID | Result | Evidence |
|---|---|---|
| J01 | Pass | The descriptor is asserted on the real `cases/current` response in the journey. Capabilities are validated in the browser and, against the backend's recorded examples, in unit tests. Only scoped routes are read and there is no `resources/case-*` request. A missing descriptor or another scope's capabilities block the case (`study-v2-integration` R01/R02 tests). Backend: `explanation_integration_contract_test`. |
| J02 | Pass (browser subset) | Browser: a baseline case makes no workspace request, even with tutorial help open, and a previous explanation scope returns 403 after advancing. v1 revocation is covered in `study.spec`. The participant journey makes no request outside `/api/v1/study/`. Backend (SQLite and PostgreSQL): session binding, stale scope, cursor and stage denials. v2 invitation reissue/revocation was not exercised in the browser; it is backend-verified. |
| J03 | Pass | Journey on a scored case: search finds the non-focal "Navigation source 012" with 5 parents (multiple inheritance); 65 children are paged 50 + 15 without duplicates; the "Comments" category is paged 20 → 65 without duplicates; a citation opens its original record with the focal IRI; the evidence list (scoped `/evidence` 200) and graph open. Main-app parity: `entity-context`, children, parents, search and every `entity-facts` page are identical, except the declared metadata of the policy-filtered copy (`context_revision`, category inventory, `extraction: complete_for_policy`) and query-bound cursors. The main-app browse view shows the same 50 of 65 children and 5 parents. |
| J04 | Pass | Labelled injections, each blocking ranking, submission and `case_ready` until a retry recovers:<br>• target context 503 (also axe-clean, with Tab reaching the retry);<br>• comparison failure;<br>• description failure;<br>• another scope's capabilities;<br>• a 200 response describing another entity;<br>• a missing descriptor.<br>A slow fifth candidate keeps the case loading, and `case_ready` and timing start only after it. A recorded `not_exported` absence is usable. A retry refetches only the failed read and keeps a saved partial ranking and its order. Replacing the session aborts the old case and nothing from it becomes ready. |
| J05 | Pass | Initial and reloaded readiness; a second tab reports its own `case_ready` from its own page instance. Rapid candidate switching neither restarts the case nor refetches contexts. A lazy hierarchy failure stays local. Pause/resume reopens case timing only after usable. No case timing segment is sent before ready. Offline recovery and 409 are covered in the tutorial and in v1 (`study.spec`, `study-recovery.spec`); stage transitions are covered by the journey. The hidden-tab non-pause telemetry is unchanged and was not newly exercised. |
| J06 | Pass | Seven cases:<br>• lesson 1 → 2, Saved, reload with no action → lesson 2 (focus on the new heading, none after restore);<br>• assessment landing → reload;<br>• a partial draft and a focused question survive reload;<br>• incorrect → feedback → retry → pass, with both receipts after reload;<br>• rapid Next/Back with 1.2 s delayed saves;<br>• offline then reconnect;<br>• a stale second tab adopts the server position after a 409.<br>Visible and server positions are asserted each time. |
| J07 | Pass | `study-v2-backend.spec.ts`: a new invitation and every step in the UI (consent, setup, background, all six lessons, all five items with one incorrect retry), then both conditions, one Yes report with two methods and later No reports, the final form and completion. |
| J08 | Pass | Backend timing oracles on SQLite and PostgreSQL (`explanation_integration_timing_test`). Admin browser: JSON and CSV exports request and receive `exact-study-analysis/3` / `exact-study-csv/3`; every exported case timing has `coverage_status: not_established` and null coverage/unobserved seconds. A returned analysis-2 manifest (diagnostic) is reported and not saved. |
| J09 | Pass | Backend PostgreSQL: 64 passed, 0 skipped, including the real server restart and `pg_dump`/`pg_restore` (v1 and v2). SQLite: 159 passed, 2 skipped (only those PostgreSQL-only tests). Browser v1 journey, offline and researcher tests pass (`study.spec`). |
| J10 | **Partial** | Automated: actual `innerWidth` 320, 390, 768, 1280 and 2560 px at 100% and 200% text (root font 32 px), light and dark; setup, tutorial and case have no overflow, the ranking control is reachable and axe reports no violations. Focus assertions for navigation, invalid answers, dialogs and the retry. **Not performed: a manual screen-reader pass and a manual keyboard pass.** The in-app browser refused the harness's self-signed origin and no assistive-technology session was available. |
| J11 | Pass | `exploration.spec` (9), `typed-browse` (2), `accessibility.spec` (27), demo boundaries, local import, study CSP (journey), ranking states, ordered and branching forms, report draft restore and clearing, researcher export controls. |
| J12 | Pass | Typecheck and production build; 61 unit tests; spec validator (27 JSON files, 166 local links, no errors); contract freeze check; browser 69 passed and 2 skipped. The 2 skips are `legacy.spec`, because the historical run directory is not on this machine. Scoped diff: frontend files only; no backend file changed. |

## Commands

```bash
python -m tools.build_explanation_v2_fixture <dir> --navigation-size 65
python -m tools.serve_explanation_v2_e2e --fixture <dir> --database-url postgresql://… \
  --tls-cert … --tls-key … --config <private.json> --port 18984 --frontend-dir explanations_visualizer/out
python explanations_visualizer/e2e/prepare_main_app_package.py <dir> <main-app-dir>
exact-inspect serve --package <main-app-dir>/package.json --profile local_app --port 18985 …
(cd explanations_visualizer && npm run typecheck && npm run test:unit && npm run build)
EXACT_E2E_STUDY_V2_CONFIG=<private.json> EXACT_E2E_MAIN_APP_URL=http://127.0.0.1:18985 … npx playwright test
python -m pytest <the S1 SQLite suite> tests/explanation_frontend_form_order_test.py tests/explanation_frontend_serving_test.py
EXACT_STUDY_TEST_DATABASE_URL=… EXACT_STUDY_TEST_PGBIN=… EXACT_STUDY_TEST_PGDATA=<marked cluster> python -m pytest <the S1 PostgreSQL suite>
python specs/explanation-framework/protocol/validate_specs.py
```

The rest of the browser suite also needs the earlier exploration and v1 harness variables; see
the frontend README. Rebuild the export and restart the study service together, because the
service hashes inline scripts for its CSP at start-up.

## Remaining gates

Still required:
- **J10 manual audit:** a screen-reader and keyboard pass over the changed controls by a person using assistive technology. This is what keeps R03 open.
- **Release integration:** the real `backend-release` package, a built Docker image, Render deployment and restore, and a trusted HTTPS deployment (not a local self-signed certificate).
- **Study owner:** formative usability and the duration pilot, consent/information approval, real case adjudication, and the historical DOID/import-closure gates.

No push, deployment, recruitment or participant contact was performed.
