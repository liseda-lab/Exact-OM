# Backend corrective implementation — 2026-10-03

The backend corrections in specifications 14 and 16 are implemented and integrated
with the corrected frontend at base commit `8f27e9f`.
Acceptance uses native synthetic ontology preparation, SQLite, PostgreSQL and a real
HTTPS browser journey. Real-package and deployment acceptance remain outstanding.
The machine-readable [verification receipt](explanation-backend-corrections-20261003.json)
records exact checks, fixture versions and limitations.

## Coverage against the corrective assignment

| Corrections | Implemented behavior and verification boundary |
|---|---|
| C01–C02 | `study/workspace.py` serves the shared typed read shapes from complete, physically filtered frozen context indexes. Search, facts, hierarchy, labels, axioms, profiles, comparisons and candidate evidence are paged and policy scoped. Every read rechecks session generation, publication, stage, condition and current presentation; cursors bind those identities and the query. Baseline and other case resources are denied. Synthetic tutorial/help scopes remain separate. |
| C03–C04, C18 | Citation admission preserves exact text, category, ordered subjects, facts and generation manifest. Hidden labels and paged facts resolve independently of rendered cards. Conflicting claim IDs are rejected within and across composed resources. Original typed values remain unchanged; structural hierarchy projections are additive. Unsupported rendering and canonical-byte/base64 original payloads retain honest availability and fallback behavior. |
| C05 | V2 forms carry explicit option and matrix-row order. Publication validates exact permutations; canonical sorted object keys cannot reorder the display or export dictionary. |
| C10–C11, C14 | Frozen six-lesson synthetic tutorial, typed action evidence, durable drafts, five server-graded core items, first/subsequent attempts and unlimited pedagogical retries. Completion verifies mandatory requirements and eventual passes before atomic allocation. Client-reported actions do not prove attention. A short request throttle protects writes without imposing a lifetime retry cutoff. |
| C12 | Tool-neutral setup has two acknowledgements, explicit resource-access state and draft/submit semantics. No installation or named-tool requirement. Submission verifies admitted downloads; drafts never allocate. |
| C13, C15 | Consultation remains one report per source-and-five-candidate case. Seven non-exclusive method codes, optional scope, durable draft and separate final commit. No automatic reuse or inferred use. Final No rejects stale methods/names/scope; the frontend clears them. Default exports redact optional names. |
| C16, C23 | Scoped case/tutorial/help observations match actual actions and admitted components. V2 late timing receives an idempotent unavailable acknowledgement with a reason; unavailable intervals never add observed duration. Tutorial/consultation time is separate from ranking time. Existing CSRF, session binding, conflict, revocation and retry behavior remains. |
| C17 | Portable export explicitly enables SQLite URI handling for its output connection before read-only immutable attachment. Tests cover URI-default-off/on builds, unusual paths, relocation, source write denial, unchanged source hashes, concurrency and failed-export cleanup. |
| C19 | Downloads have validated titles, source/target roles, ontology versions, format, size/hash and policy/license notes. Both conditions receive the same permitted case downloads. Current state does not expose future case assignments. |
| C20 | Candidate paging retains pair-ID order with explicit order/scope metadata and continuation. Original ranks are preserved. A 607-candidate tied-rank regression checks completeness without claiming that a capped client subset is globally rank ordered. |
| C21 | Authenticated bounded revision listing, current protocol/setup/tutorial/form/software/export metadata, CLI `list`, and researcher selector. No invitations, keys, consent text or credentials appear in summaries. Existing progress/export/reissue/revoke/close controls remain. |
| C22 | Manifest-only paged library metadata and local-only owned-copy deletion. Import receipts establish ownership; unmarked old copies and CLI-mounted packages cannot be deleted. Active/import/in-flight conflicts, process locks and descriptor-based deletion protect originals and prevent symlink escape. Hosted mutation routes are unavailable. Exploration review decisions remain browser-local. |
| C26 | Separate `exact-study/2.0` definitions and mutations, with regenerated runtime OpenAPI/publication/resource/tutorial schemas. Existing v1 bytes, hashes, histories and resume behavior are retained. New live v1 publication is rejected; synthetic legacy fixtures and idempotent historical republication remain supported. |
| C27 | Existing security/resource/serving regressions are retained; dated evidence and historical input gates remain separate. |
| C06–C09, C24 | Frontend-owned corrections are retained. This work connects their workspace, tutorial, telemetry, library and revision-selector contracts. The browser journey checks actual cases and selected accessibility states; it does not replace manual keyboard, screen-reader, reflow, touch, theme or usability review. |
| C25 | PostgreSQL concurrency, crash, restart, dump/restore and real HTTPS synthetic browser checks pass. Real-package integration, built-image and deployed redeploy/restore gates are not verified. |

Implementation centers on `exact_inspect/study/{v2_models,tutorial,workspace,telemetry,administration}.py`
and the existing study store/API. Preparation remains separate from serving; participant
requests do not parse OWL, dispatch models/providers or expose unrestricted paths.
The minimal study Dockerfile includes the added read dependencies. Its copied source layout
passes a cold-import check with matcher, model and native-parser imports blocked.

## Contract and operational notes

The [v2 backend contract](explanation-study-v2-backend-contract.md) provides compatibility,
migration, error/missingness and wire examples. The [service guide](../guides/explanation-study-service.md)
documents ordinary operation. No SQL migration or rewrite of active human sessions is
needed: v2 state lives in existing transactional JSON/history tables. Version changes
require a new frozen publication; no historical checkbox completion becomes an assessment pass.

The study read contract remains `exact-explain/1.0`. Runtime snapshots are generated with:

```sh
python -m tools.freeze_explanation_contracts
python -m tools.freeze_explanation_contracts --check
python specs/explanation-framework/protocol/validate_specs.py
```

Run the following from the repository root with study and native preparation dependencies
installed. Use a new empty output directory. The generator writes a synthetic package and
`publication.json`; it makes no provider requests and contacts no participants.

```sh
python -m tools.build_explanation_v2_fixture /tmp/exact-study-v2-fixture
```

For the real HTTPS integration harness, build `explanations_visualizer/out` first, create a
local TLS certificate/key, and use a dedicated PostgreSQL test database. The config is a
private mode-0600 file containing synthetic harness credentials; keep it outside Git.

```sh
python -m tools.serve_explanation_v2_e2e \
  --fixture /tmp/exact-study-v2-fixture \
  --database-url postgresql://127.0.0.1:18973/exact_corrections \
  --tls-cert /tmp/exact-study-test.crt --tls-key /tmp/exact-study-test.key \
  --config /tmp/exact-study-private.json --port 18974
```

In another terminal, run from `explanations_visualizer`:

```sh
EXACT_E2E_STUDY_V2_CONFIG=/tmp/exact-study-private.json \
  npx playwright test e2e/study-v2-backend.spec.ts
```

The harness uses actual HTTPS and PostgreSQL without rewriting Origin. The fixture is
`exact-study/2.0`, `tutorial/2.0-native-2`, `assessment/2`, `exact-study-forms/2`, and
`exact-explain-ui-1.2`; the receipt binds its policy and publication bytes. It is not a
production study template with approved consent or adjudicated biomedical cases.

## Verification

- Final SQLite regression: **260 passed, 2 skipped** in 160.02 seconds. The two
  PostgreSQL-only recovery checks are skipped in this run and executed below.
- PostgreSQL 16.2: **40 passed**, covering legacy/v2 state, concurrency, process exits
  before/after commit, actual server restart and separate-database dump/restore with
  matching tutorial state, consultation drafts and immutable exports.
- Chrome against real HTTPS/PostgreSQL: **1 passed**, through explicit setup, six lessons,
  a wrong assessment answer and retry, five passes, both conditions, all four synthetic
  cases, tutorial/report reload and final completion. No failed POST/PUT responses or
  page errors; automated axe checks at selected 1280×720 states found no violations.
- Frontend: **49 unit tests passed**, TypeScript passed, production static build passed.
- Minimal study Docker source-layout cold import: **1 passed**. This is also included
  in the SQLite suite and is not a built-image result.
- Runtime snapshot check, specification validator and changed-file formatting checks pass.

The two torch-dependent modules `explanation_decisions_test.py` and
`explanation_replay_test.py` were not run in the lightweight environment. Other selected
explanation regressions include native preparation, original facts, provenance, read
models, policy admission, serving, library operations, resources and portable exports.
The temporary loopback HTTPS server and disposable PostgreSQL cluster were stopped after
verification. Credentials, test links and database dumps are excluded from the repository.

## Remaining release gates

The current checkout has no `data/explanation-framework/backend-release/package/package.json`;
synthetic checks do not establish current real-package acceptance. Docker has no running
daemon, so image build/cold start and deployment/redeploy rehearsal were not run. Historical
G1/G5 remain as recorded: the original pinned DOID strict parse gate was blocked; a separately
pinned derivative and unresolved import are not evidence that the original gate passed.

Real case adjudication, owner information/consent approval, manual accessibility/usability,
duration pilot and Render readiness remain owner/release gates. No live publication,
participant contact, recruitment or public deployment was performed. The user authorized
committing this implementation after verification. No push was performed; unrelated
working-tree artifacts were preserved.
