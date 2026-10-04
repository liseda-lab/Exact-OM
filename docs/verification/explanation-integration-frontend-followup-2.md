# Study integration frontend follow-up 2: R10–R11 — 2026-10-04

**Implementation complete and verified on the synthetic fixture; not a release-readiness claim.**
A second review of `ad9461c` reproduced two further frontend defects that reopened R02 and R01.
Both are fixed, and their regressions pass against the real study service and the main app.
R01 and R02 are joint-verified again. R03 stays blocked on the manual J10 audit. The intermittent
reload stall recorded in the [first follow-up](explanation-integration-frontend-followup.md)
remains a separate, unresolved investigation.

| Commit | Content |
|---|---|
| `fac6056` | Specifications first: R10–R11 recorded ([review](../../specs/explanation-framework/evidence/frontend-integration-review-20261004.md)), R01/R02 reopened, F20–F21 specified |
| `d8b19a0` | R10: access checks bound to the attempt that issued them (+ unit tests) |
| `e754e77` | R11: navigation carries the recorded typed entity |
| `eb4f955` | Typed-property fixture option and browser regressions |
| this handoff | the following documentation commit |

No backend code changed and no contract gap was found. The fixture builder's default-off
`--paged-facts` option now also adds an object property and a data property, each with 55 named
superproperties (test data only).

## Outcomes and evidence

| Finding | Correction | Evidence |
|---|---|---|
| **R10** a superseded access check blocks a recovered case | `createAccessGuard` (`src/study/accessGuard.ts`) issues each scope re-check bound to the attempt current at issue time. The shared read source survives retries, so the binding is taken per check, not when the source is created. Checks are tied to their workspace and session through `live()`, and Retry and teardown abort them. A result for a superseded attempt, presentation, scope or session is ignored; a check issued and refused in the current attempt still blocks. | Unit (5): a held check across retry cannot block attempt 2; even an uncancelled late check is ignored; a current loss still blocks while an outage does not; obsolete workspace or session is ignored; cancel aborts requests. Browser: (a) a held check released after a successful Retry leaves the case usable with no alert, the draft and its order intact on screen and on the server, and no context refetched; (b) a check from a submitted presentation cannot affect the next case; (c) a check from a replaced session cannot affect the new one; (d) the existing test that a refused scope blocks still passes. |
| **R11** continued parents lose their entity type | Navigation callbacks take a typed `EntityRef`: card parents (first page from `hierarchy_projection.parent`, later pages from hierarchy edges), original-axiom and evidence-list terms (kind from the AST node type), the main app's browse view, and the pair workspace's card links and full-context dialog. Shells no longer guess from the first page. A term without a recorded type (a bare IRI value) is shown but not navigable. | Browser, both products and both property kinds: after "Load more parents", every parent carries its kind. Choosing a later-page parent navigates with the same ontology, the chosen IRI and kind `object_property` / `data_property`. The focus card names the kind, its full context loads (scoped `entity-context` with that kind, 200 in the study), no read uses `kind=class`, and the study ranking and draft are kept. Class navigation still asserts `sk=class`. |

**Baseline reproduction:** `ad9461c`, built from `git archive` and served alongside, ran the new browser regressions. **5 of 7 failed:**
- R10: the released old check re-blocked the recovered case (alert count 1, expected 0).
- R11, both products: the run failed at the new `data-kind` hook. A separate diagnostic then clicked a later-page superproperty on the baseline: it navigated with `sk=class` for both the object and the data property, while the fixed build kept `object_property` and `data_property`.
- Passed on the baseline: the obsolete-presentation and obsolete-session checks, because unmounting already isolated them. They remain as defensive regressions.

## Verification at `eb4f955`

**Fixture:** `--navigation-size 65 --paged-facts 26`, 59 files; publication SHA-256 `ea747606…22ac`. The component variants `-cmp`, `-ctx`, `-desc` and `-nav` were republished under their own form versions.

**Service:** the real study service (HTTPS, PostgreSQL 16.2, compiled export, normal CSP), plus main-app servers on the same frozen contexts. One used the study policy, for parity; one added `alternate_definitions`, for paging.

**Results:**
- type check and production build pass;
- unit tests: **73 passed** (5 new);
- browser: **87 passed, 2 skipped**. The skips are `legacy.spec`, because the historical run directory is unavailable. The run includes the fresh, unseeded participant journey, every R06–R11 regression, the earlier v2 tests, and the exploration, library, demo, accessibility and v1 study suites;
- backend suites that use the fixture builder: 40 passed, 1 PostgreSQL-only skip; black and isort clean; spec validator: no errors.

The [receipt](explanation-integration-frontend-followup-2-receipt.json) lists every test result and hash.

**Environment note:** iCloud had evicted about 7,000 files under `node_modules` and some source files, which made type checks and builds take minutes. Builds and browser runs therefore used the locked dependencies installed outside iCloud (`npm ci --prefer-offline`) and a scratch copy of the frontend synced from the repository before every build. The repository stays the only source.

## Remaining limitations

- **J10 manual audit:** a screen-reader and keyboard pass is still required. It now also covers the continuation controls, the full-context dialog, the blocked state and the property-typed navigation.
- **Intermittent reload stall:** still open and assigned to the backend. The study service logs `h11 LocalProtocolError` on client disconnect.
- **Bare IRI terms:** a bare IRI term in an original axiom, which has no recorded type, is no longer navigable rather than being opened as a class. No current fixture relies on navigating such terms.
- **Release gates (unchanged):**
  - the real `backend-release` package;
  - Docker, Render deployment and restore;
  - trusted HTTPS;
  - usability and duration pilot;
  - study-owner consent/information approval, case adjudication and the historical DOID/import-closure gates.

No push, deployment, recruitment or participant contact was performed.
