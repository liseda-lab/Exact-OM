# Study integration frontend follow-up 3: R12–R13 — 2026-10-04

**Implementation complete and verified on the synthetic fixture; not a release-readiness claim.**
A review of `e36cfa5` found two remaining frontend issues ([review](../../specs/explanation-framework/evidence/frontend-integration-review-20261004-3.md)):
- **R12** (P2): the R11 correction disabled card parents built by the frozen-resource adapter,
  which serves the tutorial and v1 study publications. This reopened R03's frontend status.
- **R13** (P3): a recovery test read recorded traffic before the `case_ready` acknowledgement
  was recorded, so it failed intermittently. This weakened R02's evidence.

Both are fixed, and their regressions pass. The R10/R11 corrections and their regressions are
unchanged and still pass. R01, R02 and R04 stay joint-verified. R03's frontend status is
verified again, but R03 stays blocked on the manual J10 audit. The intermittent reload stall
remains a separate, unresolved investigation.

| Commit | Content |
|---|---|
| `7150016` | Specifications first: R12–R13 recorded, R03 reopened, F21 extended to the frozen-resource adapter |
| `a0479ef` | R12: frozen-resource parents typed from the recorded edge or the AST (+ unit tests) |
| `1808ed0` | Browser regressions for R12 in both paths; R13 recovery test waits for the acknowledgement |
| this handoff | the following documentation commit |

No backend code, fixture tooling or fixture data changed, and no contract gap was found.

## Outcomes and evidence

| Finding | Correction | Evidence |
|---|---|---|
| **R12** tutorial and v1 card parents not navigable | `entityContextFrom` (`src/lib/workspace/resourceIndex.ts`) attaches a typed `hierarchy_projection` to each parent fact. The type comes from the recorded literal-asserted hierarchy edge for that fact when there is one, else from the type the axiom's own AST node records (`kind`, checked against the node `type`). The parent's ontology, the fact's identity and value, and its provenance are unchanged. No kind is defaulted: an unrecorded or contradictory type gets no projection, so that parent stays shown but not navigable. `EntityCard` is unchanged; it already navigates only with a typed projection. | **Unit:** the tutorial resource (complete and bounded) yields both typed parents with their fact IDs. A v1-shaped resource covers four parents: typed by edge, typed by AST, untyped and contradictory. The last two stay unprojected but listed, with provenance kept. **Real study service, tutorial:** in lesson 1 both parents ("crate", "lidded object") are enabled and carry `data-kind="class"`. Each opens the Hierarchy tab, focused on that IRI as "Focused class", and shows the parent's own recorded parent; a wrong kind or ontology would find none. The lesson, the inspected candidate and the saved progress are unchanged. In lesson 2 the same card link completes "Open a parent". The service accepts it with HTTP 200 as a typed action (source ontology, IRI, `class`), after checking that action against its recorded hierarchy. **v1 publication:** in a v1 explanation case, every recorded parent of the source opens with its recorded IRI and kind, and the ranking is kept. |
| **R13** racy recovery test | "A comparison failure blocks with its own label; a profile failure likewise" (`e2e/study-v2-integration.spec.ts`) now registers a wait for the `/study/events` response that carries `case_ready` before clicking Retry. It checks that the response is HTTP 200, that it acknowledges exactly the one `case_ready` event sent, and, with a bounded poll, that the recorded traffic holds exactly one `case_ready` request, with status 200. The assertions that loading failures block answering and that Retry restores it are unchanged, and the test has no sleeps. The R07 full-context test ended with the same synchronous count; it now uses the same bounded check. | Five consecutive passes on the real service. A diagnostic delays the real events request without altering it: the old synchronous count reads **0** at UI recovery, while the new wait sees HTTP 200 and the event acknowledged. |

## Reproduction before the correction

These ran against the pre-fix build of `eb4f955`, whose frontend is identical to `e36cfa5`
(that commit only changed documentation).
- **Unit:** both new adapter tests failed on the previous adapter.
- **Browser:** both new R12 regressions failed at the parent's `data-kind` (absent). The tutorial
  regression ran on the real study service; the v1 regression on the v1 development harness.
- **Diagnostic (unmodified real service, scratch only):** lesson 1's two parent buttons were
  `disabled` with no kind. On the fixed build they are enabled with kind `class`.

## Verification

**Service evidence.** These ran against the real study service: HTTPS with a local self-signed
certificate, PostgreSQL 16.2, the compiled static export and the service's normal CSP. Covered:
the tutorial, R10, R11, R13, every earlier v2 regression and the fresh, unseeded participant
journey (`study-v2-backend.spec.ts`).
- The main-app tests ran on `exact-inspect serve` over the same frozen contexts.
- The v1 suite (`study.spec.ts`) ran on the v1 development harness, `tools.serve_explanation_ui_study`.
  That harness is the real study application with its CSP and the compiled frontend, but it
  uses a SQLite test store over loopback HTTP, with the browser's origin presented as the
  configured HTTPS origin. Its v1 evidence is therefore not HTTPS/PostgreSQL evidence.

**Diagnostics.** The R13 delay and the R12 button state check are diagnostics, not acceptance.
They lived only in the scratch copy and were not committed.

**Fixture:** unchanged from the second follow-up (`--navigation-size 65 --paged-facts 26`, 59
files; publication SHA-256 `ea747606…22ac`).

**Results:**
- type check and production build pass;
- unit tests: **75 passed** (2 new);
- browser, complete suite at `1808ed0`: **89 passed, 2 skipped**. The skips are `legacy.spec`,
  because the historical run directory is unavailable. The run includes:
  - the fresh, unseeded participant journey;
  - the R12 tutorial and v1 regressions and the corrected R13 test;
  - every R06–R11 regression, including R11 in both products for both property kinds;
  - the earlier v2 tests;
  - the exploration, library, demo, accessibility and v1 study suites;
- spec validator: no errors.

No Python changed, so the backend suites were not rerun; the second follow-up's 40 passed, 1
skipped still applies. The [receipt](explanation-integration-frontend-followup-3-receipt.json)
lists every test result.

**Environment:** as in the second follow-up, builds and browser runs used locked dependencies
installed outside iCloud and a scratch copy of the frontend synced from the repository before
every build. The repository is the only source.

## Remaining limitations

- **J10 manual audit:** a screen-reader and keyboard pass is still required. It now also covers
  the tutorial card's parent links.
- **Intermittent reload stall:** still open and assigned to the backend (`h11 LocalProtocolError`
  on client disconnect). It did not occur in this round's runs (no such log entry), which is not
  evidence that it is fixed.
- **Untyped terms:** a parent or original-axiom term without a recorded type is shown but not
  navigable. No current fixture or frozen resource has one.
- **v1 browser evidence** comes from the SQLite development harness, as described above.
- **Backend observation (not changed):** the tutorial's action check (`entity_key` in
  `exact_inspect/study/tutorial.py`) treats a missing `kind` as `class`. The frontend always sends
  the kind, and the R12 test asserts it, so this is not a frontend defect. The backend agent may
  want to require the kind.
- **Release gates (unchanged):**
  - the real `backend-release` package;
  - Docker, Render deployment and restore;
  - trusted HTTPS;
  - usability and duration pilot;
  - study-owner consent/information approval, case adjudication and the historical DOID/import-closure gates.

No push, deployment, recruitment or participant contact was performed.
