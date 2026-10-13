# S2/Luna frontend corrections — handoff, 13 October 2026

**Status: the six findings of the [independent S2/Luna review](explanation-s2-luna-independent-review.md)
(R1–R6) are corrected in the frontend and demonstrated on the active Luna publication, in the
main exploration app and in the scoped study workspace.** The full real 24-case study journey,
the type check, unit tests and production build pass on the final build, which is bound to the
Luna publication in a new review environment. Graph discoverability at narrow study widths and
enlarged text was improved and is tested; data limitations are listed separately.

**Follow-up, 13 October:** R2-F1, the remaining finding of the
[corrections review](explanation-s2-luna-frontend-corrections-review.md), is also corrected. A
record stored under several typed entities no longer takes its subject from packet order; see
[R2-F1](#r2-f1-follow-up--one-record-under-several-typed-subjects). With it, R2 is closed for
the tested scope.

This is development-review evidence. It does not close the human-only audits (manual keyboard
and screen-reader review, owner design review, participant use), deployment, launch, or the
backend/fixture items the review assigned to a later pass. Test sessions are development-review
sessions, not research data. No backend file, contract, publication, approved prose or immutable
artifact was changed; no generation, bundle or reasoner run was started.

| Finding | Correction | Demonstrated by |
|---|---|---|
| **R1** generated prose hidden or presented as quotation | Quotation styling, titles and "original"/"definition" wording now come from each artifact's declared preparation and support (`textOrigin`), never from claim category or the paired layout. Reviewed `meaning` prose is shown by default; reviewed per-concept prose reads **"Descriptions side by side"** with generated attribution; only `fact_extract`, exact-support and fallback text is quoted. | All **9** affected comparisons in the main app **and** in study explanation cases: visible exact claim text, "Generated · reviewed", no quotation styling, no "Quoted from each ontology", no misleading "Nothing recorded" line or collapsed quote disclosure. All genuine extracts (paired and one-sided) still quoted with "Original excerpts". Unit tests for every real preparation combination, necessary conditions and fallback facts. |
| **R2** referenced-label citations named the wrong subject | Each citation passes the record's admitted typed subjects, from the artifact's support packet, to the inspector. A caller context counts only when it is one of them. Since R2-F1, all admitted subjects are kept, not just the first. The API adapter resolves against the original axiom's own subject. It keeps a supplied typed subject only for the same IRI and ontology (punning) and for axioms without a single subject (shared axioms). The inspector states the subject's kind and ontology. | All **35** referenced-label citations in the **33** profiles in the main app, plus the three in case review-01 (including ileum cancer → small intestine cancer, `DOID_10154`) through the study scope: About entity, kind, ontology, the literal against the live original axiom, and the original AST (full IRIs). Unit tests: wrong supplied subject, punned kind, shared `EquivalentClasses`, foreign-ontology subject. Typed multi-subject records: see [R2-F1](#r2-f1-follow-up--one-record-under-several-typed-subjects). |
| **R3** failed comparison-data reads looked like missing evidence | The graph distinguishes loading, a successful absence ("No comparison was recorded" / "No comparison record is available for this view"), and failure (**Recorded comparisons: … Try again**). Features, selection and the view stay; an unauthorized read shows the message without Retry. | Main app and study scope: an injected 503 on the comparison read, then Retry — recorded comparison lines return with zoom, pan and the selected feature unchanged. An injected 403 offers no Retry. |
| **R4** equivalence groups limited to one page | Bounded continuation with the service cursor, shown/total count, de-duplicated typed members that refocus the side, loading/error/Retry, and a fresh first page on any focus, scope or basis change. Copy keeps partial-ELK semantics ("a class not listed may still be equivalent"). | An injected 25-member group (20 per page, a failing later page, one duplicate) in the main app and the study scope: "19 of 24 shown" → error → Retry → "24 other classes", 24 distinct members; Asserted → ELK again reads the first page without a cursor; a member link refocuses by typed identity. |
| **R5** comparison captures showed loading placeholders | The capture spec selects every description variant from served presentation metadata and captures only after the tag, heading, exact claim text and quotation styling are asserted and no loading placeholder remains. | New captures from the final environment for reviewed summary, reviewed side-by-side, reviewed grouped, paired extracts, one-sided extracts, the necessary-conditions template and the referenced-label profile, in main and study, 1440 and 375 px, light and dark (see Screenshots). The historical captures are retained unchanged. |
| **R6** hierarchy recursion ended in an unexplained disabled control | Ancestry tracks typed `ontology|kind|IRI` paths. A repeat shows **"already in this path"** with **"Show where it appears"**, which focuses the earlier occurrence (or the focus card); no further reads. At the 12-level limit the row says so and offers **"Continue from …"**, which refocuses there. Parent/child filters compare typed identities. | Injected ancestry in the main app and the study scope: cycles back to the focus and to an intermediate class, a 15-level acyclic chain (limit stated, continuation refocuses and its parents expand), and a punned parent (same IRI, other kind) that is not treated as a cycle. No disabled expansion control remains. |

## Discoverability improvements (frontend) and data limitations (backend)

**Graph context outside the canvas.** Measured on the dense pair review-02 #4 in a study
explanation case:
- At 375 px, 9 of 11 nodes were wholly outside the visible canvas at the minimum readable zoom.
- At 320 px with 200% text the canvas is 156 px wide and all 11 nodes were outside.
- Nothing said so.

The graph now states "*N of M items are outside this view; the item list has every item.*" It
offers one-press buttons ("→ 6 to the right") that pan to the hidden side at the current zoom,
and shades the canvas edges where the drawing continues. Nothing is drawn over the graph. The
graph item list now reflows to one column, which removed a horizontal page scroll at
320 px / 200% text: its fixed 13 rem column minimum was 416 px wide. Tested at 375 px light and
at 320 px / 200% dark.

**Smaller fix.** The browser focus card says "Loading label…" while a label is loading, instead
of claiming "No label in scope".

**Backend/data limitations, unchanged by this pass:**
- **DOID asserted entry points:** 4,189 entry points, mostly obsolete classes. The service
  provides no obsolete/deprecated flag, and its `default_entity` is simply the first ordinal
  entry. Filtering by label text would invent ontology semantics, so the frontend does not.
- **Decision traces:** no recorded stage events for the presented pairs.
- **Evidence and strata:** 8 pairs have no exported selection; the 1–2-feature sparse stratum is
  absent.
- **Legacy fixture:** the exact-study/1.0 fixture no longer starts under the current backend.
- **Description quality:** low-information but accurate descriptions remain.
- **Equivalence groups:** the retained ELK groups are singletons, so R4 is demonstrated with an
  injected group.
- **Typed multi-subject records:** the review's audit found none of the 260 active Luna
  explanations citing a record under several typed subjects. R2-F1 is therefore shown in the
  browser with an injected second subject, and through the component and adapter with native
  fixtures.

## R2-F1 follow-up — one record under several typed subjects

The [corrections review](explanation-s2-luna-frontend-corrections-review.md) reproduced one
remaining typed-identity defect. One original record can belong to several typed entities: a
punned IRI, or an axiom shared by two classes. The citation wrapper kept only the first such
subject in the support packet and let it replace a correct caller context. Reversing the
packet order therefore turned the inspector's individual into a class.

**Correction (frontend only; `2547263`, `2d4600d`):**
- **Keep every subject.** `citedSubjects` keeps every typed subject that the support packet's
  facts and referenced labels admit for a record, de-duplicated by ontology/IRI/kind.
- **Caller context.** `citationSubject` decides which subject a citation is about. The caller
  context is the record's own entity in a shown context, then the block's entity. It is kept
  only when its full ontology/IRI/kind identity equals an admitted subject. Otherwise a single
  admitted subject is used. Several admitted subjects are passed on as candidates, with no
  chosen subject. Packet order never decides.
- **Original-axiom validation (unchanged).** This applies to the API adapter
  (`resolvedFromAxiom`) and the static resolver (`resolveFact`). A supplied subject or candidate
  must belong to the axiom's ontology. When the axiom names one subject, it must also have that
  subject's IRI.
  - A consistent supplied subject comes first, and the other consistent candidates follow.
  - Several candidates without one are returned as explicitly ambiguous, in identity order.
  - With neither, the axiom's own subject is used, as before.
- **Inspector.** The shared inspector is used by both the main app and the study. It shows
  **About** with **Also stored under (one shared original record)**. When the record is
  ambiguous, it instead shows **Stored under N typed entities; this citation does not say which
  one it is about**, listing each kind and ontology. For an ambiguous record it omits the
  relation sentence, which presumes a single subject.
- **The caller does not simply win.** When the artifact admits a single subject, a wrong focal
  caller is still replaced, which is the original R2 case. A caller from another ontology is
  never attached to the record.

| Evidence | Result |
|---|---|
| The review's probe on its native punned label ([probe](explanation-s2-luna-r2-f1/identity_probe.cjs), [fixture](explanation-s2-luna-frontend-corrections-review/identity-fixture.json)), caller = the individual | Before, on the `0e6bebb` source: **class** in one order, individual in the other, so the result depends on order ([probe-before.json](explanation-s2-luna-r2-f1/probe-before.json)). After: **individual in both orders**, with the class listed as also stored under ([probe-after.json](explanation-s2-luna-r2-f1/probe-after.json)). |
| Complete-path regression `src/lib/workspace/citationIdentity.test.ts` | Runs the real `GeneratedBlock` wrapper, the citation button's action and the API adapter. It uses native backend facts ([generator](explanation-s2-luna-r2-f1/native_fixtures.py)): the punned label, and an `EquivalentClasses` axiom stored under two classes. Covered: both packet orders; the caller as context record or as block subject; callers with a wrong kind, the wrong IRI (the focal entity) or a foreign ontology; a single admitted subject overriding a wrong caller; a referenced label; a legacy artifact without support subjects. **On the pre-fix source the three R2-F1 tests fail** and the R2 test passes. On the fix all 5 pass. |
| Static resolver `resourceIndex.test.ts` | The shared practice record resolves about a matching caller; otherwise it is ambiguous with both subjects in identity order; a foreign caller is ambiguous. |
| Browser, shared inspector (`luna-corrections.spec.ts`, R2-F1 block) | Uses the real ileum cancer profile (review-01, position 4). A punned individual copy of two real cited label records is injected into the served support packet by route interception, before the original in one test and after it in the other. In the main app and the study workspace, **"Target label"** opens about the class, with the individual also listed. **"Label of DOID:10154"** shows both typed subjects as unresolved: its caller is the focal entity, which is not admitted. The captures are byte-identical for the two orders. |

[explanation-s2-luna-r2-f1/](explanation-s2-luna-r2-f1/) holds the probe and its before/after
results, the fixture generator and four inspector captures (added-first; the added-last
captures have the same SHA-256). Its [receipt](explanation-s2-luna-r2-f1/receipt.json) records
the bindings, the per-test outcomes of both runs and the storage figures.

**Verification of `2d4600d`.** Browser runs used `environment-frontend-2d4600d` on the Luna
publication, with system Chrome through Playwright against HTTPS with application CSP.

| Check | Result |
|---|---|
| Unit (`npm run test:unit`) | **131 passed**, 0 failed or skipped (125 before, plus the 5 complete-path tests and 1 order-independence test for `citationSubject`) |
| Type check (`tsc --noEmit`) | passed |
| Production build | Passed (Next.js 15.5.26 static export). A rebuild after the browser runs equals the bound output in all 62 files, apart from Next.js's random build ID. |
| Acceptance set, run 1 (fresh environment) | **24 passed, 1 failed.** The R2 main-app test timed out waiting for "Label of DOID:4110": the mite infestation profile's description and comparison reads failed with `network_unreachable` (the browser's fetch was rejected) from the newly started exploration service, so the citation never rendered. All other tests passed, including R2 in the study and both R2-F1 tests (11 s and 17 s). The citation code is not on that path: the profile read happens before any citation is clicked. The service logs warnings only, so no server-side cause was recorded. |
| Acceptance set, run 2 (same environment, same build) | **25 passed**, 0 failed, skipped or flaky, in 7.1 min. The full 24-case journey passed in 91 s; R2 main-app (all 35 referenced-label citations) in 52 s, compared with 51 s in the review; R2 study; both R2-F1 tests; R1, R3, R4, R6, the narrow-canvas tests, both hierarchy tests and the 8 assessment-recovery tests. |

**Bindings.** Frontend source tree `e67b6235…` (`git rev-parse 2d4600d:explanations_visualizer`).
Compiled output `~/Library/Caches/exact-om-review/luna-bundle-20261012/frontend-2d4600d`, tree
hash `07bdf63e4954b1185a11e453f14eb597aaa34904e65039faea7fd28b7355ddb2`. Environment
`environment-frontend-2d4600d` (PostgreSQL 19623, study HTTPS 19624, exploration HTTPS 19625).
Its services run from the clean detached worktree `~/Library/Caches/exact-om-dev/s2dev/source-2d4600d`.
The test publication is `review-20261013T020941-03e25f85da`, SHA-256 `c8d523a3…`. The source
publication `51500ab8…`, package `07781b63…` and backend fingerprint `752b9144…` are unchanged.

**Remaining limitations of R2-F1:**
- **Injected browser case.** No active Luna artifact cites a record under several typed
  subjects, so the rendered browser case is an injected copy (route interception). The native
  punned and shared-axiom records run through the real component and adapter code, not through
  a native publication served to the browser.
- **Legacy artifacts.** Artifacts without support-packet subjects still treat the caller as a
  hint checked against the axiom. An annotation fixes the subject's IRI but not its kind, so a
  same-IRI caller of the wrong kind cannot be detected there. Without a caller, an annotation's
  subject kind is still shown as class. Every Luna artifact carries support subjects.
- **Wording not reviewed.** The ambiguity wording and layout have not had an owner design review.
- **Run 1 failure not diagnosed.** The dropped service reads in run 1 were not investigated
  further: the backend is out of scope, and the service has no access log. They did not recur in
  run 2.

## Commits

| Commit | Content |
|---|---|
| `f2bdf00` | R1–R4 and R6 corrections, inspector subject details, graph off-canvas cues and reflow, focus-label loading state; unit tests |
| `bb9a1a6` | `e2e/luna-corrections.spec.ts`, `e2e/reviewData.ts`, the corrected capture spec `e2e/review-visual-s2.spec.ts` |
| `0e6bebb` | this handoff, the receipt and screenshots; an optional evidence-capture helper in `luna-corrections.spec.ts` (test-only, `EXACT_E2E_SHOTS`) |
| `2547263` | R2-F1: every admitted typed subject kept; caller context only on a full identity match; explicit ambiguity in the resolvers and the inspector; native fixtures and the complete-path regression `citationIdentity.test.ts` |
| `2d4600d` | R2-F1 browser regression in `luna-corrections.spec.ts` (injected second subject, both orders, main app and study) |
| R2-F1 handoff | this document's R2-F1 updates and [explanation-s2-luna-r2-f1/](explanation-s2-luna-r2-f1/) |

## Bindings

These are the R1–R6 bindings (`bb9a1a6`). The current `2d4600d` bindings and checks are in
[R2-F1](#r2-f1-follow-up--one-record-under-several-typed-subjects).

| Identity | Value |
|---|---|
| Frontend source tree (`git rev-parse bb9a1a6:explanations_visualizer`) | `7f7164c954135b24595a0e9230ac8e17c40b0231` |
| Compiled frontend output | `~/Library/Caches/exact-om-review/luna-bundle-20261012/frontend-bb9a1a6` (Next.js 15.5.26 static export) |
| Frontend tree hash bound by the environment | `385a8c6cbf47d996f7fc919e499f707cc83180482cca081293772933b266444f` |
| Luna source publication `publication-luna-20261012.json` | `51500ab8b54615ff938da3e70faa6816af67c6867b1b528f4d2a175cb68bff93` — equals the [Luna handoff](explanation-luna-publication-20261012.md) |
| Luna package `package-luna-20261012.json` | `07781b63d2acd24f2542653112e5a20d75cea30af55685b331af3ea1e9b8907b` — equals the Luna handoff |
| Backend runtime fingerprint | `752b91446f125a12af47d9c6f7c7499c3edba40ea85d5d84fd5bfa86491c8985` — equals the reviewed Luna environment; services run from a clean detached worktree of `bb9a1a6` (`~/Library/Caches/exact-om-dev/s2dev/source-bb9a1a6`), so uncommitted edits in the shared checkout cannot enter them |
| Review environment | `~/Library/Caches/exact-om-review/luna-bundle-20261012/environment-frontend-bb9a1a6` — PostgreSQL 19613, study HTTPS 19614, exploration HTTPS 19615 (loopback) |
| Environment test publication | `review-20261012T233746-16e0d27411`, SHA-256 `883ee873aa705a97a6901784f8474b3a8c431cb5f4c7cca987883add9e080ce4` |

The existing Luna environment (`environment`, 19603–19605, frozen S2 frontend `b5511d27…`) was
not modified; a compiled frontend cannot be swapped into a prepared environment, so the
corrected build has its own root. The publication, package, data root and databases were reused
in place.

## Verification

All browser runs used system Chrome through Playwright, against the HTTPS services with normal
CSP. "Final environment" means `environment-frontend-bb9a1a6` on the Luna publication. This
table is the R1–R6 verification of `bb9a1a6`. The R2-F1 build `2d4600d` was re-verified as
described in its section.

| Check | Result |
|---|---|
| Type check (`tsc --noEmit`) | passed |
| Unit (`npm run test:unit`) | **125 passed**, 0 failed (3 new: R1 attribution for every real preparation combination; R2 support-packet subjects; R2 adapter resolution with a wrong supplied subject, punning, a shared axiom and a foreign ontology) |
| Production build | passed (Next.js 15.5.26 static export) |
| **Full real 24-case study journey** (`study-review-journey.spec.ts`), final environment | **passed** (86 s). No API-seeded tutorial completion. It covers setup/background forms, every lesson with an injected rejection and recovery, wrong assessment retries, all 24 cases in both conditions with changing methods, completion, and analysis/4 JSON/CSV exports with and without named methods. It also asserts that baseline cases make no workspace request and that only study routes are used. |
| Acceptance set, final environment | **23 passed**, 0 failed: the journey; 2 real hierarchy tests; 12 S2/Luna regressions (R1 ×2, R2 ×2, R3 ×2, R4 ×2, R6 ×2, narrow canvas ×2); 8 assessment-recovery tests (two-tab conflicts, reload, replay) |
| Visual captures (`review-visual-s2.spec.ts`), final environment | **10 passed**, 107 captures, each asserted before capture |
| Regression evidence captures (R3/R4/R6), final environment | **6 passed**, 10 captures of the injected states |
| Complete suite, final environment | **80 passed**, 26 not passed, 60 skipped. The 26 are the same synthetic-fixture specs as the [previous delivery](explanation-postmigration-frontend-integration.md#verification), test for test: their pairs, paged-fact and property fixtures and variant publications do not exist in the Luna publication. Skips are opt-in captures and synthetic-only or forms/2-only tests. |
| Complete suite, synthetic S0 harness (forms/2, same build, services from the clean worktree) | **101 passed**, 19 failed, 46 skipped. All 19 failures are the legacy exact-study/1.0 study server, which does not start under the current backend (connection refused on its port). All S0 graph, tutorial, assessment, integration and forms/2 journey regressions pass. |

Baseline isolation and questionnaire contracts are unchanged by this pass (no study-protocol
code changed). The journey asserts that baseline cases issue no workspace request. It also checks
the setup/3 receipt, the strict consultation/3 bodies and the analysis/4 exports. No backend validation was relaxed; the injected failures in R3 and R4 and the ancestry
in R6 are route interceptions in the browser, named as such in the tests and screenshots.

## Screenshots

[explanation-s2-luna-frontend-corrections/](explanation-s2-luna-frontend-corrections/) holds 38
captures from the final environment (frontend `385a8c6c…`, test publication `883ee873…`). The
[receipt](explanation-s2-luna-frontend-corrections/receipt.json) lists each with its SHA-256,
together with every test outcome. Each description capture was taken only after its heading,
attribution tag, exact claim text and quotation styling were asserted, with no loading placeholder
present. The superseded loading-placeholder images in the
[previous handoff folder](explanation-postmigration-frontend-integration/) are retained unchanged
as history.

| Area | Captures |
|---|---|
| R1 comparisons, main | reviewed summary (light/dark), reviewed descriptions side by side (1440 light, 375), reviewed grouped, paired extracts, one-sided extract (dark) |
| R1 comparisons, study | reviewed summary (1440 dark, 375), descriptions side by side (1440, 375 dark), one-sided extract, paired extracts (375) |
| R2 citations | necessary-conditions template and reviewed referenced-label profiles; the inspector for "Label of DOID:10154" (light/dark) |
| R3 / R4 / R6 (injected states) | failure and recovered graph, main and study; 25-member group after Retry, main and study; cycle rows and the depth limit, main and study |
| Graph | dense (dark) and sparsest (light) panels; narrow study graph at 375 px light and 320 px / 200% dark |
| Journey | tutorial rejection, explanation case, graph, consultation, completion; study explanation case; tutorial graph lesson (375) |

## Reproduction

The current build is `2d4600d` in `environment-frontend-2d4600d` (ports 19623–19625).
`environment-frontend-bb9a1a6` (19613–19615) is retained as the R1–R6 binding; to use it,
substitute its name, worktree and ports.

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
LUNA=/Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012
WT=/Users/pedrocotovio/Library/Caches/exact-om-dev/s2dev/source-2d4600d
cd /Users/pedrocotovio/Developer/Repos/OM/Exact-OM
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m tools.review_storage_budget check --growth-gb 1
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -m tools.review_environment status --root "$LUNA/environment-frontend-2d4600d"
```

If stopped, start it with the same code and the bounded reservation in
`environment-frontend-2d4600d/storage-reservation.json` (1 GB, 128 MiB per file):

```sh
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$LUNA/environment-frontend-2d4600d"
```

Browser tests need a private harness file (mode 0600, outside the repository) with `origin`,
`researcher_token`, `publication`, `study_revision` and `assets` read from `environment.json`
without printing it. From `explanations_visualizer` with its locked dependencies:

```sh
EXACT_E2E_URL=https://127.0.0.1:19625 EXACT_E2E_REVIEW_MAIN_URL=https://127.0.0.1:19625 EXACT_E2E_STUDY_V2_CONFIG=/private/path/harness.json npx playwright test e2e/study-review-journey.spec.ts e2e/review-hierarchy.spec.ts e2e/luna-corrections.spec.ts e2e/assessment-recovery.spec.ts
```

Add `EXACT_E2E_SHOTS=<dir>` to capture the regression states (R2-F1, R3, R4, R6). To recapture
the visual evidence, add `EXACT_SCREENSHOT_DIR=<dir>` and `e2e/review-visual-s2.spec.ts`.
Never print or commit `environment.json`, keys, invitations, cookies or participant identifiers.

## Storage

| Point | Combined use (decimal) |
|---|---|
| Start of this pass (12 Oct, 22:42 UTC) | 50.049 GB |
| Final (13 Oct, 00:40 UTC) | **50.361 GB** (118.6 GB filesystem free) |
| R2-F1 pass start (13 Oct, after the review) | 50.406 GB |
| R2-F1 pass final (13 Oct, 02:46 UTC) | **50.590 GB** (118.4 GB filesystem free) |

The R2-F1 pass added:
- `environment-frontend-2d4600d` (125 MB, including its test sessions);
- its frontend output (2.4 MB);
- the clean worktree `source-2d4600d` (56 MB);
- test output and captures in `s2dev/final-r2f1` (7.6 MB);
- the evidence folder and native fixture in the repository (0.3 MB).

It also re-used the existing Luna publication, package, data root, dependencies and toolchain,
and changed no immutable or shared file.

The R1–R6 pass added:
- the new review environment (156 MB, including its test sessions);
- the frontend output (2.3 MB);
- the clean worktree (50 MB);
- test output and captures under `~/Library/Caches/exact-om-dev/s2dev` (182 MB, including both
  of its worktrees).

Every copy, the worktree, the environment preparation (2 GB estimate) and each build (1 GB
estimate) ran under `tools.review_storage_budget`. Detached services have a recorded 1 GB
reservation and a 128 MiB per-file limit. Other sessions may also have written during the
interval; these are measured points, not an instantaneous peak.

No dependency, dataset, bundle, publication, paid call or reasoner output was created.

## Remaining limitations and open gates

**Technical, not closed by this pass (from the review's backend/fixture list):**
- The legacy exact-study/1.0 fixture does not start under the current backend, so the 19 v1
  browser tests remain unverified. Restoring a valid historical fixture is open technical work,
  not a human audit.
- Obsolete DOID entry points dominate the asserted top-level list. A useful non-obsolete choice
  needs a served obsolete/deprecated signal or curated defaults; the frontend does not infer it
  from label text.
- Low-information but accurate descriptions (for example, subtypes that restate their parent)
  need a backend content-quality pass. Approved claims were not reworded in display code.
- R4's paging is demonstrated with an injected group: every retained ELK group is a singleton.
- R6's cycles and long chains are injected: the review reproduced no cycle in the retained
  taxonomies, and none was needed to exercise the behaviour.
- R2-F1's limits are listed in [its section](#r2-f1-follow-up--one-record-under-several-typed-subjects):
  an injected browser case, legacy artifacts without support subjects, and ambiguity wording
  without owner review.
- Production deployment and restore have not been exercised. This is technical acceptance work,
  not a human audit.

**Data boundaries (unchanged):**
- no recorded decision-stage events for the presented pairs;
- 8 pairs without exported selection;
- no 1–2-feature sparse stratum;
- partial ELK only (an absence is unknown, not a negative).

**Human-only, still open:**
- manual keyboard and screen-reader audits;
- owner design review;
- participant/pilot use;
- external review.

Historical R16 was not reproduced. This is not a launch-readiness claim.

**Graph at 320 px / 200% text.** The graph canvas is only about 156 px wide, so the canvas itself
is of limited use there. The new status row and the full item list carry the content. A
list-first default at that size is a possible design follow-up, not implemented here.
