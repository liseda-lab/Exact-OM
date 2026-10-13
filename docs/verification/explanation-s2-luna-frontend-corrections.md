# S2/Luna frontend corrections — handoff, 13 October 2026

**Status: the six findings of the [independent S2/Luna review](explanation-s2-luna-independent-review.md)
(R1–R6) are corrected in the frontend and demonstrated on the active Luna publication, in the
main exploration app and in the scoped study workspace.** The full real 24-case study journey,
the type check, unit tests and production build pass on the final build, which is bound to the
Luna publication in a new review environment. Graph discoverability at narrow study widths and
enlarged text was improved and is tested; data limitations are listed separately.

This is development-review evidence. It does not close the human-only audits (manual keyboard
and screen-reader review, owner design review, participant use), deployment, launch, or the
backend/fixture items the review assigned to a later pass. Test sessions are development-review
sessions, not research data. No backend file, contract, publication, approved prose or immutable
artifact was changed; no generation, bundle or reasoner run was started.

| Finding | Correction | Demonstrated by |
|---|---|---|
| **R1** generated prose hidden or presented as quotation | Quotation styling, titles and "original"/"definition" wording now come from each artifact's declared preparation and support (`textOrigin`), never from claim category or the paired layout. Reviewed `meaning` prose is shown by default; reviewed per-concept prose reads **"Descriptions side by side"** with generated attribution; only `fact_extract`, exact-support and fallback text is quoted. | All **9** affected comparisons in the main app **and** in study explanation cases: visible exact claim text, "Generated · reviewed", no quotation styling, no "Quoted from each ontology", no misleading "Nothing recorded" line or collapsed quote disclosure. All genuine extracts (paired and one-sided) still quoted with "Original excerpts". Unit tests for every real preparation combination, necessary conditions and fallback facts. |
| **R2** referenced-label citations named the wrong subject | Each citation carries its exact typed subject from the artifact's support packet (or the context record) into the inspector; the API adapter resolves against the original axiom's own subject, keeping a supplied typed subject only for the same IRI and ontology (punning) and for axioms without a single subject (shared axioms). The inspector states the subject's kind and ontology. | All **35** referenced-label citations in the **33** profiles in the main app, plus the three in case review-01 (including ileum cancer → small intestine cancer, `DOID_10154`) through the study scope: About entity, kind, ontology, the literal against the live original axiom, and the original AST (full IRIs). Unit tests: wrong supplied subject, punned kind, shared `EquivalentClasses`, foreign-ontology subject. |
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
- **Duplicate citation subjects:** if a cited record appears under several subjects in one
  artifact's support packet, the inspector opens it about the first. The resolved original
  axiom still decides when it names a single subject.

## Commits

| Commit | Content |
|---|---|
| `f2bdf00` | R1–R4 and R6 corrections, inspector subject details, graph off-canvas cues and reflow, focus-label loading state; unit tests |
| `bb9a1a6` | `e2e/luna-corrections.spec.ts`, `e2e/reviewData.ts`, the corrected capture spec `e2e/review-visual-s2.spec.ts` |
| this handoff | this document, the receipt and screenshots; an optional evidence-capture helper in `luna-corrections.spec.ts` (test-only, `EXACT_E2E_SHOTS`) |

## Bindings

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
CSP. "Final environment" means `environment-frontend-bb9a1a6` on the Luna publication.

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

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
LUNA=/Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012
WT=/Users/pedrocotovio/Library/Caches/exact-om-dev/s2dev/source-bb9a1a6
cd /Users/pedrocotovio/Developer/Repos/OM/Exact-OM
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m tools.review_storage_budget check --growth-gb 1
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -m tools.review_environment status --root "$LUNA/environment-frontend-bb9a1a6"
```

If stopped, start it with the same code and the bounded reservation in
`environment-frontend-bb9a1a6/storage-reservation.json` (1 GB, 128 MiB per file):

```sh
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$LUNA/environment-frontend-bb9a1a6"
```

Browser tests need a private harness file (mode 0600, outside the repository) with `origin`,
`researcher_token`, `publication`, `study_revision` and `assets` read from `environment.json`
without printing it. From `explanations_visualizer` with its locked dependencies:

```sh
EXACT_E2E_URL=https://127.0.0.1:19615 EXACT_E2E_REVIEW_MAIN_URL=https://127.0.0.1:19615 EXACT_E2E_STUDY_V2_CONFIG=/private/path/harness.json npx playwright test e2e/study-review-journey.spec.ts e2e/review-hierarchy.spec.ts e2e/luna-corrections.spec.ts e2e/assessment-recovery.spec.ts
```

Add `EXACT_SCREENSHOT_DIR=<dir>` and `e2e/review-visual-s2.spec.ts` to recapture the evidence.
Never print or commit `environment.json`, keys, invitations, cookies or participant identifiers.

## Storage

| Point | Combined use (decimal) |
|---|---|
| Start of this pass (12 Oct, 22:42 UTC) | 50.049 GB |
| Final (13 Oct, 00:40 UTC) | **50.361 GB** (118.6 GB filesystem free) |

This pass added:
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
- If one artifact lists the same record under several subjects, the inspector opens it about the
  first, unless the original axiom itself names a single subject.

**Data boundaries (unchanged):**
- no recorded decision-stage events for the presented pairs;
- 8 pairs without exported selection;
- no 1–2-feature sparse stratum;
- partial ELK only (an absence is unknown, not a negative).

**Human-only, still open:**
- manual keyboard and screen-reader audits;
- owner design review;
- participant/pilot use;
- production deployment and restore;
- external review.

Historical R16 was not reproduced. This is not a launch-readiness claim.

**Graph at 320 px / 200% text.** The graph canvas is only about 156 px wide, so the canvas itself
is of limited use there. The new status row and the full item list carry the content. A
list-first default at that size is a possible design follow-up, not implemented here.
