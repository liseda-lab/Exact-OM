# Human-review readiness — frontend corrections (spec 24, stage R1)

13–14 October 2026. Frontend handoff for
[spec 24](../../specs/explanation-framework/24-human-review-readiness-corrections.md) after the
[independent readiness review](explanation-human-review-readiness-20261013.md). Base commit
`1bd10c9`; final frontend commits `bbedb3c`, `fc5558c`, `fffa6d8`, `1b878dd`, `6dc6cbe`. Sanitized
evidence: [explanation-readiness-frontend-corrections-20261013/](explanation-readiness-frontend-corrections-20261013/)
([receipt](explanation-readiness-frontend-corrections-20261013/receipt.json),
[browser outcomes](explanation-readiness-frontend-corrections-20261013/browser-outcomes.json)).

**Status:**
- **HR2 and HR4 are fixed and verified** against the real review package in a new managed
  environment bound to the final export.
- **HR1 is unresolved: not reproduced in the recorded follow-up runs.** Boundary-specific style
  regression checks and bounded diagnostics are now part of the journey, and their ability to
  detect the symptom is demonstrated by fault injection. No frontend or server cause was
  established.
- **HR3 belongs to the backend** and was not touched.

This is a frontend stage handoff, not a clean combined verdict. No backend file, contract,
publication, package or immutable artifact was changed.

## Dispositions

| Item | Disposition | Evidence / next owner |
|---|---|---|
| **HR1** study lost styling after the assessment reload | **Unresolved / not reproduced** in the recorded follow-up runs. Regression coverage and diagnostics added; no causal fix claimed. | [HR1](#hr1--study-styling-after-reload). Backend: static-delivery verification (§3.4) and the [dependencies](#backend-dependencies) below. |
| **HR2** fresh main visit started outside the prepared pool | **Fixed and verified** | [HR2](#hr2--prepared-examples-on-first-entry); real catalogue browser paths plus labelled injected negatives. |
| **HR3** formatting baseline | **Outside frontend ownership; not run** | Backend/code-quality owner. |
| **HR4** curated hierarchy default not consumed | **Fixed and verified** | [HR4](#hr4--curated-hierarchy-start). |

| Check | Disposition |
|---|---|
| **A24-01** styling and reload | Boundary-specific checks added, and the original failure is preserved. Both allocation orders pass on the final binding, with normal CSP. Bounded diagnostics recorded. **The causal disposition is unresolved.** |
| **A24-02** static delivery | **Backend-owned; not run as a delivery test here.** Browser-side network observations of the bound CSS (status, MIME, length, complete body, hash against the export, cache headers) are recorded for backend use. |
| **A24-03** first entry | **Verified:** real fresh landing; example switching and selection beyond the first source page; All sources; deliberate unprepared deep link; remembered choice; Back/Forward; reload. Missing-pair and pair-only links, failed/late/slow catalogues and candidate paging are covered too (injected cases labelled as such). |
| **A24-04** hierarchy entry | **Verified:** real curated default selected and labelled; explicit, pinned and in-flight choices respected; other roots and continuation without duplicates. Older and default-free pages (injected) and partial ELK are covered. No root preload: only the first entry page is read. |
| **A24-05** study isolation | **Verified** (details below): only study routes are read, the study service has no catalogue route, and About still strips `cases`. |
| **A24-06** quality | **Frontend verified:** fresh type check, 141 unit tests and production build. **Backend formatting and CI belong to HR3 and were not run here.** |
| **A24-07** rendered experience | **Captured and inspected.** Light and dark, desktop, 375 px and 320 px at 200% text, with Axe and overflow checks before every capture. One keyboard-focus bug and two narrow-layout defects were found and fixed. Remaining layout limits are listed below. |
| **A24-08** preservation and operations | **Verified:** a new managed binding; unchanged package and publication hashes; budget receipts; managed status/stop/start/resume and `fresh-session` exercised. |

**A24-05 details:**
- The journey asserts that only `/api/v1/study/` routes are read and that baseline cases make
  no workspace request.
- A static unit test keeps the catalogue out of all participant and shared-workspace code.
- The study service returns 404 for `/api/v1/review-bundle`.
- Study About still strips `cases` (`exact_inspect/study/store.py:578`, unchanged).

## HR1 — study styling after reload

### Original observation (preserved, unchanged)

The reviewer's failed run is retained in
[the readiness evidence](explanation-human-review-readiness-20261013/):
`unstyled-study-baseline.png`, `styled-tutorial-before-failure.png` and `browser-outcomes.json`
(24 passed, 1 failed at `study-review-journey.spec.ts:295`). That run captured no stylesheet
network evidence.

### What was added

**Boundary style checks.** The review journey asserts application styling at nine stages:

- study entry;
- the lesson-two reload;
- lesson six;
- the assessment question-five reload;
- the first case after the tutorial;
- the first baseline case and the first explanation case;
- the consultation reload;
- completion.

At each stage the stylesheet's own token (`--paper`) must be computed, and inline icons must be
sized by it (≤ 64 px; the incident showed about 900 px). The wait is bounded (10 s) and the
viewport is captured immediately on loss. The journey also asserts that **no application
stylesheet transfer failed**; a reload's cancellation does not count.

**Diagnostics** (`e2e/styleDiagnostics.ts`):
- Per document:
  - stylesheet link insertion and removal;
  - `href` path, `rel`, `media` and `disabled`;
  - load and error events, and rule availability;
  - CSP violations;
  - page errors and unhandled rejections.
- Per request, through the DevTools protocol:
  - CSS initiation and cache source;
  - status, MIME, declared length and content encoding;
  - encoded and decoded sizes;
  - completion, failure or cancellation;
  - the body's SHA-256 compared with the export file.
- Paths only. No query, fragment (invitation), DOM text, answer, token, cookie or storage value
  is recorded. Output is written only to `EXACT_E2E_STYLE_DIAGNOSTICS`, and it is separate from
  study responses and telemetry.
- `EXACT_E2E_BOUNDARY_RELOADS` (at most 10) adds investigation reloads at the assessment
  boundary.

**Check validity** (`e2e/style-boundary.spec.ts`, synthetic fault injection, not a reproduction).
One document's stylesheet transfer is aborted. The page renders exactly the incident's symptom
([capture](explanation-readiness-frontend-corrections-20261013/screenshots/injected-stylesheet-failure-synthetic.png)):
browser-default text and an icon 1,424 px wide. The check rejects it and the failed transfer is
recorded; the next ordinary reload is styled again.

### Bounded investigation (§3.3)

All runs used the managed HTTPS services with normal application CSP, system Chrome, one worker
and no retries.

| Run | Binding | Result | Style evidence |
|---|---|---|---|
| 1. Instrumented four-file batch | Original failing export `07bdf63e` (environment-final, 19653–19655) | **25 passed** | Baseline-first, the same order as the incident. 9/9 boundary checks styled. 4 CSS transfers: 200 `text/css`, 70,441 bytes decoded, all completed, bodies equal to the export. Each document: link added and loaded; no removal, CSP violation or page error. |
| 2. Targeted journey with 5 extra assessment reloads | Same | **1 passed** | Explanation-first. 14/14 checks styled; 9 transfers complete and matching the export. |
| Diagnostic: keep-alive timing | Same study origin | **0 of 32** CSS fetches failed | Idle gaps of 4.8–5.2 s around uvicorn's 5 s keep-alive timeout. Labelled diagnostic, not a reproduction. |
| Diagnostic: cache behaviour | Same | Every reload refetched the CSS | Same result with the test-certificate allowance and with the certificate trusted by SPKI. The cause is server headers (below). |

One of the two allowed targeted journeys and 5 of the 10 allowed extra boundary reloads were
used. The second targeted journey was not needed because the final binding covered both orders.
After the fix work, the final matrix added four more instrumented journeys, all styled at every
boundary:
- final run 1 and final run 2 (baseline-first);
- an additional baseline-first journey (an allocation-order miss: earlier study tests had
  consumed allocation slots);
- an explanation-first journey.

### Findings that constrain the hypotheses

- **The stylesheet is never served from cache.** Both services send `Cache-Control: no-store`
  (exploration) or `private, no-store` (study) on `/_next/static/` assets. This overrides the
  `public, max-age=31536000, immutable` that `exact_inspect/frontend.py` sets. The overriding
  layers are the exploration HTTP middleware (`exact_inspect/service.py:405`) and the study's
  `StudyProtectionMiddleware` (`exact_inspect/study/api.py:138`). Every reload, in every browser,
  therefore re-transfers the whole stylesheet (70 KB), so a failed transfer always leaves the
  page unstyled. A "same-context cached reload" cannot happen in this deployment.
- Chrome also caches nothing from a certificate-error origin. The test allowance alone would
  prevent caching, but the headers above prevent it regardless.
- On the final binding, both services deliver `16a388574bac8a7c.css` as 200
  `text/css; charset=utf-8`, 71,010 bytes, complete, with a body identical to the export. A
  missing CSS path returns a 404 JSON response, not HTML.
- The exploration service wraps every response, including static files, in Starlette's
  `call_next` HTTP middleware. The study service uses pure ASGI middleware. Versions: Starlette
  0.48.0, uvicorn 0.35.0 (default 5 s keep-alive, warning-level logs, no access log). There is no
  server-side record of individual static transfers.

### Remaining hypotheses and the next discriminating experiment

1. **A failed or truncated stylesheet transfer at a reload** (reset, TLS close or
   content-length mismatch). It reproduces the symptom exactly (injected check above); it is
   neither observed nor excluded.
2. **The stylesheet link removed or replaced** during a client stage transition. No removal was
   seen in six instrumented journeys.
3. **A harness lifecycle effect** (the lesson-four route interception or reload timing). There
   is no supporting evidence.

**Next experiment.** Keep `EXACT_E2E_STYLE_DIAGNOSTICS` on in every acceptance run; the journey
now fails at the exact boundary, with the CSS transfer record attached. On a recurrence, the
transfer outcome and link lifecycle distinguish hypothesis 1 from 2 and 3. If it shows a failed
transfer, the backend's opt-in static-read observation (§3.4) correlates it with the server. No
further blind repetition was done.

## HR2 — prepared examples on first entry

**Behaviour** (`CompareView`, `PreparedExamples`, `lib/reviewCatalogue.ts`):

- **Catalogue read.** The main app reads `GET /api/v1/review-bundle` (exact-review-about/1) once
  per package, with a cancellable request.
- **Rail switch.** The rail offers **Prepared examples · 24 | All sources · 300**: a two-choice
  radio group, one tab stop, with arrows to switch. Focus stays on the switch when the list
  changes.
- **Landing priority:**
  1. a valid explicit URL;
  2. a deliberate choice saved for the **exact package and run** (localStorage
     `exact.compare.choice`, written only on deliberate clicks);
  3. the catalogue's first example;
  4. the run's first source.

  A visitor who chooses All sources, or picks a source while the catalogue loads, is never moved
  by a late response. A catalogue slower than 8 s leaves ordinary browsing in place.
- **The example's pair.** The pair is the first candidate, in the order the rail shows (the
  matcher's), whose typed target (ontology/IRI/kind) the case names. It is resolved through the
  run's own candidate list. There are no derived pair hashes, the catalogue's target order is
  not treated as a rank, and no IDs are hard-coded in React.
- **All sources unchanged.** All sources keeps the full source pool and paging, and the same
  candidate list, order, scores and decisions. Catalogue targets are only tagged **Prepared
  example**, with "5 of these candidates are prepared for this review; the order is the
  matcher's." Unprepared pairs still show "No generated description was prepared."
- **Selections outside loaded pages.** A source beyond the loaded pages stays selected
  ("Selected · not in the sources loaded below"); 15 of the 24 example sources are beyond the
  first page.
- **Explicit pairs.** An explicit pair beyond the first candidate pages is searched for (at most
  20 more pages) and stays selected. If it is not a candidate, an alert offers **Open the first
  candidate** or **Open a prepared example**; it is never silently replaced. A pair-only link is
  resolved to its source through the run's pair record.
- **Catalogue failures.** A declared absence (`review_bundle_not_available`, or `not_found` from
  a service without the route) means ordinary browsing. Invalid (409), incompatible schema,
  denied (401/403) and unreachable catalogues are shown with **Try again**. A catalogue for a run
  not in the bundle says so.
- **Browse → Compare return** restores the pair, source kind, list, open tab and each side's
  hierarchy focus.

**Design choices.**
- The default example is the catalogue's first case. Catalogue order is used, not a quality
  rank.
- Examples mode takes the first prepared candidate, while selecting a source in All sources
  keeps the matcher's first candidate.
- "Prepared" means named by the development-review catalogue. It is not a claim of rich prose or
  a correct mapping.

**No contract change.** The new URL parameters (`list`, `sk`, `from_*`) and the localStorage key
are frontend state only.

## HR4 — curated hierarchy start

**Behaviour:**
- `EntryPointPage.selection` is consumed (`entryStart` in `lib/workspace/navigation.ts`). A
  curated start is applied only when all of these hold:
  - `selection.kind = curated_default`, with a navigation-only assertion;
  - a `default_entity` for this exact ontology version and kind, within the response scope;
  - an available page.
- `default_entity` alone, as on prepared-order and partial-ELK pages, is never treated as
  curated.
- Older pages without `selection` keep the single-root rule. Several ordinary roots stay a list.
- With no explicit, remembered or pinned focus, the browser opens the default and its children
  and labels it **Suggested starting point**, with **Show all top-level classes**. The suggestion
  is local navigation: it is not written to the URL or history or remembered as a deliberate
  focus, so it applies again after a reload, and any deliberate focus, including one chosen
  while the default loads, wins.
- The top-level list keeps "1 of 4189 top-level entries shown", **Load more** and every root,
  with the suggestion marked once. Partial ELK keeps the current focus but shows no label.
- The study's and Compare's compared/pinned entity is never replaced; in the pair workspace
  the suggestion appears only as a label in the list.

## Verification

| Command (from `explanations_visualizer`, locked dependencies) | Result |
|---|---|
| `npm run typecheck` | passed |
| `npm run test:unit` | **141 passed**, 0 failed or skipped (131 before; new: 4 `entryStart`, 5 catalogue/landing, 1 catalogue isolation) |
| `npm run build` (under the storage guard) | passed; Next.js 15.5.26 static export |
| Final matrix run 1 — the four required files + `readiness-entry` + `readiness-visual` | **49 passed, 3 failed** (preserved; see below) |
| Final matrix run 2 — same binding, after test corrections | **52 passed**, 0 failed, skipped or flaky (7.9 min) |
| Additional full journeys (same binding) | baseline-first **passed**; explanation-first **passed** |
| `style-boundary.spec.ts` (injected) | **passed** |
| Smoke after managed stop/start | 2 passed |

**Final run 1 failures**, all test-side and preserved in the outcomes file:
- Two narrow study captures assumed the wide "Candidates and your answer" landmark. Narrow
  layouts use "Candidates" and "Your answer" regions.
- `review-hierarchy.spec.ts:77` asserted that a fresh DOID side shows the top-level list, which
  HR4 deliberately replaces with the curated start. The test now asserts the curated start, then
  opens the list and keeps all its continuation, duplicate-free and one-side-focus assertions
  (`1b878dd`).

No retries were enabled and no assertion was weakened.

**Development-server iterations.** During development, `readiness-entry` ran against a
development exploration server (19485, now stopped) and exposed:
- keyboard focus lost in the source switch (fixed);
- 320 px / 200% overflow and clipping (fixed).

These iterations are not acceptance evidence.

**Rendered inspection.** The
[captures](explanation-readiness-frontend-corrections-20261013/screenshots/) were each taken
after settled content, an Axe pass and an overflow check, and were then inspected:
- the selector (desktop light; All sources dark; 375 px; 320 px at 200% text);
- the catalogue failure notice;
- the curated start and entry list (light/dark; 375 px; 320 px at 200%);
- the study case after a reload (desktop dark; 375 px light; 320 px at 200% dark).

**Known layout limits**, not regressions:
- At 320 px with 200% text, the hierarchy focus card's column is about 120 px wide. IRIs break
  per character and the label wraps across lines. The earlier handoff already noted this
  narrowness at that size.
- The study header stacks controls at 320 px / 200%.

## Bindings and identities

| Identity | Value |
|---|---|
| Final managed environment | `~/Library/Caches/exact-om-review/backend-integration-20261013/environment-frontend-fffa6d8` — PostgreSQL 19663, study HTTPS 19664, exploration HTTPS 19665 (loopback) |
| Service code | Clean detached worktree `~/Library/Caches/exact-om-dev/s2dev/source-fffa6d8`; backend fingerprint `831a0f01746b4a34fe2c63d4e60d6e47d6f76e26478a5e062f79ec13f1903588` (57 files), **equal to the reviewed backend** |
| Frontend export | `…/backend-integration-20261013/frontend-fffa6d8`, environment hash `a3e5b711d1c037d0ca3d0be601c70a331784d9526213aa0c7a6ba460fedcb0d4`. `src` tree `5ba9e338…` is identical at `fffa6d8`, `1b878dd` and `6dc6cbe`. |
| Package `package-definition-review-20261013.json` | `b89ae882e62ceb9ad810aa063b739429c08027fef26fe0ef0dc75ca0007bc18a` — unchanged |
| Source publication `publication-definition-review-20261013.json` | `3893f7607b229fbbd3957d382ad3d27063293a642e18f15ddce8605eddfe209f` — unchanged |
| Environment publication | `review-20261013T231535-27ac5d757d`, SHA-256 `831bac60ce48c5bdad257ae568c07feccc39e606aa7e9d0ae8e36d776a7fc6bd`, closes 2026-11-12 |
| HR1 investigation binding | `environment-final` (19653–19655): export `07bdf63e…`, revision `review-20261013T062053-62b43d70b7`. Only development test sessions were added; it is still running as found. |

**Changed paths** (`explanations_visualizer/`):
- **New code:** `src/components/explore/PreparedExamples.tsx`, `src/lib/reviewCatalogue.ts`.
- **Edited code:** `src/components/explore/{CompareView,HierarchyBrowser,SourceRail,BrowseView}.tsx`,
  `src/lib/types.ts`, `src/lib/workspace/navigation.ts`, `src/styles/app.css`.
- **New tests:** `src/lib/reviewCatalogue.test.ts`, `e2e/styleDiagnostics.ts`,
  `e2e/style-boundary.spec.ts`, `e2e/readiness-entry.spec.ts`, `e2e/readiness-visual.spec.ts`.
- **Edited tests:** `src/lib/workspace/navigation.test.ts`, `e2e/study-review-journey.spec.ts`,
  `e2e/v2Harness.ts`, `e2e/review-hierarchy.spec.ts`.

**Dirty-source disclosure.** Pre-existing uncommitted owner spec edits and untracked files were
left untouched and uncommitted. They include specs 20–24, other agents' review documents,
`AGENTS.md`, the storage guard and cluster tools, deploy scripts and tarballs. The services run
from the clean worktree, so none of these can reach them.

## Access, operation and rollback

From the active checkout (`PY=~/Library/Caches/exact-om-dev/pyenv/bin/python`,
`WT=~/Library/Caches/exact-om-dev/s2dev/source-fffa6d8`,
`E=~/Library/Caches/exact-om-review/backend-integration-20261013/environment-frontend-fffa6d8`),
with `PYTHONDONTWRITEBYTECODE=1`:

```sh
"$PY" -m tools.review_storage_budget check --growth-gb 1
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -m tools.review_environment status --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -m tools.review_environment stop --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -m tools.review_environment fresh-session --root "$E"
```

**Main app:** `https://127.0.0.1:19665/` opens a prepared example directly. **Study:**
`fresh-session` writes a private test invitation under `$E/invitations/` (mode 0700). Never
print or commit it, `environment.json`, keys, cookies or browser storage. These are development
sessions, not research data. Browser trust in the local certificate remains a local setup
matter.

All five commands were exercised in this pass. The detached services have a recorded 1 GB
reservation and a 128 MiB per-file limit.

**Rollback:**
- The reviewed `environment-final` (19653–19655, export `07bdf63e`) is untouched and running.
- The earlier `environment-frontend-bb9a1a6` and `environment-frontend-2d4600d` are also still
  running.
- Reverting `bbedb3c..6dc6cbe` restores the reviewed frontend source; no data migration is
  involved.

## Backend dependencies

1. **HR1 static delivery (§3.4)** — verify GET/HEAD, completion, length/hash, interruption and
   missing-asset handling against this export. Decide whether `/_next/static/` assets should
   keep the `immutable` caching that `frontend.py` sets, instead of the `no-store` the service
   middleware imposes:
   - **Observed:** every reload re-transfers the stylesheet, so any transfer failure becomes an
     unstyled page.
   - **Proposal:** apply `no-store` to HTML and API responses only, and keep content-hashed
     assets cacheable.
   - **Constraints:** keep the profile allowlists and the strict study CSP.
   - **Tests:** header assertions for an asset, an HTML page and an API response on both
     profiles.

   This is not shown to cause HR1.
2. **HR1 server-side observability** — if a recurrence shows a failed transfer, add the opt-in,
   bounded static-read observation of §3.4. Correlate an allowlisted asset digest or request ID;
   do not buffer, retry or change `Content-Length`.
3. **HR3** — locked Black/isort and the CI checks. The backend fingerprint did not change in
   this pass, so a formatted backend needs a new managed binding.
4. **R3 integration** — rerun the matrix above against the final backend identity. If the
   backend changes nothing that the frontend consumes, no frontend code change is expected.

## Storage

| Point | Combined use (decimal) |
|---|---|
| Start of this pass (13 Oct, 22:25 UTC) | 51,831,623,194 bytes (51.832 GB) |
| Final (13 Oct, 23:45 UTC) | **51,796,822,266 bytes (51.797 GB)**, 116.7 GB filesystem free |

This pass added:
- the managed environment, 119 MB including its test sessions;
- the export, 2.4 MB;
- the clean worktree, 56 MB;
- test output and diagnostics, 29 MB;
- the repository evidence, 2.2 MB.

The other sessions' activity during the interval explains why the net total is slightly lower.
These are measured points, not an instantaneous peak.

Every build, copy, worktree, environment preparation and browser batch ran under
`tools.review_storage_budget`. No dependency, dataset, bundle, publication, generation, matcher
or reasoner run was created, and no protected inode was touched.

## Readiness statements

- **Frontend implementation (spec 24 R1):** complete for HR2 and HR4. HR1 has its investigation,
  instrumentation and regressions in place, but its cause is unresolved.
- **Integrated local review:** usable for owner exploration on 19663–19665. **Not a clean
  combined handoff:** HR1 is unresolved and HR3 is open (backend).
- **Full CI:** not claimed. The frontend checks pass; backend formatting, flake8, mypy and
  import boundaries were not run here.
- **Research, deployment and launch:** not ready and not assessed. Docker, deployment/restore,
  manual accessibility, pilot, owner review and final admission remain open, as do historical
  R16 and the profile-read incident, which are not linked to HR1 by any evidence here.

Two issues found in this pass were introduced and fixed within it:
- the keyboard focus on the source switch;
- the 320 px / 200% overflow and clipping of the new controls.

No inherited issue was relabelled.
