# Product-review frontend pass S0 — handoff, 2026-10-09

**Status: S0 is implemented and verified against the existing (unchanged) backend and synthetic
data. This is not a claim of usability, realistic-data quality or study readiness.** S0 is the
frontend-first pass of [20](../../specs/explanation-framework/20-review-corrective-programme.md):
- F01–F04 and F09;
- the compatible F08 graph work;
- F02's guided layout, using the existing tutorial semantics;
- an inventory of every backend dependency.

The [review evidence](../../specs/explanation-framework/evidence/product-review-20261007.md)
supplied the reproduction targets. No backend, contract, tutorial content or frozen-form
semantics changed.

The next step is backend S1 (B01–B07 in [22](../../specs/explanation-framework/22-review-backend-and-data.md)),
then frontend S2 (F05–F07 and the data-dependent parts of F02, F06 and F08).

> Specs 20–22 and the review evidence were uncommitted, owner-authored files in this checkout
> during S0. They were read but not modified or committed by this pass.

## Commits

Base: `70c2c73`. This is a merge of `origin/dev` that landed in this checkout during the session.
It changed matcher-experiment tooling only, with no frontend, `exact_inspect` or study files.
The session started at `092c7fe`. Unrelated commits by others (`647c027`, `exact-repair` specs)
landed on top while S0 was being committed; they were left untouched.

| Commit | Content |
|---|---|
| `1b7ca13` | F01: valid-only tutorial evidence, scoped feature resolution, inspector intent, `LessonEvidence` store, `DiscardedMutation` |
| `8e3f45b` | F02 / F01 UI: guided task bar with Show me, Why?, step chips, in-place 422 recovery, one question at a time |
| `17f60a3` | F04 / F09: short state-specific copy, Browse scope disclosure, Sources and preparation, calmer dark tokens |
| `2874e62` | F08: graph model and layout, recorded-only comparison lines, typed identity, inspector and keyboard item list |
| `f9cdfa6` | F03, plus F02 lesson 2 and F06 wording: reveal-and-focus, Back to compared concept, mode labels, Exact matching trace |
| `762cde0` | Tests: S0 regressions on the real service, updated suites, review screenshot capture |
| `9a2beda` | Test fix: the v1 typed-parent regression (R12) no longer depends on which case is assigned first. One fixture source has no recorded parents, which made it intermittently fail. |
| this handoff | the following documentation commit, with the screenshots and receipt |

Each intermediate commit type-checks on its own.

## The failing path, reproduced and then corrected

| | Against the unchanged API and build (`092c7fe`) | After S0 |
|---|---|---|
| **U01:** bridge selected in the graph lesson | `PUT /tutorial/progress` → **422** *"Evidence action requires an admitted original fact identity"*. The actions were `graph.inspect` and `graph.view`, both without `fact_id`. | Bridges and context lines are inspection only: no request, no tick. A feature saves the inspected candidate's own record (200). |
| **U02:** then a valid different action, then Help | Both rejected with **422**, because the fact-less entries were resent every time. The checklist showed "Done · saving" ticks, the server recorded `[]`, and the recovery and error banners stayed. | A rejection drops exactly the evidence it carried. The bar names the step; the next valid action, Help, ranking, Next lesson and reload all save (200). No later request carries the rejected record. |
| **U03:** lesson 2, child reaches the compared concept | The client reported `navigate_child` only. The Return button then disappeared, so the instructed return was impossible on that path. | The real transition is reported as `return_to_compared` with the compared typed entity, and the server accepts it. "Back to compared concept" appears whenever focus is away. |
| **U05:** Open decision trace | Set a URL parameter only; nothing visible changed. | Selects the tab, scrolls the trace below the header and focuses its heading, on every click. |

The baseline reproduction ran in a real browser against the real v2 study service: HTTPS,
PostgreSQL 16.2, compiled export and normal CSP, with the session API-seeded up to the tutorial.
It was a scratch-only diagnostic, not committed.

The post-fix evidence is the committed suite below. The one injected fault (Q02) rewrites a
request body so that the real server rejects it; it is labelled as a diagnostic.

## Requirement disposition

| ID | Disposition in S0 | Evidence |
|---|---|---|
| **F01** (U01/U02) | **Done** within the existing contract. See the item list after this table. | Unit: 7 new cases covering bridge, expanded edge, scoped feature, another candidate, missing or unresolved records, citation versus record, return via child, and store invariants. Browser: Q01 and Q02 below. |
| **F02** (U03/U04) | **Done for the existing tutorial semantics.** See the item list after this table. A new stored `step_id` is a B01 option, not used. | Browser: Show me, Q03, the journey, R03 position tests. |
| **F03** (U05/U07) | **Done.** See the item list after this table. | Unit: 5 trace cases. Browser: Q04 at 1280×720, 1440×900 and 375×812, 200% text and reduced motion. |
| **F04** (U06) | **Done for frontend-owned copy.** See the copy inventory. Backend-authored claim text, tutorial and form wording were not trimmed; they belong to B04 and B06. | Screens; suite. |
| **F05** (U08) | **Not started (S2).** The S0 copy changes only remove the false completeness claim and move provenance into a disclosure. Prose versus "Available facts" needs B04's preparation kind. | — |
| **F06** (U10–U12) | **Wording only.** See the item list after this table. Entry points, the inferred view and "What changed?" need B03. | Screens. |
| **F07** (U13/U14) | **Not started (S2), deliberately.** Frozen `/1` and `/2` forms keep their semantics. The practice reports still use the current seven-method list until B06 publishes forms/3, setup/3 and consultation/3. | — |
| **F08** (U15) | **Semantics and design done.** See the item list after this table. | Unit: 6 model and layout cases. Browser: Q01, journey, screenshots. |
| **F09** (U16) | **Done for the shared tokens and compositions.** See the item list after this table. | Screens; accessibility suite (axe, 320–2560 px, 200% text). |

**F01 detail:**
- Interactions are separated from evidence.
- Features are resolved against the inspected candidate's evidence, using the actual inspected record.
- Inspector intent separates citation from record.
- Pending work is computed from valid queued evidence; ticks appear only after acknowledgement.
- Acknowledged evidence is never resent or overwritten.
- After a 422, the evidence that request carried is dropped and still-valid evidence is resent.
- After a 409, the session refreshes and only still-missing evidence is resent.
- Offline and 5xx failures retry with the same key and the same bytes.

**F02 detail:**
- One sticky task bar beside the workspace shows the next unsaved requirement, one or two lines of help, its status, Show me, Why?, step chips and Back/Next.
- Show me opens the right tab, then reveals, focuses and outlines a real control. A missing control gets a specific message, and Show me never performs the step.
- The lesson stepper is one row.
- The five questions are shown one at a time with Previous/Next and durable drafts.
- Lesson 2 keeps the compared concept visible in each hierarchy panel, with "Back to compared concept" or "Viewing compared concept".
- A real return transition counts as the return; page load does not.
- Lesson 1's "Return to a candidate" targets the Inspect control and is worded separately.

**F03 detail:**
- The shared reveal-and-focus operation covers Open decision trace, card parent → hierarchy, and Show in graph / Show in evidence list.
- Async data and autosaves never scroll.
- The trace opens with a summary, then the recorded sequence.
- `prefilter` reads as "Exact matching", and the five producer codes map one to one.
- Runs of unrecorded or skipped stages collapse; codes appear under Technical details.
- An exact route is never inferred from retained events or scores.

**F06 detail:**
- The modes are labelled Asserted / Structural / Inferred, each with one short meaning and its directness.
- An unprepared mode explains itself instead of showing an empty tree.

**F08 detail:**
- A comparison line is drawn only when both sides recorded the channel. A one-sided channel is named and gets no line.
- Node identity is side, ontology version, kind and IRI. Literal records are scoped by record.
- Every feature line keeps its evidence and record IDs. Records with no graph form are counted.
- Quantifiers and cardinalities are kept.
- Layout: column-based, collision-free and width-fitted, with relations written inside nodes and long literal lists grouped.
- Below the canvas: an inspector (original record, show in list, add parents) and a complete keyboard item list.
- The view persists across ranks, autosaves, tabs and themes.

**F09 detail:**
- Muted dark source, target and bridge accents, with lower surface steps.
- Empty states are plain text.
- Cards, focus cards and comparison groups use neutral surfaces with small side accents instead of tinted fills.
- Ordinary text stays at or above 4.5:1.

**Design choices, for the owner's review:**
- **Task bar:** it sits above both columns rather than inside the rail, so the task and the target are visible together at 1280×720.
- **Lesson titles:** hidden visually in the stepper (still in names and tooltips), since the heading names the current lesson.
- **Graph labels:** relations go inside nodes rather than on lines. This was chosen over edge labels, which overlapped in fans.
- **Graph side panel:** the inspector and list go below the canvas, chosen over a side panel that made the canvas too narrow to read at a legible zoom.

## Copy inventory (frontend-owned strings changed)

| Where | Before | After |
|---|---|---|
| Empty category | "No definition is stated for this entity in the loaded scope (root document; imports not loaded). This does not mean one exists nowhere." | "No definition in this ontology." With a partial scope: "Definition not included in this view." |
| Empty parents | "No named parent is stated … does not mean one exists nowhere." | "No parent shown." |
| Browse scope | "Root document only: imports were not loaded …", on every root load | A notice only for unresolved declared imports; "About this ontology" disclosure otherwise |
| Unprepared description | "… The original facts above are complete for the loaded scope." | "No generated description was prepared." |
| Generated header | "Cites N facts · grounding validated", plus a category badge per claim | "Generated" tag; claims; one "Sources and preparation" disclosure (model, checks, limits) |
| Comparison | Four always-present tinted groups, including empty ones | Only groups with claims; "Nothing recorded for: …" once |
| Matcher bar | "Matcher record … matching score 0.90 · Not a probability of being correct." | "Matcher · Selected · in saved alignment · matching score 0.90" (the meaning on request) |
| Trace | "Prefilter"; codes as prose; "The saved run did not keep a record of this stage …" | "Exact matching"; mapped meanings; "No record in this run" collapsed |
| Evidence list | "N features Exact selected when scoring this pair … Each rests on …"; a weights caveat on each item | "N features Exact used for this pair"; caveat once when weights are shown |
| Hierarchy | "Return to X" inside the focus card; "Reasoner inferred · not prepared" | "Back to compared concept" next to search; Asserted / Structural / Inferred with meanings |
| Tutorial | Intro, all steps, a tall checklist, a dashed synthetic banner and a "Practice · not scored · synthetic" pill | Intro on lesson 1; one task; steps under Why?; one "Practice · not scored · invented examples" pill |

Not changed (backend-owned): generated claim text and comparison templates (B04); tutorial
lesson and requirement wording, intro and assessment (B01/B06); questionnaire wording (B06).

## Tests executed

These ran on 2026-10-09 against the S0 review environment described below, on the build of
the committed frontend tree (`9a2beda`; the frontend sources are unchanged since `762cde0`).

| Check | Result |
|---|---|
| Type check; each S0 commit type-checked on its own | pass |
| Unit (`npm run test:unit`) | **93 passed**, 18 of them new: 7 F01, 5 trace, 6 graph model; the outbox test was extended |
| Production build (Next 15.5.26 static export) | pass |
| Browser, complete suite | **98 passed, 0 failed, 9 skipped**. The skips are the 7 opt-in screenshot captures and 2 legacy-viewer tests (historical run directory unavailable). It includes the fresh, unseeded participant journey, every R01–R13 regression, and the accessibility, exploration, library, demo and v1 suites. |
| S0 regressions (`product-review-s0.spec.ts`) | 9 passed (included above) |

**Q coverage in S0:**
- **Q01:** one-sided channels draw no comparison line. An added context line saves nothing. A feature saves the inspected candidate's own record (second candidate). The bridge-first, expanded-parent-first and list-alternative orders are unit-tested, because the served practice data has no two-sided channel.
- **Q02:**
  - Forced 422, then: a valid action, Help, a ranking change, Next lesson, reload. Offline evidence shows "Saving…" without a tick.
  - The two-tab conflict is covered by the existing R03 navigation test, not with lesson evidence.
- **Q03:** search, then the compared child, then the return is acknowledged; then a parent; the target side's Back. Restored progress and the remembered candidate focus are not separately tested.
- **Q04:** trace, card parent and Show in graph at three viewports, 200% text and reduced motion. "No focus jump on autosave" holds by construction (only explicit actions scroll) and is not separately asserted.
- **Q09:** semantics and layout are unit-tested. Rendered review is sparse only, pending B02.
- **Q10:** light and dark screenshots on synthetic data; automated axe, reflow and keyboard suites pass.
- **Q11:** the fresh unseeded journey now includes free exploration of the optional graph lesson and answers the questions one at a time. The deliberate rejected action is in Q02, not in the journey.
- **Q05–Q08 and Q12:** depend on S1 (B02–B07).

## Screenshots

The [receipt](explanation-product-review-frontend-s0-receipt.json) lists every browser test result. The screenshot files are in [explanation-product-review-frontend-s0/](explanation-product-review-frontend-s0/):
`before-*` were taken on the build of `092c7fe`, `after-*` on the build of the committed tree.
All were captured by `e2e/review-screenshots.spec.ts` at 1440×900 (narrow: 375×812).

**Data used:**
- **Main app:** the synthetic UI fixture (`tools.build_explanation_ui_fixture`): 3 NCIT sources, a synthetic matcher run `synthetic-ui-fixture`, real NCIT/DOID excerpts. Package `package.json` SHA-256 `3689e862…2609`.
- **Study and tutorial:** the native synthetic v2 fixture (`tools.build_explanation_v2_fixture --navigation-size 65 --paged-facts 26`, 59 files). `publication.json` SHA-256 `04db8d2b…085a`; revision `synthetic-v2-e040b72a…`, tutorial `tutorial/2.0-native-2-nav65`.

This is not realistic review data: the graphs are sparse and the descriptions are offline extracts.

| View | Before | After |
|---|---|---|
| Main compare | `before-main-compare-{light,dark}` | `after-main-compare-{light,dark}` |
| Open decision trace (viewport after the click) | `before-main-trace-after-click-light` | `after-main-trace-after-click-light`, `after-main-trace-{light,dark}` |
| Evidence graph | `before-main-graph-{light,dark}` | `after-main-graph-{light,dark}`, `after-main-graph-selected-{light,dark}` |
| Tutorial lesson 1 | `before-tutorial-lesson1-{light,dark}` | `after-tutorial-lesson1-{light,dark}`, `after-tutorial-lesson1-showme-dark` |
| Tutorial lesson 2 | `before-tutorial-lesson2-hierarchy-light` | `after-tutorial-lesson2-hierarchy-light`, `after-tutorial-lesson2-away-light` |
| Tutorial graph | `before-tutorial-graph-light` (the bridge drawn from a one-sided channel) | `after-tutorial-graph-{light,dark}` |
| Scored explanation case | `before-study-case-dark` | `after-study-case-{light,dark}` |
| Narrow tutorial | `before-tutorial-lesson1-narrow-light` | `after-tutorial-lesson1-narrow-light` |
| Browse | `before-main-browse-light` | `after-main-browse-light` (still the empty entry state: U11, B03) |

## Backend dependencies for S1 (from the S0 inventory)

| Need | Frontend use (S2) | Backend item | Notes from S0 |
|---|---|---|---|
| Structured 422 detail (error code, affected requirement or action IDs) | Name the exact rejected step. Today S0 drops all the evidence the rejected request carried, which is safe but coarser. | B01 | Keep the short `detail` for old clients. |
| Reachability-validated tutorial publication; optional `step_id` | Restore the exact guided step; use instructions guaranteed to have targets | B01 | S0 derives the task from the next unsaved requirement; no stored step. |
| Publication-owned task hints per requirement | Replace the frontend-authored `TASK_HINTS` (proposed wording is in `LessonCoach.tsx`) | B01 content (frontend proposal) | The hints are presentation only today and keyed by action. |
| Richer practice evidence | A graph lesson with a two-sided channel, a hierarchy feature on both sides, and correct channel names | B01 / B02 | Served fixture: 5 items, all on the target side, all `lexical`, including restriction records. No comparison line is possible, and "lexical" is attached to restrictions. |
| Realistic review bundle (12, then 24 sources) | Q05, realistic Q09/Q10 screenshots, dense graphs (50+ features), description review | B02 | S0 screenshots are synthetic and sparse. |
| `/entry-points` (exploration and study) plus inferred adjacency, with capability and contract version | F06: initial focus, top-level list, an Inferred mode with directness and "What changed?" counts | B03 | S0 only labels the modes; an unprepared Inferred mode explains itself. |
| Description preparation kind and status, supporting facts | F05: prose versus "Available facts"; a distinct status for unprepared, rejected, failed or insufficient | B04 | S0 removed the false completeness claim only. |
| Shorter comparison and missingness templates (new template version) | Less repeated caveat text inside claims | B04 | The frontend never edits claim text. |
| Real exact-route trace examples (protected, removed, retained, disabled, unrecorded) | Verify the F03 mapping on real bytes | B05 | The mapping is unit-tested against producer codes only. |
| Exported comparison metadata for bridges (channel compared for this pair, or a recorded correspondence) | Replace the client rule "recorded on both sides" with an exported fact; show correspondences if recorded | B05 | S0's rule is a conservative channel-level reading of recorded evidence. |
| Distinct evidence statuses (none selected, not exported, unresolved, withheld, unsupported projection) | Graph and list empty states | B05 | |
| forms/3, setup/3, consultation/3 with server-ordered options, experience bands and Other text | F07 everywhere, including the practice reports | B06 | S0 did not change frozen questionnaire semantics. |
| Persistent review start/stop and storage | Replace this pass's cache-directory environment | B07 | See the environment below. |

## Preserved safeguards

The following are unchanged:
- page gating;
- study CSP: no inline styles or `unsafe-inline`, and Cytoscape's stylesheet stays reserved;
- baseline isolation: the baseline lesson and cases still show no explanations;
- typed facts and identities;
- readiness timing and `case_ready` behaviour;
- the v1 resource adapter;
- server-owned completion, unlimited retries and the five server-graded items;
- the outbox's idempotency: retried bytes never change under a sent key.

## Environment (S0 review, reproducible)

The previous `/tmp` toolchain had been partly removed by the system's temporary-file cleaner, so S0
used a durable cache directory outside iCloud: `~/Library/Caches/exact-om-dev/`.
- Node 24.21.0, checksum-verified from nodejs.org.
- Locked frontend dependencies (`npm ci`).
- Python 3.12 venv with `deploy/render/exact_inspect_prepared_requirements.txt`, the study requirements, `pyowl-core` 0.2.1, and `pgserver`, which provides PostgreSQL 16.2 binaries.
- A frontend build copy synced from the repository before every build (`fe-sync.sh`).

Under `s0/` are the fixtures, a dedicated PostgreSQL cluster (port 19083), loopback TLS (self-signed), `restart.sh` and `run-suite.sh`:
- real v2 study service: `https://127.0.0.1:19084`;
- main app on the UI fixture: 19065;
- main app on the v2 contexts: 19085 (study policy) and 19087 (paging);
- demo: 19067;
- local fixture: 19068;
- v1 study harness: 19066.

Services must be restarted after each build, because the CSP hashes are computed at start.
`s0/new-invite.sh` issues one fresh test invitation and prints its private link (loopback only; the
certificate is self-signed). This is an agent environment, not B07's persistent review workflow.

## Remaining issues and limitations

- S2 work as listed above (F05–F07; F02/F06/F08 data-dependent parts). S0 does not close U08–U14, U17 or U18's realistic-data clauses.
- The graph is reviewed only on sparse synthetic data. Dense (50+), restriction-heavy and long-label review waits for B02.
- A two-tab conflict combined with lesson evidence is not separately browser-tested. After a 409, still-missing evidence is resent; the practice answer is not.
- On a 422, S0 drops all the evidence the rejected request carried, not only the invalid item, because the server does not identify it yet (B01). The participant can redo any valid step.
- The tutorial task hints are frontend-authored until the publication owns them (B01 proposal).
- Human-only follow-ups, recorded separately and not claimed: a screen-reader and keyboard audit of the task bar, Show me, the graph item list and the trace (the old J10 audit remains open); and the owner's visual and usability review.
- Still open and unchanged: the R16 intermittent-reload investigation, real-package/import admission, deployment/restore and the final study-owner gates.
- Not done: no push, deployment, recruitment or participant contact.
