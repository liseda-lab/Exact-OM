# Product-review frontend S0 follow-up — handoff, 2026-10-10

**Status: the five findings of the [independent S0 review](explanation-product-review-frontend-s0-review.md)
are corrected and verified against the existing (unchanged) backend, using synthetic data only.
This is not a claim of usability, realistic-data quality, accessibility conformance or study
readiness.** Owner visual approval, manual accessibility audits, real-package and deployment
gates remain open.

| Finding | Disposition | Main evidence |
|---|---|---|
| **S0-R1** shared graph targets borrowed the first relation | **Corrected.** Each feature line carries its own predicate and qualifier. Entity nodes show only the entity. | Browser test across graph, list, inspector and original record, before and after selection, tab and theme changes. Shared-target screenshots. |
| **S0-R2** assessment navigation lost the visible pending answer | **Corrected.** A session-scoped draft store sits above the keyed question cards. | 5 unit cases. 4 browser tests on the real study service: offline/reconnect, immediate Next/Back with a delayed save, reload with queued work, competing tabs. |
| **S0-R3** Show in graph could reveal an unrelated previous selection | **Corrected.** Selection resolves through every record of the feature, including grouped and undrawable ones, and clears unrelated focus. | Browser test over every grouped record, after a previous selection and a tab restore. Also covers multi-record, unsupported and unavailable features. |
| **S0-R4** selection faded labels below 4.5:1 | **Corrected.** De-emphasis fades geometry only. Every label stays at ≥ 5.55:1 (light) and ≥ 6.37:1 (dark). The view is kept across text-size changes. | 4 contrast tests on Cytoscape's resolved styles (light/dark × 100/200% text). A text-size view test. Colour calculations. Screenshots. |
| **S0-C1** repeated score caveats shown by default | **Corrected.** A concise "Matching score" label. Meanings, stage and calibration are behind keyboard/touch disclosures. | Browser test of the trace and Scores panel, including the keyboard. Screenshots. |

No backend code, contract, validation, tutorial content or frozen form changed. Specs 20–22,
the owner's spec edits and the reviewer's evidence directory were read but not modified or
committed.

## Commits

| Commit | Content |
|---|---|
| `5185f3e` | S0-R1, S0-R3, S0-R4: graph model records, relation-labelled lines (local names for unlabelled properties, label-aware spacing), record-exact selection, readable de-emphasis, text-scaled layout and view preservation |
| `848cc53` | S0-R2: `AssessmentDrafts` store, outbox `queued()`, bounded re-save after a conflict or discarded write |
| `7c1fd00` | S0-C1: `Disclosure` / `ScoreMeaning`, concise score labels in the trace, Scores panel and ranking list |
| `d293657` | Regressions (`product-review-s0-followup.spec.ts`), synthetic graph fixtures (`graphFixtures.ts`), extended review screenshot capture |
| this handoff | the following documentation commit, with the screenshots, contrast calculation and receipt |

## Reproduction before correction

The failures were reproduced on builds of the committed S0 tree (`f80bfcf`; its frontend
equals `1fb1cee`).

1. **The review's three diagnostics**, unchanged except for their screenshot directory: all three
   reproduced their defects (offline draft blank after Next/Previous; stale hierarchy inspector
   after Show in graph on a grouped annotation; the selected *only* line ending at a node
   labelled *some*).
2. **The new expected-behaviour regressions**: these fail on the pre-fix build for the reason each finding describes.

| Test | Pre-fix result |
|---|---|
| S0-R1 shared targets agree | Fails: no node labelled with the shared entity alone (the relation was written into the node). |
| S0-R1 unlabelled property, no label overlap | Fails: the line draws no relation (empty label). |
| S0-R3 every grouped record | Fails at the first grouped record. After Show in graph on *gamma synonym*, the inspector still describes the previous "subclass of Parent P". |
| S0-R4 contrast × 4 | Fails: the opposite compared concept had opacity 0.35. |
| S0-R4 text-size view | Fails: graph text stayed 15 px at every application text size (the graph ignored the setting). |
| S0-C1 trace and Scores panel | Fails: no "Matching score" label. |
| S0-R2 offline Next/Previous | Fails: the answer is unchecked on return. |
| S0-R2 immediate Next/Back, multi-part | Fails: the first row's choice is unchecked on return. |
| S0-R2 reload with queued work | Fails: the queued answer is not shown. |
| S0-R2 competing tabs | Fails: the stale tab's conflicting draft is never saved. |
| Dense columns, long labels | Passes on both builds: node layout was not defective. It guards the new label spacing. |

The main-app tests ran against the pre-fix build served on a spare port, from an archive of
`HEAD` plus the new test files. The R2 tests ran against the real study service before any fix.

Two early pre-fix runs of the R3 test stopped on test-harness problems before reaching the
defect, so they are not counted:
- The first used a malformed synthetic record, an `EquivalentClasses` without its `expressions` list, which made the page throw. The fixture was corrected, and `owl.ts` now tolerates the missing list.
- The second used a list attribute that this pass added. The test now finds the item by its text.

The result in the table comes from the corrected test.

3. **On the corrected build, the review's three diagnostics no longer show their defects.**
   - Offline-draft diagnostic: fails at its defect assertion, because the answer is still checked after Next/Previous.
   - Grouped-record diagnostic: fails because the inspector shows the requested literal instead of the previous hierarchy feature.
   - Shared-filler diagnostic: it only captures the canvas. Its capture now shows two separately labelled lines, "holds only" (selected) and "holds some", into one *Syndrome* node.

### Found during rendered inspection in this pass

The review asked for the rendered results to be inspected. Doing so found four further problems, fixed in this pass:
- **Unlabelled properties drew their full IRI on the line.** This was found by running the
  review's own R1 diagnostic on the first corrected build. Its property has no label, so each
  line showed "https://example.org/review#holds only". Two such parallel labels overlapped each
  other and the *Syndrome* node. Now:
  - a line draws an unlabelled property outside a known prefix by its local name ("holds only"), while the list, inspector and original record keep the full IRI;
  - a long property name is clipped on the line, but its qualifier never is;
  - parallel lines are spaced from their angle, so their labels stack vertically, 24 graph units apart per text line;
  - the column layout reserves vertical room for the labels on each node's incoming lines, so labels of neighbouring nodes cannot collide;
  - choosing a line from the graph's own list pans it into view if it is not (zoom kept), as Show in graph already did. Tapping the canvas never moves the view.
- **Text-size change could push the selection off the canvas.** Increasing text to 200% doubled
  the layout but kept the old pan, so the selected line's far end and the compared pair went
  off screen. See `before-graph-text200-selected-light.png`, where the visible pair sits under
  the sticky header and *Syndrome* is off the canvas. Spec 21 requires pan, zoom and selection to
  be preserved across text-size changes. A text-size or canvas-size change now keeps the zoom
  and the view's centre (scaled), and pans only to keep the selected line in view, with both
  compared concepts too whenever they fit at that zoom.
- **The start view centred the compared pair even when the whole graph fits.** One deeper side
  could then be cut, as with the tutorial's *round item*. The start view now centres the whole
  width when it fits; otherwise it still centres the pair at a readable zoom.
- **"None shown." contradicted a "record not drawn" entry** in the same column, and the
  record status in the inspector could wrap as a stray "· …" line. Both were reworded.

The element screenshots also had the sticky header across the graph, because tall elements are
captured while scrolled. Only the capture script changed: the header is made static there.

## Corrections

### S0-R1 · relationships sharing a target keep their own meaning

- `evidenceGraphModel.ts` keeps typed entity reuse: one node per typed entity.
- Every feature line is labelled with its own relation, read from its own record, e.g. *holds some*, *holds only*, *has part at least 2*, *located in exactly 1*, *subclass of*.
- Entity nodes show only the entity name.
- Literal (annotation) nodes are fact-scoped and show their own predicate and quoted value.
- Lines into literal nodes and the group node carry no duplicate label.
- The inspector, the keyboard item list and the original-record dialog all read the same edge. Selecting a different line never inherits another's meaning.
- An unlabelled property is drawn on its line by its local name, and the list and inspector give the full IRI. A clipped property name always keeps its qualifier.
- Parallel lines and the column layout leave room for every label, so labels do not overlap each other or a node.
- Tested with:
  - *some* and *only* on one property, one target;
  - a qualified cardinality on another property, same target;
  - an exact cardinality on a third property, another target;
  - a subclass edge to the same target.
- Each is checked in the drawn Cytoscape state (the line's own resolved label), the item list, the inspector and the original record, before and after selection, a tab round trip and a theme switch.
- A second test uses the review's own shape: *some*/*only* on an unlabelled property to one filler. It adds two long, wrapping property names on parallel lines to a second filler, and a name long enough to be clipped. It checks:
  - the drawn local names;
  - the kept qualifier;
  - that no label box overlaps another label or any node;
  - the full IRI in the list and inspector;
  - that a line chosen from the list is fully in view.

### S0-R2 · visible assessment drafts survive navigation and recovery

The assessment card is still keyed per question, but its answer now lives in
`AssessmentDrafts` (`src/study/assessmentDrafts.ts`), a session-scoped store in `Tutorial`.

**Pending edits.** An edit is shown at once and stays *pending*. It is acknowledged only when
the server's draft equals it (canonical JSON, key-order independent) and no save for that
question is still queued, in flight or being replaced. With nothing pending, the server's
draft wins, for example when another tab or a replay changed it.

**Reload.** After a reload, unsent drafts in the outbox (`MutationQueue.queued()`) seed the
store, so queued work is visible before it is acknowledged.

**Conflicts.** After a 409 conflict or a `DiscardedMutation`, the visible draft is saved again
on top of the refreshed state. This happens at most three times per question, so two tabs
cannot loop. The tab still adopts the server's position after a conflict (R03 behaviour,
unchanged). The competing-tabs test therefore navigates back to its question and asserts that
both tabs' drafts are visible and saved.

**"Changed since your last check".** This is now derived from the latest graded attempt rather
than component state, so it survives remounts.

**Unchanged:**
- coalescing (`tutorial:item:<question>`) and the 500 ms debounce;
- server grading and attempts;
- Back/Next navigation;
- outbox idempotency: a re-save is a new write after the previous one settled, and sent bytes never change under a key.

### S0-R3 · Show in graph reveals the exact requested record

The model records every selected record with its status: drawn, grouped (behind "+N more
records"), no graph form (an unsupported shape) or unavailable (no resolved original fact).
A selection from the evidence list resolves through those records:
- **Grouped:** shows that side's group and focuses the record's own line.
- **Drawn:** focuses its line and pans only if it is off screen (the explicit reveal).
- **Several records:** focuses the first drawn one. The inspector lists every record with its status, a link to each drawn one, and Original record for undrawn ones.
- **Not drawn:** clears any previous line or node focus. The inspector says "This feature is not drawn in the graph.", lists each record and why, and offers Original record and Show in evidence list.

The graph's item list also shows undrawable features ("record not drawn"), so it stays at parity with the evidence list.

Tested:
- each of six grouped annotation records, each after a different previous selection;
- a tab restore;
- a two-record feature;
- an unsupported (EquivalentClasses) record;
- a feature with no resolved facts.

The tests assert selected evidence and fact identity, inspector contents and visibility.

### S0-R4 · labels stay readable during selection; the view is kept

- Selection no longer sets element opacity.
- Faded nodes keep their text and lower their border opacity (0.4). Faded focal nodes keep `--ink` text with a 0.55 border.
- Faded lines lower their line opacity (0.3), and their labels use `--ink-2`.
- Relation labels sit on an opaque `--surface` text background.
- The selected line's label is `--ink` at weight 600.
- Graph font sizes and layout distances follow the application's text size. Before this pass the graph ignored that setting: its text stayed 15 px at 200%.
- The view-preservation correction above keeps the zoom and selection when the text size changes.

Colour calculations ([contrast.json](explanation-product-review-frontend-s0-followup/contrast.json),
from the `base.css` tokens; the method is the review's):

| Label | Light | Dark |
|---|---|---|
| Before: faded focal label (`--ink` at 0.35) | 2.10 ✗ | 2.81 ✗ |
| Before: faded feature label (`--ink` at 0.35) | 2.11 ✗ | 2.82 ✗ |
| Focal label, any state | 14.68 | 12.76 |
| Entity or literal label | 15.57 | 13.51 |
| Faded entity or literal label (`--ink-2`) | 7.65 | 8.70 |
| Relation label on a line (`--ink-2` on `--surface`) | 7.65 | 8.70 |
| Selected relation label | 15.57 | 13.51 |
| Group count label (`--ink-3`) | 5.55 | 6.37 |

The browser tests check the same thing on the rendered graph: Cytoscape's resolved `color`,
`opacity`, `text-opacity` and label background for every labelled element, with a feature
selected, in both themes at 100% and 200% text. These are style computations, not pixel
measurements, and a DOM accessibility scan does not cover canvas text.

### S0-C1 · score caveats on request

- **Trace rows:** show **Matching score 0.90** (the recorded `S_final`) and a **What this score means** disclosure.
- **Scores panel:** puts its general caveat under **About these scores**. Each score has a concise label, the raw name as metadata, and a disclosure with its meaning, stage and calibration.
- **Ranking list footer:** uses **What is a matching score?**.
- **Disclosures:** real buttons with `aria-expanded`/`aria-controls` (`src/components/common/Disclosure.tsx`), never hover-only.
- The recorded meanings are unchanged and the onboarding explanation is untouched.

## Richer synthetic graphs

`e2e/graphFixtures.ts` overrides only the main app's `pair-evidence`, `/api/v1/axioms/*` and
`/api/v1/labels` read responses, inside isolated test browsers. The compared entities, their
ontologies and the served app are real. Every invented record uses an `https://example.org/review#` IRI
and an obviously synthetic fact ID. Nothing is written, and none of it is an ontology finding or
a matcher result.

| Shape | Content | Used by |
|---|---|---|
| Shared targets | *some*/*only*/min-cardinality/subclass to one filler; exact cardinality and subclass to another | R1 test; `graph-shared-*` |
| Grouped | 6 synonym annotations (4 grouped by default), a parent, an unsupported target record | R3 test; `graph-grouped*`, `graph-undrawn-*` |
| Multi-record / unavailable | a two-fact feature; a feature with no resolved facts | R3 test |
| Dense | 56 source features (every fourth with an 80-character label), 10 target features including 4 parallel *holds only* lines to one filler | dense layout test; `graph-dense-*` |
| Unlabelled and long relation names | the review's *some*/*only* on an unlabelled property; two wrapping property names on parallel lines; one name long enough to be clipped | R1 label test; `graph-labels-*` |

## Tests executed

These ran on 2026-10-10 against the S0 review environment (below), on the build of the final
working tree. That tree is identical to the committed frontend sources.

| Check | Result |
|---|---|
| Type check; each follow-up commit type-checked on its own | pass (final tree and `5185f3e`, `848cc53`, `7c1fd00`, `d293657` each from a clean archive) |
| Unit (`npm run test:unit`) | **102 passed**, 9 new (5 draft store, 4 graph model: record statuses, shared-target relations, text-scaled layout, a dense column). Also at each code commit: 97, 102, 102. |
| Production build (Next 15.5.26 static export) | pass |
| Follow-up regressions (`product-review-s0-followup.spec.ts`) | **14 passed** in each of two separate runs on the final build, and again within the complete suite |
| The review's three diagnostics on the final build | Defects absent. The two asserting diagnostics fail at their defect assertions; the capture-only one shows the corrected graph. |
| Browser, complete suite | **112 passed, 0 failed, 0 flaky, 21 skipped**, on the build of the committed tree. The skips are the 19 opt-in screenshot captures and 2 legacy-viewer tests (historical run directory unavailable). It includes the fresh, unseeded participant journey, every R01–R13 and S0 regression, the follow-up regressions, and the accessibility, exploration, library, demo and v1 suites. An earlier complete run, before the last label change, also passed: 112 passed, 0 failed, 19 skipped. |
| Review screenshot capture (`EXACT_SCREENSHOT_DIR`) | All 19 captures passed. In the complete capture run, 17 passed and the 2 assessment captures failed: the test went offline before the practice material loaded. The test now waits for the question, and both passed when re-run. The 2 main-app captures were retaken after the final label change. |

The fresh, unseeded participant journey (`study-v2-backend.spec.ts`) is part of the complete
suite. It covers consent, setup, background, tutorial interactions, the five questions answered
one at a time, both conditions, per-case reports, the final form and completion. It ran on the
real HTTPS/PostgreSQL service with the normal application CSP.

## Screenshots

The files are in [explanation-product-review-frontend-s0-followup/](explanation-product-review-frontend-s0-followup/).
They were captured by `e2e/review-screenshots.spec.ts` at 1440×900 on the final code. The
main-app captures were retaken after the last change, a disclosure label. There are two
exceptions:
- `before-graph-text200-selected-light` was taken on the build before this pass's view correction.
- `review-diagnostic-shared-filler-after` is the review's own diagnostic capture.

The reviewer's own before images are in their evidence directory.

| View | Files | Shows |
|---|---|---|
| Shared targets | `graph-shared-{light,dark}`, `graph-shared-selected-only-{light,dark}` | One *Syndrome X* node with four differently labelled lines. Selecting *holds only* highlights that line, and the inspector agrees. |
| Unlabelled and long relation names | `graph-labels-{light,dark}`, `review-diagnostic-shared-filler-after` | Local names on the lines and full IRIs in the list. Wrapped and clipped names keep their qualifiers, and no label overlaps another. The second file is the review's own R1 diagnostic re-run on the final build (its capture includes the sticky header). |
| Grouped record revealed | `graph-grouped-light`, `graph-grouped-revealed-{light,dark}` | "+N more records" before; after Show in graph on the fifth synonym, the group is shown and that record's line is focused. The previous hierarchy selection is gone. |
| Undrawable record | `graph-undrawn-{light,dark}` | No stale focus; "This feature is not drawn in the graph." with the record's status and Original record. |
| Dense, long labels | `graph-dense-selected-{light,dark}`, `graph-dense-fit-light` (graph area only; the full-text list is asserted by the dense test) | Readable start zoom around the selection, and stacked labels on the four parallel *holds only* lines. *Fit all* is an explicit overview. |
| Real fixture at 100% and 200% text | `graph-expanded-selected-light` (after Add parents; the fixture has no asserted parent, as the inspector says), `graph-text200-selected-{light,dark}`, `before-graph-text200-selected-light` | Selection and compared pair readable. At 200% the selected line and both its ends stay in view; before the fix they were under the header and off the canvas. |
| Score labels | `main-trace-{light,dark}`, `main-scores-light`, `main-scores-open-{light,dark}`, `study-case-light` | "Matching score" with disclosures, closed and open; the ranking list footer. |
| Assessment drafts | `assessment-offline-back-{light,dark}`, `assessment-reconnected-light` | The answer still selected after offline Next/Previous, and after reconnecting (saved). |
| Real-fixture and tutorial graphs | `main-graph-selected-light`, `tutorial-graph-light` | The real-fixture graph with a selection, and the tutorial graph with the new start view (no side cut off). |

**Data used:**
- **Graph shapes:** the synthetic example.org overrides above, over the real compared pair of the main-app UI fixture (`tools.build_explanation_ui_fixture`; package `package.json` SHA-256 `3689e862…2609`, run `synthetic-ui-fixture`).
- **Study, tutorial and assessment:** the native synthetic v2 fixture (`publication.json` SHA-256 `04db8d2b…085a`; tutorial `tutorial/2.0-native-2-nav65`), as in S0.

This is not realistic review data.

## Preserved safeguards

The following are unchanged:
- admitted-fact validation: only a feature line with its own resolved record is lesson evidence (a grouped record's line once shown); the group node, undrawable records, bridges and context lines save nothing;
- study isolation: the baseline condition still shows no explanations, and the graph is the same shared `PairWorkspace` component;
- the study CSP: no inline styles, and Cytoscape's stylesheet stays reserved;
- outbox idempotency: retried bytes never change under a sent key, and coalescing keys are unchanged;
- server-owned grading and completion;
- the shared main-app/study workspace;
- the v1 resource adapter.

The client adds no fallback for missing backend data. Unavailable records are reported as unavailable.

## Remaining limitations (frontend, this pass)

- **Dense graphs.** Label room is reserved per node and parallel lines are spaced by angle, so
  labels no longer collide in the tested shapes. There is no general collision avoidance,
  though. The highlighted selected line can still cross another line's label, and many long
  wrapped names make tall columns that need panning. At *Fit all* on 60+ features the labels are
  too small to read: this is the explicit overview, and Reset view restores the readable start
  zoom. Lines steeper than about 70° get no more spacing than at 70°.
- **Short forms on lines.** An unlabelled property outside a known prefix is drawn by its local
  name. Two different properties with the same local name would look alike on their lines; the
  list and inspector tell them apart by full IRI.
- **Very large text with a wide graph.** At 160% and 200% text, the selected line and both
  compared concepts can be wider than the canvas at the kept zoom. The selected line always
  stays in view, the view leans toward the other compared concept, and the zoom is not changed
  automatically (spec 21: preserve zoom).
- **Faded geometry.** De-emphasised lines and borders are intentionally below 3:1 while a
  selection is active, though their labels stay ≥ 4.5:1. Whether WCAG 1.4.11 needs 3:1 for
  de-emphasised context is a question for the manual audit.
- **Draft re-saves are bounded** to three per question after conflicts. If two tabs keep
  editing the same question, the server's last write wins and each tab shows the server draft
  once nothing is pending there.
- **Synthetic data only.** Realistic density, real restriction mixes and real annotation
  volumes have not been reviewed (B02).

## Backend-dependent work (not part of this correction)

None of the five findings needed a backend change. The S1 dependencies recorded in the
[S0 handoff](explanation-product-review-frontend-s0.md#backend-dependencies-for-s1-from-the-s0-inventory)
are unchanged:
- B01: structured 422 detail, a reachability-validated publication, an optional `step_id`, task hints;
- B02: realistic 12/24-source review bundles, needed for realistic Q05/Q09/Q10 and dense real graphs;
- B03: entry points and inferred adjacency;
- B04: description preparation kinds and shorter templates;
- B05: real trace examples, exported bridge metadata and distinct evidence statuses. B05 would let "unavailable" and "no graph form" distinguish *not exported*, *withheld* and *unsupported projection*;
- B06: forms/3, setup/3 and consultation/3;
- B07: a persistent review environment.

## Still open (human or environment gates)

- Owner visual review of the corrected graph, the assessment and the score disclosures.
- Manual accessibility audit (J10): screen reader, keyboard-only and zoom, including canvas labels.
- The reload-stall investigation, real-package verification and deployment gates, which stay open as in S0.

## Environment

This pass used the S0 review environment described in the
[S0 handoff](explanation-product-review-frontend-s0.md#environment-s0-review-reproducible):
- `~/Library/Caches/exact-om-dev/`, with Node 24.21.0 and locked dependencies;
- a PostgreSQL 16.2 cluster on 19083;
- the real v2 study service on `https://127.0.0.1:19084`;
- main apps on 19065, 19085 and 19087; demo 19067; local fixture 19068; v1 harness 19066.

Services were restarted after every build, because CSP hashes are computed at start. The
[receipt](explanation-product-review-frontend-s0-followup-receipt.json) lists the commits,
fixtures and per-test results.
