# Product-review frontend S0 follow-up 2: assessment recovery — handoff, 2026-10-10

**Status: the two findings of the [independent follow-up review](explanation-product-review-frontend-s0-followup-review.md)
are corrected and verified against the existing (unchanged) backend, using synthetic sessions
only.** The verified graph and copy corrections are unchanged. This is not a claim of usability,
accessibility conformance, real-data quality or study readiness.

| Finding | Disposition | Main evidence |
|---|---|---|
| **S0-FU-R1** an older same-question answer was saved automatically over a newer one after a 409 | **Corrected.** After a failed write, an answer is saved again automatically only when the saved answer to that question still equals the answer this tab's work started from. Otherwise the saved answer is shown, and this tab's answer is kept for an explicit choice. The choice survives a reload. | Browser tests on the real study service: same question, single choice and multi-part; independent questions. Unit cases. |
| **S0-FU-R2** reload cleared the restored latest answer before older queued writes settled | **Corrected.** Restored outbox writes now settle like new ones, and each counts until it settles. Equality with the saved answer never clears pending work early. | Browser tests on the real study service: saved A → in-flight B → latest A → reload; distinct A/B/C multi-part; terminal replay failures (another question changed; the same question changed); a delayed acknowledgement. Check answer submits the latest answer. Unit cases. |

No backend code, contract, validation or grading changed. Server conflict detection,
idempotency, session isolation and server-owned grading are untouched. Neither finding needed
a contract change.

## Commits

| Commit | Content |
|---|---|
| `b6dc527` | S0-FU-R1 and S0-FU-R2: outbox settlement for restored writes and client-only write context; base-proven re-saves; conflict choice in the question card |
| `850239d` | `e2e/assessment-recovery.spec.ts`: the review's diagnostics as expected-behaviour regressions, with opt-in screenshots |
| this handoff | the following documentation commit, with the screenshots and receipt |

## Reproduction before correction

Both failures were reproduced first, on the build of the reviewed commit `f5eca54`.

1. **The review's two diagnostics** (unchanged except for their screenshot directory) passed,
   which means both defects were present.
   - **Same question:** the held older write got a 409 at expected revision 4. An automatic
     re-save at revision 5 then succeeded and replaced the other tab's *certain* with *advice*.
   - **Restored replay:** the intermediate *certain* was visible while the latest *advice* was
     still queued.
2. **The new regression spec on the same pre-fix build.** Five of its eight tests failed, each on
   its finding:

| Test | Pre-fix result |
|---|---|
| Same question, single choice | Fails "no automatic re-save": the writes were 409 then 200, and the older answer replaced the newer one. |
| Same question, multi-part | Fails "no automatic re-save" in the same way. |
| Saved A, in-flight B, latest A, reload | Fails: A is unchecked once B is acknowledged, so the intermediate answer is shown. |
| Terminal replay failure, another question changed | Fails: only the 409 is recorded. The restored latest answer C is never saved, because restored writes had no settlement. |
| Terminal replay failure, same question changed | Fails: the saved answer (changed elsewhere) is not shown, and no choice is offered. |
| Independent questions | Passes on both builds; the earlier code also re-saved here. Kept as a guard, because the new rule must still save it. |
| Distinct A/B/C multi-part restore | Passes on both builds; the old defect needs the latest answer to equal the saved one. Kept as a guard. |
| Delayed acknowledgement, then Check answer | Passes on both builds. Kept as a guard. |

3. **On the corrected build both review diagnostics fail at their defect assertions.** That
   means the defects are gone:
   - the writes stay a single 409 and never become 409, 200;
   - the intermediate answer is never the checked one.

## Corrections

### The rule (`src/study/assessmentDrafts.ts`)

Each piece of unsaved work for a question has a **base**: the saved answer it started from.

**Why the base proves safety.** Every write the outbox sends carries the revision this tab last
saw, so the server accepts it only if nothing changed since this tab's last observed state. The
only way to overwrite another tab's answer is therefore to write after observing it. So:
- **A write for a question is issued only while the observed saved answer equals the base.** This tab's own acknowledged answer becomes the new base.
- **After a write fails** (its own 409, or discarded behind another failure), the answer is saved again automatically only if the saved answer, in the latest accepted state, still equals the base. This is a value-based three-way merge: the other side left this question unchanged. Automatic re-saves stop after three without a success.
- **Otherwise the question becomes a conflict.** The saved answer is shown and is what Check answer would use. Check answer is disabled until a choice is made, and this tab's answer is kept, described in words. The participant chooses **Use this tab's answer** (written on top of the saved answer they were shown), **Keep the saved answer**, or edits the shown answer. Editing is also a deliberate choice, based on what they see.
- **Conflicts stay recoverable.** Unresolved conflicts are kept in this tab's session storage, so a reload still offers the choice. The outbox's recovery archive also keeps the rejected writes, as before.
- **Multi-part answers are compared as a whole.** A change to any part elsewhere is a conflict. No field-level merge is attempted.

### Settlement for restored writes (`src/study/mutationQueue.ts`)

Before this pass, restored outbox entries had no-op callbacks. The page could not know when they
settled, so it counted them as settled immediately. Now every entry has its own settlement
promise, and `queued()` returns it with copies of what the entry carries:
- On reload, the tutorial counts each restored write for a question until it settles. The latest local answer stays visible until all of them have settled and the saved answer equals it.
- An unobserved restored rejection is marked handled.
- A coalesced draft is rejected with `ReplacedMutation`, so it is not treated as a failure.

Queued writes also carry optional `meta`, client-only context. It is persisted and restored with
the entry, and the session never sends it (the request body is still built from `body` only).
Assessment drafts use it for their base, so the proof survives a reload. Older outbox entries
without a base are treated as unknown: they never re-save automatically after a failure.

### Tutorial wiring (`src/components/study/Tutorial.tsx`)

The store is now driven by settlements: new and restored writes acknowledge, fail or are
replaced, and each settlement is compared with the latest accepted state. A new synchronous
`session.current()` makes that comparison use exactly the state the next write is made against.

The previous automatic re-send loop has been removed. The question card shows the conflict
choice, and the question list marks a question that needs one.

Check answer submits the visible answer:
- normally the latest local answer, queued behind its pending draft writes;
- during a conflict, nothing until a choice is made.

**Unchanged:** coalescing keys, the 500 ms debounce, retry identity (key, revision and body),
grading, attempts, Back/Next navigation, the session-level conflict banner and recovery
download, and the conflict handling of other stages.

## Tests executed

These ran on 2026-10-10 against the S0 review environment, on the build of the final working
tree. That tree is identical to the committed frontend sources.

| Check | Result |
|---|---|
| Type check | pass (final tree, and `b6dc527` and `850239d` each from a clean archive) |
| Unit (`npm run test:unit`) | **112 passed**, 0 failed, at each commit and on the final tree. Draft store: 12 cases, replacing the earlier 5. Outbox: 3 new cases (restored-write settlement and context, observable terminal failures, coalesced replacement). |
| Production build (Next 15.5.26 static export) | pass |
| New regressions (`assessment-recovery.spec.ts`) | **8 passed** in each of two separate runs on the final build (one capturing the screenshots), and again within the complete suite. An earlier run before the last copy change also passed 8/8. |
| The review's two diagnostics on the corrected build | Both fail at their defect assertions (the defects are absent) |
| Browser, complete suite | **120 passed, 0 failed, 0 flaky, 21 skipped** on the build of the committed tree. The skips are the 19 opt-in screenshot captures and 2 legacy-viewer tests (historical run directory unavailable). An earlier complete run, before the last copy change, gave the same result. |

The complete suite includes:
- the fresh, unseeded participant journey (`study-v2-backend.spec.ts`): consent, setup, background, tutorial interactions, the five questions one at a time, both conditions, per-case reports, final form and completion;
- the earlier S0 and follow-up regressions, including the original offline, reload and competing-tabs draft tests;
- the graph and copy regressions;
- the accessibility, exploration, library, demo, v1 and v2 integration suites.

It ran on the real HTTPS/PostgreSQL service with the normal application CSP.

## Screenshots

The files are in [explanation-product-review-frontend-s0-followup-2/](explanation-product-review-frontend-s0-followup-2/).
They were captured by the new regressions with `EXACT_SCREENSHOT_DIR` set, at 1440×900 in light
and dark, on the synthetic v2 study fixture (`publication.json` SHA-256 `04db8d2b…085a`).

| Files | Shows |
|---|---|
| `fu-r1-conflict-single-{light,dark}` | The saved answer from the other tab is checked. The card says another tab saved a different answer and names this tab's answer, with the two choices. Check answer is disabled, and the question list marks the question. |
| `fu-r1-conflict-multipart-{light,dark}` | The same for a multi-part answer. |
| `fu-r1-resolved-this-tab-{light,dark}` | After **Use this tab's answer**: this tab's whole answer is shown and saved. |
| `fu-r2-latest-visible-while-replaying-{light,dark}` | After a reload, with the intermediate answer acknowledged and the latest still held, the latest answer is checked. |
| `fu-r2-replay-conflict-{light,dark}` | A restored replay that failed after the same question changed elsewhere. The saved answer is shown and the restored latest answer is offered. |

## Remaining limitations

- **No field-level merge.** Multi-part answers conflict if any part changed elsewhere, even a
  different part. The participant resolves this; "Use this tab's answer" writes this tab's
  whole answer.
- **Value-based proof.** If another tab changes a question and then changes it back to this
  tab's base, the change is invisible to the proof, and this tab's answer is saved on top. This
  is the standard three-way-merge reading of an unchanged value. Per-question revisions would
  need a backend contract and are not required here.
- **Bounded automatic re-saves.** After three re-saves without a success (for example, two tabs
  continuously answering different questions), the question becomes a choice. It never loops.
- **Conflicts are kept per tab.** They are in session storage: a different tab or a new browser
  does not inherit another tab's unresolved choice. The recovery download still contains the
  rejected writes, as before. Browser-loss guarantees remain limited to server-acknowledged data
  (spec 12).
- **Outbox entries written before this change carry no base.** If such a restored write fails,
  the question becomes a choice instead of a re-save.

## Backend-dependent work and open gates

None. Backend S1 (B01–B07 in [22](../../specs/explanation-framework/22-review-backend-and-data.md))
is independent and unchanged. Still open:
- owner review;
- the manual accessibility audit (J10), including the new conflict panel;
- real-package and deployment gates;
- the historical intermittent reload investigation;
- S1, S2 and S3.

## Environment

This pass used the S0 review environment, as in the
[S0 handoff](explanation-product-review-frontend-s0.md#environment-s0-review-reproducible):
- `~/Library/Caches/exact-om-dev/`, with Node 24.21.0 and locked dependencies;
- PostgreSQL 16.2 on 19083;
- the real v2 study service on `https://127.0.0.1:19084`;
- main apps on 19065, 19085 and 19087.

Services were restarted after every build. The
[receipt](explanation-product-review-frontend-s0-followup-2-receipt.json) lists commits, the
reproduction and per-test results.
