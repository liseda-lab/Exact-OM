# Follow-up integration review — 2026-10-03

**Reviewed checkout: `2d9715b`, following `a18a752`, `ffff5cf` and `8f27e9f`.**
This records the review preceding the new specifications. It does not claim the corrections
in [17](../17-integration-corrective-programme.md),
[18](../18-backend-integration-corrections.md) and
[19](../19-frontend-integration-corrections.md) have been implemented. The specification-writing
pass rechecked the relevant current source and contracts; it did not repeat the entire runtime
review. The remaining findings are bounded to the behavior inspected and tested.

## Improvements confirmed and retained

The main app and study now share `PairWorkspace` and workspace-source adapters. Exact fact
inspection, original-axiom/citation provenance, persistent ranking controls, ordered forms,
interactive six-lesson synthetic training, five server-graded comprehension items, durable
attempts/drafts and tool-neutral setup have been implemented. The corrected consultation
workflow permits multiple methods or none and records the combination per source/candidate-set
case. Backend study workspace routes, publication isolation and expanded persistence checks
exist. The gaps below concern actual wiring, readiness, navigation recovery and timing meaning;
they do not justify replacing these improvements with another simplified interface.

## R01 — Full study workspace is not discovered (P1, confirmed)

In [store.py](../../../exact_inspect/study/store.py), `current_case()` projects a case response
without `workspace`. The v2 model has a workspace field and scoped routes exist, but those
facts do not make the descriptor appear at the HTTP boundary.
[CaseView.tsx](../../../explanations_visualizer/src/components/study/CaseView.tsx) selects the API
adapter only if `current.workspace?.scope_id` exists, then otherwise loads frozen explanation
resources. In the browser, the study therefore displayed “Only the information prepared for
this case is available here” and “Children were not prepared for this study case”, with
structural navigation unavailable. This is narrower than the intended full permitted workspace.

Reproduce with a newly assigned v2 explanation case: inspect `/cases/current`, then its
workspace requests and hierarchy behavior. The missing descriptor and excerpt adapter are
the defect; the existence of shared components is insufficient. The existing
[backend browser test](../../../explanations_visualizer/e2e/study-v2-backend.spec.ts) only checks
that a hierarchy region appears in the scored case and can pass in this state.

## R02 — Target failures do not control case readiness (P1, latent path reproduced)

The API branch of `CaseView.tsx` waits for capabilities and source context, sets content ready,
then permits ranking submission and reports `case_ready`. Target context and other required
visible content load separately in the shared workspace.

Because R01 prevents that branch in the normal service, the review used a **separate temporary
diagnostic service** that supplied the missing descriptor and injected a target-context 503.
The UI displayed a candidate-content error while Submit remained enabled. No product source
was modified to conduct this check. This proves a defect in the latent API path; it is not
evidence that the normal unmodified service already supplied the descriptor. Closing R01
alone would expose R02. Both need an integrated readiness regression against the corrected
real contract, without a synthetic success shim.

## R03 — Tutorial screen position does not reliably resume (P2, confirmed)

In [Tutorial.tsx](../../../explanations_visualizer/src/components/study/Tutorial.tsx), `go(next)`
sets React state and then calls a save closure containing the **old** view. The separate
`lesson_id` in that body is action scope, not the persisted `current_lesson_id`. On the backend,
[tutorial.py](../../../exact_inspect/study/tutorial.py) unconditionally assigns the latter
from `data.get`, so omission can also clear it. Assessment position is represented by null,
which the frontend restores as the first lesson.

Observed reproduction: move from lesson 1 to 2, wait for Saved without doing a lesson-2
interaction, reload, and return to lesson 1. The existing test performs additional lesson-2
actions before reload, masking the navigation-only bug. Assessment navigation/drafts require
their own reload regression. Actions, assessment answers and attempts do persist; this finding
must not be reported as blanket loss of tutorial progress. The current navigation also uses
scrolling without moving keyboard focus to the new lesson, which the frontend correction covers.

## R04 — Multiple pages can be mistaken for full elapsed coverage (P2, confirmed)

[telemetry.py](../../../exact_inspect/study/telemetry.py) checks segment overlap within a page
instance, as expected for page-local monotonic clocks. The export in `store.py` then sums
eligible case observations from all page instances and computes
`max(0, raw_elapsed_seconds - observed_segment_seconds)` as unobserved elapsed time.

One synthetic completed session using normal and diagnostic pages exported, for its second
case, raw elapsed **457.105115 seconds**, observed page segments **589.2970999999046 seconds**,
unobserved **0**, and three contributing page instances. The raw observations are not inherently
invalid; interpreting their sum as unique coverage is unsupported. Existing
`active_duration_known: false` is correct and must remain, but does not repair the inferred
zero unobserved value. Independent pages can overlap, and their monotonic clocks cannot be
unioned without additional evidence. The correction uses versioned exports with explicit
unknown coverage and retains the raw page observations.

## Verification recorded in the preceding review

- Frontend: 49 unit tests passed; production build including type checking passed.
- Selected backend invocations: 60 passed/1 skipped in a focused run, 105 passed/1 skipped
  in a broader run, then the source-image import check passed once its interpreter environment
  was corrected. These are invocation counts (166 pass results), not a claim of 166 distinct
  new tests or complete repository coverage.
- PostgreSQL: a 20-pass/1-skip run, followed by two targeted restart/restore passes. Some
  coverage overlaps; this is not a claim of 22 unique database scenarios.
- Specification validation: 26 JSON files, 13 schema fixtures, 131 local links, eight schedules
  and nine scoring oracles passed at the review baseline. These counts change with the new specs.
- Browser: setup and part of training/assessment were inspected from a fresh session. A
  separate session prepared through the API completed the four-case UI journey in both
  conditions, consultation, final form and completion. This was **not** an independently
  completed fresh all-training/all-case browser journey in that review.
- Desktop at 1280px with 200% text had no horizontal overflow. The attempted narrow viewport
  override did not take effect, so 320px behavior was not independently reverified.

The relevant source was rechecked while writing these specs. Temporary diagnostic changes
were confined to the test environment; review servers were stopped and browser preferences
restored. Test invitations, researcher tokens and raw session records are intentionally not
included in this repository record. Future acceptance receipts must capture exact commands,
environments, fixture hashes and outputs reproducibly rather than relying on these counts.

## Explicit remaining verification limits

The latest real backend-release package was unavailable in this review. PostgreSQL tests
are useful evidence but not proof of a Docker/Render deployment. A copied-source image-layout
import check is not an actual built container. Manual screen-reader checks, renewed 320px
verification, formative usability and the tutorial/case duration pilot remain outstanding.
Historical original pinned DOID/import-closure G1/G5 blockers, real case adjudication,
study-owner information/consent approval and deployment/recovery launch gates retain their
existing status. No human participants were recruited and no public deployment was performed.

The follow-up therefore requires backend-first delivery plus joint browser/API/database
acceptance. Green unit suites, generated schemas or earlier “implemented” handoffs alone
cannot close these four findings or certify the entire study for launch.
