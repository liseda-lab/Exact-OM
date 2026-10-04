# Frontend assignment: consume and verify the complete study workspace

**2026-10-04. F11–F15 implemented in `6895bc6`; F16–F19 (R06–R09) verified at `23aa74f`
([follow-up](../../docs/verification/explanation-integration-frontend-followup.md)); F20–F21
(R10–R11) in `d8b19a0`/`e754e77`, verified at `eb4f955`
([second follow-up](../../docs/verification/explanation-integration-frontend-followup-2.md)).
F21 is reopened for the frozen-resource adapter (R12, [third review](evidence/frontend-integration-review-20261004-3.md)).
R01, R02 and R04 are joint-verified on the synthetic fixture; R03 awaits the manual
screen-reader/keyboard audit in J10.** See the
[frontend handoff](../../docs/verification/explanation-integration-frontend-handoff.md) and the
[joint acceptance record](../../docs/verification/explanation-integration-acceptance.md). The
requirements below remain normative. Follow
[17](17-integration-corrective-programme.md), consume the backend handoff from
[18](18-backend-integration-corrections.md), and preserve the shared components and study
decisions already implemented under [15](15-frontend-corrections.md). This is an integration
and recovery correction, not a second design rewrite.

## F11 / R01 — Explicit adapter selection and compatibility

Before implementation, verify the backend handoff's SHA, fixture hashes, runtime extension
`study-integration/1` and actual HTTP examples. For `exact-study/2.0` explanation cases, require
the current-case `workspace.scope_id`, then validate returned capabilities against current
study revision, scope, ontology identities and admitted components. Use `createApiSource`
and the shared `PairWorkspace` with the authorized `/api/v1/study/workspace/…` routes.

Never fall back to `createResourceSource` merely because a v2 descriptor is missing, a read
fails or capabilities are incompatible. Show a recoverable service/version error, retain
saved work, expose retry and the ordinary pause/support path, and block scored submission.
If the service lacks the extension, label the incompatibility honestly; do not collect new
corrected-flow answers against a guessed contract. Existing server state must remain resumable
after the compatible service is restored. Reject malformed or cross-presentation responses.

Keep the explicitly selected v1 legacy adapter and the synthetic tutorial resource adapter
where authorized; these are not emergency fallbacks for v2 scored cases. Baseline continues
with identity, ranking and download controls and sends no scored workspace requests. Opening
synthetic help in baseline must not switch its scoped source or disclose scored explanations.
An explanation error must never switch the participant into baseline.

Successful v2 study navigation supports the same permitted information operations as the main
app: typed labels, entity context, both declared hierarchy bases, bounded parent/child/fact
pagination, exact citation/original axiom inspection and admitted explanation/evidence views.
Reasoner-inferred views remain unavailable when inference was not run. Show real scope and
completeness status from capabilities; remove excerpt-only notices only after real full-scope
support is confirmed. Never replace an honest status with a claim of unrestricted coverage.

Cache keys and invalidation must include the study revision, session/invitation binding,
presentation, workspace scope, policy, entity and query/basis where relevant. Abort or ignore
old requests when any identity changes. Do not reuse a tutorial cache as scored evidence or
leak previously authorized data after revocation/session replacement. Avoid duplicate fetching
by sharing the readiness loader's validated results with the actual workspace components.

## F12 / R02 — Required content, rendered readiness and submission

Use an explicit per-presentation readiness state: `loading`, `usable`, `blocked` and
`submitting`. It must be derived from the shared source/load state, not from a source-only
prefetch or the existence of a mounted heading. Loading and failure are distinct from valid
absence. Keep the ranking visible, but prevent scored submission until required content is usable.

### Bounded minimum before initial readiness

For every case, validate the current presentation, source identity, exactly five distinct
candidates in their declared display order and the required ontology download metadata/access
status. Preserve the existing setup verification of frozen downloadable bytes; do not download
entire ontologies on each case merely to mark readiness.

For an explanation case, also resolve and validate:

1. The authorized workspace capabilities.
2. The source and **all five candidates'** initial entity-context responses, including the
   first bounded pages and completeness/availability metadata used by the description view.
3. Initial source/candidate profile and per-pair comparison results for every component
   admitted and displayed by the frozen condition. Explicit successful `not_requested`,
   `not_exported`, policy exclusion or genuinely absent content is a terminal honest status;
   request failure, malformed data or a dangling advertised resource is not.
4. The currently selected source/target description view and ranking controls can render
   those validated results, including unavailable-content explanations. A render error or
   invalid entity binding prevents readiness even when transport returned 200.

This is bounded to six focal entities and five comparisons, with deduplication and capped
concurrency. It does **not** require participants to inspect every candidate, nor does it
fetch all facts, all hierarchy pages, every citation or the full ontology/graph. Baseline's
minimum deliberately excludes integrated explanation content; no hidden prefetch is allowed.

Fact expansion, hierarchy exploration, additional category pages, graph expansion and exact
axiom dialogs remain lazy. Their loading/failure is local and visible, with retry and preserved
selection. An optional detail failure alone does not erase already usable core content or
automatically prevent submission. A failure invalidating required core content, the session
or the policy does block submission. Do not infer absence from either kind of failure.

### State and timing rules

| Transition | Required behavior |
|---|---|
| New/reloaded presentation → loading | Bind all requests to its identity, show progress, preserve any recovered ranking separately, disable submission; no new page's case timing before its minimum is usable. |
| Required content resolves and renders → usable | Mark this page/presentation ready once, enable ranking submission, begin eligible case observations. All five initial contexts are already available for switching. |
| Required request fails → blocked | Show which content failed and a retry action. Preserve rankings, inspected candidate, draft and usable content; do not offer “insufficient evidence” as a substitute for a transport failure. |
| Retry → usable | Reuse still-valid results, fetch failed dependencies, discard stale results. Do not reset the logical case's server start or duplicate a submission. |
| Later core invalidation/session change | Disable submission, retain recoverable draft under the original binding; never mark another case ready from late responses. |
| Submit | Check current readiness, response validity and session/revision; flush timing and required draft queue in the existing order, then submit once with idempotency. |

Rank editing is available after the initial minimum is usable; during a blocking core error
retain the last answer visibly and allow pending saves of that existing draft to settle safely.
Do not erase a saved answer just because the content temporarily fails to reload. Recovery,
conflict and pause controls remain available. A 409 refreshes canonical state and cannot merge
old answers into a new presentation.

Treat first usable content as one logical presentation start across pages. Each page may send
its own idempotent ready observation after its own successful load; the backend's earliest
accepted start remains authoritative. Candidate switching, retry, re-render, assessment help
and a second tab must not restart the task. Initial load is excluded from that page's case
segments; later interruption does not retroactively alter the start or fabricate active time.
Record already-supported failure/recovery events and keep uncertainty in exports. Do not add
an uncontracted event type or silently subtract an estimated outage. Hidden tabs may represent
external inspection and must not automatically pause the study. Explicit pause behavior remains
as specified. Technical delays are not evidence about participants' attention or method use.

Tests must include source success/target 503, slow fifth candidate, profile/comparison failure,
authorized true absence, malformed descriptor/capabilities, render failure, abort on session
replacement and a lazy hierarchy-only failure. Check both disabled submission and actual
absence of premature `case_ready`/eligible timing requests. An enabled button assertion alone
does not establish timing correctness. Retry must preserve partial ranking and its order.

## F13 / R03 — Save the destination and restore the tutorial

Use the explicit `position` union from 18 for local and persisted navigation. On Next/Back,
lesson selection, assessment entry or assessment-question navigation, construct the **destination**
first, then save that exact object through the session queue. Do not obtain it from a closure
over the previous React state. The assessment landing view must persist even when no answer
has been entered. Restore the server's acknowledged position on reload/new device and after
conflict reconciliation; local optimistic position must never be shown as durably saved.

Non-navigation actions and assessment drafts omit position unless deliberately changing it.
Make navigation coalescing safe: the latest destination can replace an unsent destination,
but cannot drop accumulated action evidence, practice answers, drafts or assessment attempts.
Never coalesce distinct assessment attempts or claim a failed/offline write was saved. On
reload while offline, restore only a session-bound pending draft and visibly distinguish it
from acknowledged server state. Flush needed navigation/draft saves before stage completion.

Use the server's normalization for historical progress; do not maintain a different inference
rule in the client. Reopening completed synthetic help preserves completion, assignment and
the scored case. Correctly handling position must not skip the server's requirements or
assessment gate. A read-only/help view is not a new attempt.

After intentional lesson/view navigation, move focus to the newly displayed heading (or an
appropriate assessment heading), expose its name to assistive technology and scroll it into
view. Do not leave keyboard focus on a button at the bottom of the previous lesson. Initial
restoration should provide a clear current heading; background autosaves must not steal focus.
Dialogs return focus to their trigger; invalid assessment responses focus their feedback or
first invalid field with an accessible announcement.

Required browser regressions: lesson 1 → 2 → wait Saved → reload **without performing any
lesson-2 action**; assessment entry → wait Saved → reload **without an answer**; partially
answered assessment → reload; incorrect attempt → feedback → retry → server pass; rapid
Next/Back while saves are delayed; offline/reconnect; two-tab conflict; help after completion.
For each, assert both visible destination and server position, with preserved answers/attempts.

## F14 / R04 — Consume the corrected exports honestly

In the researcher UI explicitly request `analysis_schema=exact-study-analysis/3` for v2
exports in both JSON and CSV formats. Use the frozen v1 path for v1. Verify the returned
manifest/version instead of changing a download label. An old unsupported backend reports
incompatibility; do not silently retry with 2 while presenting it as corrected analysis.

Show source publication protocol and derived export schema separately where version metadata
is presented. Historical downloads retain their original identity. If any UI displays timing,
label page observations as such and unknown coverage as unknown; never show null as zero or
as a corrected active duration. Do not recompute timing from events in the browser. Preserve
researcher-only authorization, test-session exclusions and existing export options.

## F15 — Test the connection, not just the appearance

Strengthen `e2e/study-v2-backend.spec.ts` and related tests to assert actual successful scoped
network requests and the descriptor, not just a hierarchy region. Use the backend-generated
fixture from S1. Search outside the focal excerpt and traverse/paginate actual ontology data;
open a citation and assert the original identity. Compare this path with the same prepared
information in the main app. No private answer keys or unrestricted exploration routes may
appear in the participant's network responses.

At least one full acceptance journey uses the compiled frontend served by the real backend,
its normal CSP, HTTPS/cookie/origin rules and PostgreSQL. Browser fault injection is allowed
for separate error tests, with injected failures labeled. Development previews, manually
patched descriptors and mocked successful APIs are not integrated acceptance evidence.
Record console/request failures, service/database/build versions and actual viewport metrics.

Complete J03–J07 and J10–J12 from 17, plus browser coverage of J01/J02/J08/J09. Include the
whole tutorial in the unseeded journey. Unit/API tests can cover combinations, but must not
replace observed error recovery, keyboard focus and real request boundaries. Verify the
actual 320px layout, 200% text and desktop case, dark theme and ranking visibility. Perform
manual keyboard and screen-reader checks on the changed controls; automated accessibility
results alone are not a manual audit. Record any unavailable audit as unverified.

Recheck existing behavior: ranked/none/insufficient answers, ordered and branching forms,
per-case draft restore, Yes/No method clearing, mixed method use across cases, downloads,
baseline isolation, invitation reissue/revocation, 409/offline recovery, static route gating,
CSP, local import, fixed demo, graph/list equivalence and exact evidence provenance. Preserve
the main app's appearance and working shared components while correcting study integration.

Deliver S2/S3 handoffs with backend/frontend SHAs, contract and fixture hashes, test results,
newly discovered issues and release gates. Do not close R01 on a shared import, R02 on one
prefetch, R03 on a reload after extra actions, or R04 on a summed timing field. Any backend
failure goes back to its owner with the request, response and minimal reproduction.

## F16 / R06 — Continue every truncatable fact category

The shared entity card must continue every category it displays and the response pages
(definitions, alternate definitions, synonyms, each defining-fact category, parents and the
other recorded facts) from that page's own cursor:
- fact categories through `/entity-facts` with the category;
- parents through the parent hierarchy route.

Show “N of M” when a total is known and “more exist” when only a cursor is known. Never claim
“all are shown” for a page that has a continuation. Continuation is lazy, local and
deduplicated by fact or edge identity, appends in the service's order, and keeps earlier items,
citation access and fact identities. Loading, failure and retry are local and never read as
absence. The main app and study use the same component. Tests must page beyond the first page
in both products and prove every item becomes reachable exactly once.

## F17 / R07 — Full context inside the study workspace

The study's hierarchy browser offers “Open full context” for any entity it focuses, opening
the shared full-context card over the authorized scope. Original facts appear only when
`original_context` is admitted. Generated descriptions appear only for focal entities and
admitted `entity_description`. Restricted reads are never requested, and the baseline still
has no workspace. Opening or closing the dialog keeps the ranking, draft, inspected candidate
and hierarchy navigation; a parent selected inside it navigates the hierarchy. Opening reports
an existing declared event type.

## F18 / R08 — Validate before caching; retry refetches unusable results

The shared read cache accepts a focal context or explanation only after its required
structure validates: identity, label, pages with item lists, category pages and completeness
for contexts; task, entities and claims for available explanations. Anything else is a local
failure, evicted so that the next read or Retry fetches it again.

A render failure evicts the cached focal results before Retry remounts the workspace.

Readiness transitions are tagged with their attempt, so a late load, render report or older
attempt cannot overwrite a blocked state or ready another attempt, presentation or session.

A 403 from an optional read blocks submission only if the scope's capabilities are then also
refused. Otherwise it stays local to its panel.

## F19 / R09 — Readiness follows the admitted components

The render check requires exactly what the frozen condition displays:
- the pair question always;
- both entity cards when `original_context` or `entity_description` is admitted;
- generated descriptions when `entity_description` is admitted;
- the comparison when `pair_comparison` is admitted.

Lazy tabs are never required. Absent components are not awaited, and admitted ones are not
skipped. Tests cover the full set and comparison-only, original-context-only,
description-only and hierarchy/evidence-only publications. Each must show readiness,
recorded absence, failure and retry.

## F20 / R10 — Access checks belong to the work that issued them

When an optional read is refused and the shell re-checks the scope's capabilities, bind that
check to the attempt, workspace scope and session current when it is **issued**. The shared
read source survives retries, so the binding cannot be fixed when that source is created.

Retry and teardown cancel outstanding checks. A check that still resolves for a superseded
attempt, presentation, scope or session is ignored. A check issued and refused within the
current attempt still blocks submission. Recovery keeps the ranking, draft and validated
cached content.

Tests must hold an older check across a successful Retry and release it afterwards. They must
also show that a check from an obsolete presentation or session cannot affect the current case,
and that a genuine current loss still blocks.

## F21 / R11 — Navigation carries typed entities

Every navigation callback receives the complete typed entity the backend supplied
(`ontology_version_id`, `iri`, `kind`), never an IRI plus a guessed kind:
- card parents from either the first page or a continuation;
- original-axiom terms;
- evidence-list terms;
- the main app's browse view and comparison workspace, and the study's pair cards and
  full-context dialog.

A term without a recorded type, such as a bare IRI value, is shown but not navigable.

The frozen-resource adapter (the synthetic tutorial and v1 study publications) supplies the
same typed parent for every card parent (R12): the recorded hierarchy edge for that fact, else
the type the axiom's own AST node records. It keeps the parent's ontology, the fact's identity
and its provenance. It never defaults a kind; a parent whose type is unrecorded or
contradictory stays shown but not navigable.

Tests cover object and data properties with more than one page of named superproperties in
both products, and keep class coverage. A test chooses a later-page parent and checks its
ontology, IRI and kind, that its context loads, and that the study ranking and draft are kept.
Tests also open the tutorial's card parents (checking the lesson, the inspected candidate and
progress are kept, and that the service accepts a card parent as a typed lesson action) and a
v1 publication's card parents, and show that an untyped parent stays non-navigable.

Assertions about recorded traffic wait, with a bound, for the response they depend on. The
traffic recorder adds a request only after it finishes, which can follow the UI's recovery
(R13).

