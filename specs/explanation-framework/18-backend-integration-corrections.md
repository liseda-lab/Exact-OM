# Backend assignment: discovery, recovery and timing integrity

**S1 baseline (2026-10-03): implemented and backend-verified in `96b76fe`.**
Current disposition and the subsequent R14–R16 backend review are recorded in the
[final backend review](../../docs/verification/explanation-integration-backend-final-review.md).
See the [S1 handoff](../../docs/verification/explanation-integration-backend-handoff.md)
and its exact verification receipt. The requirements below remain normative. Read
[17](17-integration-corrective-programme.md) for order, ownership and acceptance. This assignment
extends [16](16-backend-corrections.md), rather than replacing its implemented security,
preparation, publication or storage behavior. Deliver the backend handoff before frontend
integration. Review the actual serialized routes, not only the response model declarations.

## B10 — Extension and compatibility boundary

Keep the publication protocol `exact-study/2.0`. Add the runtime extension identifier
`integration_contract: "study-integration/1"` to authenticated v2 study state. This is a service
capability, not a change to frozen publication bytes, tutorial content or assignment. Advertise
it only when all wire changes in B11–B13 are supported. V1 state and mutations keep their
original contract. The frontend must not guess support from the existence of unrelated v2 routes.

Update strict request/response models, route projections, OpenAPI, runtime schemas and actual
HTTP contract tests together. Explicitly model the current-case descriptor, tutorial position
and export selector. Preserve field presence through validation: an omitted patch field must
not become an explicit null/default before it reaches the store. Test serialization through
the HTTP router and database round trip. Extra fields merely present in an internal dict are
not a delivered contract.

Existing v2 publications/sessions resume under their original conditions and content hashes.
New state metadata can be derived on read; do not rewrite old publications to add it. Bind
newly published revisions to their actual compatible software build under the existing rules.
Any deployment incompatible with an active revision must fail admission explicitly, not
reset the session or silently change its protocol. Preserve immutable old mutation receipts;
idempotent replays may return their original shape, after which the client can refresh state.

### R15 / R16 — Interrupted response integrity and reload investigation

Security/origin middleware must forward the producer's body, completion and cancellation
semantics without manufacturing a successful ending for an interrupted response. Preserve
declared Content-Length, duplicate cookies, exact document CSP, authentication, response
bytes and producer/transport exceptions. Test a producer interrupted before and after body
bytes with real HTTP framing, cancellation propagation, disconnected transport cleanup,
HEAD and subsequent requests. Do not suppress exceptions or change dependencies solely
because an h11 error appeared near a disconnect.

R15 covers the reproduced middleware framing defect. R16 separately tracks the historical
browser reload stall: a deterministic injected protocol failure does not prove that it
caused that symptom. Exercise the compiled frontend on the actual HTTPS/PostgreSQL service
under normal CSP, recording ordinary reloads, outstanding responses, cancellation,
interrupted streams and later requests. If the stall remains unreproduced, retain it as
unresolved with exact scenario counts and instrumentation; do not mark it fixed by R15.

## B11 / R01 — Discover the participant's authorized workspace

`GET /api/v1/study/cases/current` for a v2 explanation presentation must include:

```json
{
  "workspace": { "scope_id": "opaque-current-case-scope" }
}
```

This is an additive fragment of the existing case response, not a replacement response.
Select the descriptor from the **frozen current presentation's** admitted `WorkspaceScope`.
Return only the scope ID, never context file paths, database locators, other case scopes or
private keys. V2 baseline returns `workspace: null` and no explanation refs. V1 behavior
remains explicit legacy behavior. Missing or invalid required v2 workspace configuration is
a typed service/configuration failure (503), not a successful case with null/empty context.

The corrected frontend constructs the existing same-origin route family
`/api/v1/study/workspace/{encoded_scope_id}/…`. Do not add a participant-provided arbitrary
base URL or infer a scope from `case_id`. Capabilities must match scope, study revision,
ontology versions and permitted components, and describe actual declared context coverage.
Use the existing `complete_declared_context_scope` meaning: complete for the frozen filtered
scope, not unrestricted access to every ontology or every case.

Scope IDs are locators, not bearer authorization. Every descriptor and workspace read must
retain session, invitation-generation, consent, stage, condition, publication/presentation
and policy checks. Current-case discovery must not authorize a previously denied read during
pause, consultation or after advancing. Preserve existing route-specific stage permissions;
test them explicitly. Synthetic tutorial help must never grant scored-baseline access.
Stale cursors and caches must remain bound to the original scope/query/policy/session rules.

Deliver actual serialized fixtures for explanation/baseline, missing workspace, denied stale
scope and mismatched cursor. Build a synthetic ontology with a non-focal searchable entity,
multiple inheritance and enough facts/children to require continuation. Assert full paging,
exact axiom/citation identity and no duplicate/lost facts. The main app and study must read
the same frozen information when the same scope permits it; an excerpt-only fixture cannot
prove navigation parity. Never access the unrestricted exploration API from the study client.

## B12 / R03 — Explicit durable tutorial position

### R14 — Explicit typed tutorial action evidence (2026-10-04)

When a v2 `TutorialAction` supplies `entity`, its `kind` is required at the HTTP
request boundary: exactly `class`, `object_property`, `data_property` or `individual`.
Omitted, null or unknown kinds return 422 before mutation. The full typed identity
must belong to the frozen tutorial scope and satisfy the action's recorded relation;
a legal kind with the wrong identity also returns 422 atomically. Never infer a class
from an IRI or fill the kind before validating action evidence. The frontend already
sends recorded kinds (19 F21 / R12).

Use an action-specific entity model. Preserve shared legacy entity defaults and
frozen v1/v2 publication canonicalization, content hashes, resume state and exports.
Previously stored action receipts remain immutable. An old request that omitted kind
now fails validation, including retries; resending its explicit normalized kind under
the same idempotency key preserves the original canonical request hash and replays
the original receipt. This is stricter runtime action validation, not a publication
protocol migration. Regenerate OpenAPI and test actual HTTP rejection, exact typed
identity, unchanged state/history on failure and historical receipt replay.

### Durable position contract

Add `position` to `TutorialReceipt` and the v2 tutorial progress patch. Its two complete shapes
are:

```json
{ "view": "lesson", "lesson_id": "lesson-2", "question_id": null }
```

```json
{ "view": "assessment", "lesson_id": null, "question_id": null }
```

An assessment position may set `question_id` to an ID in the frozen assessment; null means
the assessment landing view. A lesson position requires a real frozen lesson ID and null
question. Use a typed discriminated union, bounded IDs and strict unknown-field rejection.
No magic lesson string called “assessment”. Position is navigation state, not a completion
predicate, attempt, grade or change to the frozen tutorial. Navigation to a lesson must not
automatically satisfy an interaction requirement.

Patch rules:

| Request | Effect |
|---|---|
| `position` omitted | Preserve the previous position; actions/drafts can still change independently. |
| Valid complete `position` object | Atomically store exactly that destination with the rest of the mutation. |
| `position: null`, unknown IDs or inconsistent view/ID combination | 422, no partial state change. |
| Position-only mutation | Valid; participates in revision/idempotency/history and can be acknowledged without an action. |
| Changed body under the same idempotency key or stale revision | Existing 409 rules; preserve canonical state and acknowledge identical replay as before. |

Keep `current_lesson_id` as a deprecated v2 compatibility field for old clients; corrected
clients send only `position`. A non-null legacy lesson updates position to that lesson. An
omitted **or null** legacy lesson preserves position, since neither distinguishes assessment
from a missing value. If both fields are supplied, accept only a matching lesson ID in both,
or an assessment position with a null legacy lesson; otherwise reject 422. Derive the response
alias from position (lesson ID, otherwise
null). `lesson_id` used to scope a tutorial action/practice payload does not change position.

For old stored progress without `position`, normalize reads deterministically: a valid
non-null old lesson wins; otherwise existing assessment drafts/attempts select the assessment
landing view; otherwise select the first frozen lesson. This is a compatibility fallback,
not recovered knowledge of a historical screen. Preserve all practice, requirements, drafts,
attempts, help counts and completion timestamps; persist the normalized state on the next
valid mutation, without rewriting history or allocating again. A completed tutorial remains
completed regardless of the help view. Never turn old checkbox training into a v2 pass.

Test position-only save/reload, navigation without further actions, assessment landing/draft
and focused-question resume, omission after a non-position mutation, explicit invalid null,
legacy request compatibility, changed-body replay, rapid queued saves, stale revision,
process restart and PostgreSQL recovery. State and mutation receipts must agree. Backend
tests may submit actions through the API; the joint browser journey must perform them in the UI.

## B13 / R04 — Versioned timing export with explicit uncertainty

Client monotonic clocks belong to individual page instances. A sum of valid page observations
is **page-seconds**, not the union of elapsed intervals and not active participant time. A
multiple-page count alone does not prove concurrency, but current instrumentation cannot
prove disjointness either. Do not compare different clocks' raw offsets, subtract their sum
from elapsed time, or clamp an overcount to hide it. Do not implicitly pause hidden tabs:
external inspection is allowed.

### Version selection without rewriting frozen evidence

Introduce `exact-study-analysis/3` and `exact-study-csv/3` for the corrected derived exports.
Extend the existing admin create-export route with optional query parameter `analysis_schema`.
For v2 it accepts `exact-study-analysis/2` or `exact-study-analysis/3`; omission selects the
publication's frozen export version, preserving old-client compatibility. V1 continues its
v1 contract and rejects unsupported selectors. Unsupported combinations return 422.

New corrected v2 publications explicitly freeze `protocol_versions.export` to
`exact-study-analysis/3`. Preserve parsing/default canonicalization and idempotent publication
of previously frozen v2/analysis-2 records; do not change an old default to 3 and thereby
change their hashes. Update new fixture builders to supply 3 explicitly. Existing synthetic
v2/analysis-2 fixtures remain available for compatibility tests.

Researchers can explicitly request a new analysis-3 export from an existing v2/analysis-2
publication. The export manifest records schema 3, source study revision, content hash and
`source_protocol_versions` copied unchanged from the publication. Preserve the existing
data-level frozen `protocol_versions`; it is not the derivation schema. Update revision-list
metadata/help to distinguish the frozen source export version from supported derived versions.
Fetching a saved export returns its original content/version/hash; CSV conversion dispatches
on that saved manifest. No automatic relabeling or regeneration of archived exports.

The corrected admin UI explicitly requests 3 for v2. Analysis-2 compatibility output is
historical semantics and must not be used as corrected elapsed-coverage evidence. Document
that limitation in the handoff and data-analysis guide; do not alter archived dictionaries.

### Required analysis-3 observation summary

For case timing, add the following fields alongside `raw_elapsed_seconds`:

```json
{
  "page_observation_seconds": 120,
  "per_page_observation_seconds": [
    { "page_instance_id": "page-a", "seconds": 60 },
    { "page_instance_id": "page-b", "seconds": 60 }
  ],
  "page_instance_count": 2,
  "coverage_status": "not_established",
  "coverage_reason": "multiple_page_clocks",
  "unique_elapsed_coverage_seconds": null,
  "unobserved_elapsed_seconds": null,
  "active_duration_known": false
}
```

Use this same observation-summary shape for `tutorial_timing` at session level and
`consultation_timing` per case, grouping only the relevant stage/case. For this extension,
coverage remains **not established even for one page**: its monotonic samples have no
validated mapping to the server's ready/submit receipt interval. Do not invent such a mapping
from network receipt time. Reason is `no_eligible_observations` for zero contributing pages,
`single_page_clock_unmapped` for one and `multiple_page_clocks` for more than one. Future
calibrated coverage needs a separately specified contract; this assignment does not require
a cross-tab leader election, clock synchronization service or attention measurement.

Within each page/stage/case, use accepted, bounded, deduplicated, non-overlapping intervals
under existing validation rules. Exclude `availability: unavailable` intervals and retain
them with their reasons in raw records. Derive contributing page count and totals from that
same eligible set. A duplicate idempotent upload contributes once. Preserve legacy
`observed_segment_seconds`, `tutorial_observed_seconds` and `consultation_observed_seconds`
as documented aliases of the corresponding raw page sum in schema 3, never as unique coverage.
The new named fields and dictionary are authoritative for interpretation.

`raw_elapsed_seconds` remains first accepted usable-content **server receipt** to ranking
submission for that presentation, including breaks and gaps. It is null when either endpoint
is absent; a new page must not reset its start. No tutorial/consultation time is added to
scored case time. No readiness reset or export change alters ranking-quality denominators.

Provide JSON and CSV dictionary entries for units, aliases, null meaning, eligibility,
clock scope and source/derivation versions. CSV nested summaries retain JSON null; top-level
nullable scalar cells are empty with their meaning documented, never zero. Keep checksums,
deterministic archives, anonymization and formula-injection protections. Ensure schema-3 CSV
still includes tutorial attempts/outcomes and consultation drafts; the existing `== /2`
branches must not silently drop these tables when adding a version.

### Timing oracles (seconds)

| Scenario | Raw elapsed | Eligible observations | Page sum | Coverage / unobserved |
|---|---:|---|---:|---|
| One page | 100 | A: 0–60 | 60 | Both null; single page unmapped |
| Two independent pages | 100 | A: 0–60, B: 0–60 | 120 | Both null; multiple clocks, not zero missing time |
| Two possibly sequential pages | 100 | A: 0–20, B: 0–30 | 50 | Both null; do not infer overlap or disjointness |
| No eligible observations | 100 | none | 0 | Both null; absence of observations is not zero task time |
| Retry plus unavailable late interval | 100 | A: 0–60, identical retry, unavailable A: 60–80 | 60 | Both null; unavailable record retained, retry deduplicated |
| Unsubmitted case | null | A: 0–60 | 60 | Both null; answer remains missing |

Exercise the same rules for tutorial and consultation, and cross-page events after navigation,
reconnect and process restart. Compare exported JSON with every CSV representation. A sum
larger than elapsed is permissible raw evidence; its semantics and uncertainty are mandatory.

## B14 / R02 — Support the real readiness boundary

The bounded required-content set and UI state machine are defined in 19. Supply accurate
capabilities and terminal availability states for each admitted component. A timeout, 5xx,
authorization failure or malformed response must never be translated into successful absence,
an empty prepared packet or a changed condition. API error/absence fixtures must be distinct.

Retain pre-ready and late-stage telemetry handling, session/presentation binding, event and
segment idempotency and server receipt timestamps. `case_ready` remains a client assertion
that content is usable, not server proof of attention or comprehension. Test that duplicate
ready events from reloads/tabs do not restart the exported presentation interval, and that
unavailable intervals remain excluded. Do not introduce a server assertion that the UI
rendered merely because an API read succeeded. Coordinate any necessary typed event change
through the generated contract before frontend implementation.

## Backend completion checklist

- Satisfy J01/J02/J06/J08/J09 at the API/database boundary; provide delayed/error fixtures
  for J04/J05 and full-context data for J03. Include SQLite and PostgreSQL, restart/restore,
  current-session binding, denial and strict-model projection checks.
- Regenerate model schemas/OpenAPI and run the specification validator. Add regressions
  that fail for the actual old behaviors, not just newly added fields in hand-authored JSON.
- Update the backend contract guide and create S1's handoff/receipt with exact commits,
  hashes, launch instructions and unresolved gates. Historical dated receipts remain intact.
- Declare **backend verified / frontend and joint verification pending**. A passing backend
  suite does not close UI readiness, navigation focus or the fresh integrated browser journey.
