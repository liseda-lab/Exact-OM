# Frontend corrective assignment

**Initial assignment: 2026-10-02; implementations delivered, integration findings remain.**
For the next work read [17](17-integration-corrective-programme.md) and
[19](19-frontend-integration-corrections.md) first, after the backend handoff from 18.
The original F0 workflow design phase below is historical and must not be restarted as a gate.
Read [14](14-corrective-programme.md) for the preserved product/study decisions;
its protocol decisions and issue register are normative. Coordinate contract changes with
[16](16-backend-corrections.md). Own the interaction design and shared workspace across
the main exploration app, public demo, tutorial and explanation-study condition.

## F0. First handoff: design against actual data

Inspect `explanations_visualizer/src/components/{explore,study,owl,common}`, the study
types/queue/telemetry and existing e2e tests. Start from the current implementation;
do not replace working persistence/security code with a mock product. Run a known
synthetic package and inspect a real prepared package when available. Clearly distinguish
ontology excerpts, synthetic matcher scores, generated text stubs and real provider output.

Before backend integration, deliver:

- A reusable workspace/component map and state ownership diagram, including both conditions.
- Desktop/narrow/text-enlarged layouts for a long source, long candidate and a partially
  ranked answer; loading, missing, unsupported, error, conflict and disconnected variants.
- The complete tool-neutral setup, interactive tutorial, five-item assessment and
  per-case multiple-method reporting flow with exact proposed participant wording.
- A table of backend operations needed by each control, scopes/cursors/availability
  states, and example success/error data. Mark current versus proposed operations.
- A parity checklist for the same case in exploration and study. Freeze these data needs
  with the backend owner before integrating new routes. Implementation may continue on
  independent shared components and existing-route defects in parallel.

No approval round is required for routine design choices already authorized. Report
material protocol questions explicitly; use the decisions in 14 instead of reopening
Protégé choice, per-case reporting or method-switching as unanswered questions.

## F1. Shared workspace and state model — C01, C02, C18, C19

Extract presentational entity, meaning, comparison, hierarchy, fact-inspector, evidence
list/graph and tab primitives from fetch-bound exploration components. Use an explicit
data-source interface for exploration, participant study and synthetic tutorial. The
study adapter must use only participant-safe routes/resources. Do not use an iframe,
fetch exploration endpoints and hide content, or copy the main components into another
study-specific rendering tree. Study ranking and progress remain shell responsibilities.

Keep separate state for:

1. Server-owned case/presentation, frozen initial candidate order, condition and capabilities.
2. Inspected candidate, source/target browsing focus and independent navigation history.
3. Participant answer, submission acknowledgement and pending mutations.
4. Graph view/selection, open fact, active detail view and scroll position.

Inspecting/navigating is never ranking; changing a rank never resets inspection. Changing
candidate retains source navigation; switching case resets only case-bound state. A cache
key includes publication/package, policy, typed ontology identity and resource version;
no content from a previous session, bundle, case or restricted condition may flash during
loading. Do not serialize secrets, answers or private invitation fragments into navigation URLs.

Parity requirements in the explanation condition: equivalent source/target meaning,
original and generated distinction, claim citations, parent/child search/navigation,
multiple inheritance, asserted/structural/inferred labels, pagination/counts, original
axioms, evidence-list semantics, graph inspection and honest missingness. Final ratings
must describe the frozen components actually available. Do not add a decision-trace
study component silently: it remains an optional main-app view unless separately admitted.

Keep baseline identities, scores, initial ordering, ranking and downloads equivalent to
the explanation condition. No integrated explanation content, shortcuts or hidden fetched
data in baseline. Tutorial resources stay explicitly synthetic even when opened as help.
Required explanations failing to load must keep the failure/retry state and prevent a
case being timed as usable; do not quietly turn an explanation case into baseline.

## F2. Candidate selection, comparison and answer layout — C06

Use a persistent or readily reachable candidate/ranking area and aligned source/target
meaning cards. Put the question of equivalence and balanced comparison ahead of secondary
metrics. Avoid placing all candidate controls below an arbitrarily long source card.
Use one primary reading scroll where possible; secondary scroll regions need clear bounds,
keyboard access and a visible purpose. Preserve reading position during autosave.

For each row distinguish initial position, matcher score, inspected status and participant
rank. A row can be inspected without being ranked. Keep accessible add/remove/move/undo,
partial ranking, explicit Keep initial order, None and Insufficient information. No ties,
auto-filled omitted candidates or inference from an empty draft. Retain confirmation of
server receipt separately from a locally queued submit; prevent double submission.

At 1280×720 with default text, participants must discover and operate candidate selection
and answer status without first reading a whole long source card. Do not achieve this by
shrinking text or hiding source information. On narrow/enlarged layouts use a labeled,
reachable answer region or explicit switch preserving state, with no required hover.

## F3. Fact inspection, faithful readings and completeness — C03, C04, C18

Replace bare fragment navigation with a shared fact resolver/inspector. Every displayed
claim citation must open the exact admitted original record; include typed subject and
ontology context where an axiom has more than one focal subject. Deduplicating labels or
synonyms for display must not remove their inspectability. Resolve facts outside the
currently loaded page on demand. For unavailable/filtered/unresolved originals, explain
that specific status; never leave a dead link or invent supporting text.

Expose citations for generated entity profiles and comparisons in study and exploration.
The Evidence tab must offer original-axiom inspection even when no natural-language or
graph form exists. Reuse `AxiomBlock`/typed-AST formatting through the adapter; no client
OWL parsing, equivalence inference or invented verbs. Preserve quantifiers, cardinality,
negation, conjunction/disjunction, direction, inheritance/onset qualifiers and full IRIs.

Render backend interpretation for each original/projected/derived/inferred/generated
record; remove the hardcoded assertion that every evidence item is projected from an
asserted fact. Preserve capabilities, limitation reasons, original availability and
selection/truncation counts when normalizing study resources. A missing readable form
gets an original/fallback action and an honest explanation, not an empty evidence row.
An absent fact, unprepared page, policy restriction, unresolved import, unsupported
constructor and service failure are different states. One-sided information is not an
incompatibility. Regression fixtures must include the previous same-attribute conflict
and inheritance/onset complaints; do not invent a biomedical correction from UI feedback.

## F4. Graph behavior and visual semantics — C07, C08, C18

Memoize semantic bundles and separate graph creation from incremental labels/theme/answer
updates. Persist pan/zoom/focus for a pair while ranks save, questionnaires/state refresh,
tabs change and responsive layouts switch. Reset explicitly or when the actual scoped
pair changes; no retained nodes from a prior case. Give Fit and Reset distinct, documented
behavior. Bound expansion, disclose limits and retain undo/remove; typed identities must
survive expansion, including property exploration in the main app.

Remove the invalid Cytoscape `double` style. Choose supported patterns/width/labels or
implement a real double-stroke representation with a tested accessible equivalent. Legend,
rendered edges and evidence descriptions must agree. Distinguish original assertion,
structural relation, projected matcher feature and channel-level comparison. A channel
bridge is not a paired-feature correspondence, inferred axiom or proof of equivalence.
Parallel edges remain distinguishable; unsupported forms remain in the evidence list.

Initial labels must remain readable: prefer bounded layout, sensible initial focus and
pan over fitting every label to illegible size. No geometry/selection changes on hover,
wheel interception over the page, unexpected recentering or loss of source/candidate
identity. Numeric matcher weights are secondary details, not semantic cardinalities.

## F5. Tool-neutral setup and resource access — C12, C13, C19

Replace Protégé installation/opening checkboxes, title, required version field and
desktop-with-Protégé notices. Audit welcome, setup, baseline copy, hierarchy footer,
phone notice, pause/gap copy, help, tutorial, downloads and fixture instructions. A device
recommendation may describe screen space; it must not imply a software requirement.

Setup acknowledges task instructions, availability of the supplied resources and
understanding that external inspection is optional. Offer download/open help and an
optional report of intended/familiar methods, with no mandatory selection or use. An
explicit resource-access problem offers retry/help/save-and-return, never false success
or a condition switch. Merely choosing no external inspection must not block continuation.

Display ontology human name, source/target role, version, format, size, full-hash copy and
information-scope notice. Keep source/candidate IRI copy and the same download entry points
available in both conditions, including explanation cases. Do not require that a resource
be downloaded to claim setup or tutorial completion. Never monitor external applications.

## F6. Interactive tutorial and comprehension assessment — C10, C11, C14

Implement all lessons and five assessment items in 14 using the production shared
workspace with backend-frozen synthetic resources. The existing four shape lessons can
remain as controls lessons, but cannot be the only tutorial. Do not use fallback examples
to certify a new production publication whose required tutorial is missing or incompatible;
show a setup/publication error. Legacy publications remain explicitly versioned.

Use short instructions tied to actual controls and a visible lesson checklist. Prefer
normal workspace actions over fragile modal tours that cover the target. Provide Back,
Retry, Help and Save and return. Preserve inspection, practice responses, completed
lessons and assessment attempts after reload/pause/new-device return. Show saving and
acknowledged completion honestly. Resuming an incomplete lesson does not erase prior work.

Core interactive evidence includes candidate inspection, real context navigation,
citation/original inspection, graph-or-list equivalent inspection and ranking/reporting
actions. Accommodations must be semantically equivalent, not a demand for dragging,
pointer precision or installing an external program. An opened panel alone does not
establish understanding. Keep observable action evidence separate from assessed answers.

Assessment UI sends question/answer IDs to the server, displays specific returned feedback,
allows repeated supported attempts and announces result changes accessibly. Do not grade
by answer-position, option order, locally invented rules, tutorial checkbox count or a
generic client `passed` flag. Show how to revisit the relevant lesson. No scored-case
answer feedback. Do not label failed first attempts as lack of expertise or remove users
silently. Allocation starts only after the server acknowledges completion and background
requirements; updating a setup checkbox must not allocate cases.

Use the same tutorial and assessment sequence for everyone before condition assignment.
Reopening synthetic help within a case preserves its pending answer and timer. Keep
practice responses/events out of scored metrics and candidate-inspected statistics.

## F7. Per-case consultation and ordered questionnaires — C05, C13, C15

After each committed ranking, show the source/case identity and explain the answer is
saved. Ask actual external inspection Yes/No and all methods used for that case; allow
any combination and an Other route, optional tool names and optional resource scope.
Offer optional reuse of a prior selection only after an explicit participant action;
display what was copied and require normal confirmation. No mandatory per-candidate report.

Autosave incomplete consultation drafts through the participant outbox. A reload restores
unanswered/partly answered states without advancing or editing the ranking. Final Save
validates, commits and advances exactly once. Yes→No removes irrelevant methods/details
from the current answer; keep history only under the declared revision policy. A 409
refreshes state and explains the conflict without silently overwriting another tab.

Consume explicit question, option and matrix-row order from the new contract; never rely
on dictionary key order or sorting English labels. Keep ordinal scales in meaningful order
and special responses separate. Preserve stable codes and exclusive choices, conditional
visibility/pruning, field validation, optional text, matrix/card equivalence and accessible
focus to errors. Historical v1 forms use their versioned compatibility presentation,
not guessed reordering of saved responses. Freeze final component ratings with the actual
capability matrix, distinguishing original context from generated prose.

## F8. Events, timing and persistence — C14, C15, C16, C23, C26

Integrate new mutation routes with the persisted session-bound outbox and its route
allowlist, coalescing, idempotency and expected revisions. Coalesce draft updates only;
never coalesce/reorder an assessment attempt, completion or consultation submit across
another operation. Preserve retry identity, bounded storage, sign-out/new-invitation cleanup
and cross-tab rejection. A local pending action is not a durable server acknowledgement.

Replace the old `changesStage` test of Protégé setup booleans with explicit operation
semantics: setup submit, tutorial complete, ranking submit and final consultation save can
advance; setup/tutorial/consultation drafts and individual assessment attempts cannot.
Flush/close timing for the former and resume the correct stage if a transition fails.
Do not treat the new consultation-draft path as a final stage transition just because
it contains the word consultation.

Emit actual actions: a tab open is not hierarchy expansion; hierarchy collapse/expand
names match actual branches; definition/axiom inspection and graph/list selection identify
the frozen resource. Extend practice/help event scope with the backend; do not fabricate
scored case IDs to reuse the existing emitter. Save assessment outcomes through authoritative
mutations, not optional telemetry. No screenshots, entered free text, URLs or external app
monitoring in research events.

Case readiness requires usable required content. Stop case timing at submit intent as
specified, distinguish intent/receipt latency, include prior external work and never pause
automatically on invisibility. Explicit pauses and unknown gaps remain distinguishable.
Optional telemetry must not block answer submission indefinitely; retry or disclose missing
segments under the backend contract. Tutorial/help/consultation durations must not be
misclassified as an extra scored case. Replace Protégé-only gap wording with external work.

## F9. Main-app and researcher completeness — C20, C21, C22, C27

Do not regress typed property search, stale-request cancellation, source retention,
rank/score provenance, run discovery errors or return-to-comparison navigation. A loaded
subset sorted by rank is not the complete top-ranked set. Display paging/limit basis and
provide continuation; never relabel subset position as original rank. Decision trace remains
separate from semantic evidence, with unknown/unrecorded states intact.

Library: verify preview, compatibility, corruption/cancel/progress, atomic import, selecting
two bundles, reopening offline and preserving a usable selection on failure. With backend
support, show useful bundle metadata and offer confirmed removal of a library copy only;
explain active-copy behavior and preserve original run/source files. Browser-local review
decisions must say where they are stored and how they can be lost; do not imply server sync.
If persistent reviews are later implemented, require a separate versioned contract.

Researcher UI: inspect revision selection and add a backend-supported selector when available.
Review publication validation errors, read-only frozen summaries, tutorial/policy versions,
test-session marking, progress stages, invitations/reissue/revoke, export scope and closing.
Keep private links/token state cleared across sign-out and revision changes; no tokens in
URLs/storage/screenshots. Public/demo/study profiles retain their existing boundaries.

## F10. Acceptance and evidence

Each test below is a scenario, not a test that mirrors implementation internals. Add them
to the existing e2e/state suites where appropriate. Retain the earlier regression suite.

| IDs | Required acceptance scenario |
|---|---|
| C01, C02, C18, C19 | Same prepared case in main app and explanation study: inspect equal permitted facts, browse both sides beyond first page, return, copy IRIs/download resources; baseline direct requests remain denied. |
| C03, C04 | Enumerate and activate all profile/comparison citations, including labels, duplicate synonyms, shared axioms and next-page facts. Verify identity, typed origin, unsupported fallback and keyboard return. |
| C05 | Render deliberately alphabetically scrambled JSON keys with a declared order; ordinal scales and matrix rows follow declared order before/after reload. |
| C06 | Long source/target at 1280×720: choose a candidate, compare, edit a partial answer and submit without losing context; record a human usability observation as well as geometry checks. |
| C07, C08 | Zoom/pan/select evidence, add/move a rank, wait for autosave, switch tabs/theme/text size and return; preserve view. Change case and ensure no stale nodes. Compare every legend type to rendered edges; no invalid-style warning. |
| C09, C24 | Keyboard-only full case/tutorial/report, roving tabs with arrows/Home/End, labeled panels, fact dialog focus return, combobox/list navigation, non-drag ranking; automated axe plus manual screen-reader checks. |
| C10, C11, C14 | Complete lessons and five questions with an intentional wrong answer, targeted feedback and retry; reload midway and resume on a second device/session exchange. No scored allocation until acknowledgement; identical training for both future orders. |
| C12, C13 | Complete setup without Protégé or any external tool; use two methods in one case, none in the next and a different method later. No lock or inherited answer; downloads are equally accessible. |
| C15, C23 | Reload an incomplete consultation; simulate offline final save and a two-tab conflict; ranking is unchanged and next case advances once after acknowledgement. |
| C16 | Open Hierarchy without expanding: no expand event. Expand, inspect axiom, open tutorial help; scopes/types match. Hide tab, pause, disconnect and submit; durations/gaps remain truthful. |
| C20, C22 | More than 500 candidates and two bundles; honest rank scope/continuation, import failure/cancel and safe copy removal; no loss of current usable bundle/review attribution. |
| C21 | Researcher publish/test-invite/progress/export/close journey with validation failures, empty revisions, unauthorized access, revoke and sign-out privacy checks. |
| C25, C26, C27 | New and legacy publications, runtime/version mismatch, real package plus PostgreSQL/HTTPS; preserve prior regressions and clearly separate synthetic/real evidence. |

For C24 test 320/360/390/768/1024/1280/1440/2560 CSS px where applicable, landscape,
200% text, light/dark/system themes, reduced motion, long IRIs/labels, sparse and dense
cases. No document-wide horizontal scrolling or clipping of primary controls. A 2D graph
may pan inside its own region. Do not hide required information behind hover or a phone-only
block; any device constraint must be a frozen research choice, not a software-install claim.

Deliver source changes, component/contract map, reviewed tutorial and participant copy,
test commands/results with build/package IDs, representative redacted screenshots, manual
accessibility/usability notes, issue dispositions and a backend integration handoff. Do not
call the tool ready from a build, a welcome-screen axe test or synthetic screenshots alone.
