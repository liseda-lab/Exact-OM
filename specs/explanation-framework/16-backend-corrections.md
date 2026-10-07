# Backend corrective assignment

> **Current follow-up:** review at `2d9715b` reopened discovery, readiness, tutorial-position
> and timing-export issues. Start with [17](17-integration-corrective-programme.md) and
> [18](18-backend-integration-corrections.md), delivering backend contracts before frontend
> integration. The delivered work and historical evidence below do not close those findings.

**Implemented 2026-10-03; release acceptance remains partly gated.** See the
[dated backend handoff](../../docs/verification/explanation-backend-corrections-20261003.md)
and [v2 contract](../../docs/verification/explanation-study-v2-backend-contract.md).
Read [14](14-corrective-programme.md) for
protocol decisions and priorities, then [15](15-frontend-corrections.md) for interaction
requirements. This assignment supplies the shared workspace safely, removes the mandatory
Protégé gate, makes training/reporting durable and preserves study validity and recovery.

## B0. Contract freeze and version boundary — C01, C02, C26

Frontend leads the initial workflow/component design. Backend can start verified isolated
fixes immediately, but freeze schemas, route behavior and examples with that design before
full integration. The interfaces below describe the corrective requirements; exact implemented
shapes are frozen in `protocol/study.runtime-openapi.json`, the publication/resource/tutorial
runtime schemas and the linked v2 contract. These snapshots have been regenerated from code.

Use `exact-study/2.0` for new publications: changed setup gating, ordered forms, durable
tutorial/assessment and consultation drafts are semantic changes, not a cosmetic UI patch.
Version tutorial content, assessment, questionnaires, information instructions, resource
policy, software compatibility and export schema independently within the frozen definition.
The exploration read contract remains `exact-explain/1.0` unless its wire primitives actually
change. Prefer additive exploration endpoints and existing typed contracts.

Keep existing v1 publication canonical bytes, hashes, response codes and submitted records
unchanged. Explicitly support reading/resuming legacy sessions through a version adapter
or a pinned legacy deployment; document the choice and test it. Do not relabel legacy
checkbox completion as an assessment pass, rewrite historical question order/results,
or make an active session restart training silently. Newly published corrected studies
must use v2 and cannot omit training under the legacy fallback. Administration must identify
which protocol each revision uses. Any migration of active human sessions needs a separately
declared study procedure; this assignment does not authorize it.

Deliver model/schema examples, OpenAPI, error/missingness examples, migration notes,
fixture-generation commands and a compatibility matrix. Bind each frontend adapter to
an explicit supported version; mismatches fail visibly before collecting new responses.

## B1. Participant-safe exploration workspace — C01, C02, C03, C04, C18

Build a read facade over the frozen policy-filtered context and explanation stores, not
a second ontology engine. Query prebuilt SQLite/indexed resources; no runtime OWL parsing,
import fetching, model loading, generation or unrestricted filesystem paths. Keep the
minimal study deployment independent of matcher/GPU/native-parser dependencies.

Implemented route family: `/api/v1/study/workspace/{scope_id}/…`, with operations for:

| Operation | Required behavior |
|---|---|
| capabilities / scope | Publication, policy and ontology versions, current focal entities, admitted components, completeness/reasons, supported hierarchy bases and page limits. |
| entities/search | Bounded search in admitted frozen ontologies; typed identity, labels, ambiguity and keyset continuation. |
| entities/context | Candidate-independent original facts by typed entity and category, stable counts/cursors and availability. |
| hierarchy | Parent/child navigation with multiple inheritance, equivalents, basis and counts; do not infer reasoner results from structural traversal. |
| labels | Bounded policy-filtered labels for focal and referenced IRIs with unavailable/unresolved states. |
| facts / axioms | Resolve admitted citation IDs plus ontology/typed subject scope, lossless typed/original data, interpretation and original-format availability. |
| explanations | Frozen entity profiles and comparisons admitted for the active case; claims with citations/scoped entities and generation provenance. No arbitrary provider dispatch. |
| evidence | Recorded matcher evidence for admitted active-case pairs, with fact links, channel, interpretation, selection accounting and unsupported graph-form status. |

Freeze exact operation suffixes and request/response models at B0. Reuse main read shapes
where they meet study policy; do not return a lossy hand-maintained substitute. `scope_id`
is an opaque locator, never authorization by possession. Every request checks participant,
session generation, publication, stage, current presentation, condition and resource policy.
Only the active explanation case gets its case workspace. Baseline case workspace reads
are denied server-side, including labels/facts/cursors guessed from an earlier case.

Tutorial scopes are separately identified, synthetic and usable during training or help.
They must never point at scored cases, their explanations or private keys. Opening help
in baseline cannot authorize its scored-case explanation. Ontology search may reach any
entity in the admitted frozen ontology scope, including an entity that occurs elsewhere
in the case bank; that does **not** authorize other cases' candidate sets, generated
comparisons, assignments or metadata. Avoid conflating ontology navigation with case access.

Bind cursors/cache entries to policy, publication, ontology, typed query and allowed scope;
reject cross-scope reuse. Apply query/page/IRI limits and efficient indexes. Do not load a
whole ontology or all explanations per request. Keep private no-store/referrer/CSP rules,
revocation and cross-session denial. A disallowed field is not merely hidden by the UI.

Publish complete declared context coverage or report exact bounded coverage with a usable
continuation mechanism. The current `limit=20` first-page snapshot is not a hierarchy
browser. Include source and all five candidates, references, and data required to navigate
their allowed context. Policy-filter context indexes, downloads and generation inputs
consistently. Publication rejects a promised component that cannot work with its resources;
do not disguise missing prerequisites as an ordinary empty result.

## B2. Tool-neutral setup and resource metadata — C12, C19, C26

Replace required `protege_installed`, `protege_version` and Protégé-specific openability
semantics for v2. No tool ID/version is a prerequisite. Define a setup draft with:

- `setup_version` and the normal idempotency/session/expected-revision envelope.
- `instructions_acknowledged`: boolean.
- `external_inspection_optional_understood`: boolean.
- `resource_access`: `not_checked | available | needs_help`; this describes access to the
  supplied resources, not downloading, opening them in a named app or actually using them.
- `submitted`: boolean, false for draft. Optional broad preferred methods/tool labels
  are descriptive only and cannot satisfy or block readiness.

An explicit setup submit requires the two acknowledgements and `resource_access=available`.
Verify the server actually has admitted downloadable resources, expose help when access
fails, and allow the participant to acknowledge availability without using an external
tool. Draft saves never auto-advance when the last checkbox becomes true. `needs_help`
offers retry/help/save-and-return, not forced false confirmation or a condition change.
Setup success advances to background; case allocation remains gated on B3 completion.
Drop obsolete installation/openability claims from new owner-generated instructions too.

Each ontology download has asset ID, human title, source/target/both role as appropriate,
ontology version, format, bytes/hash, license/treatment and policy information. Both
conditions receive identical permitted downloads. Publication validates roles against
the case's ontologies, not filename conventions. Unknown historical roles stay unknown
in the legacy adapter. Optional current-condition/header metadata must agree with
`cases/current` and expose no future assignment or private case kind.

## B3. Frozen tutorial, interactive progress and assessment — C10, C11, C14

Add `TutorialDefinition` and `TutorialReceipt` to the v2 publication/state. Definitions
contain stable tutorial/version/hash, ordered lesson IDs and typed requirements, synthetic
resource scope, five assessment items, feedback, completion policy and compatible build.
Practice entities, candidate IDs, assets and content must be disjoint from scored cases
and source-transfer groups; validate the whole resource graph, not just top-level IDs.
Required lessons follow 14. Publish no real answer key or information that previews a
scored case. No runtime generation of new lessons/questions.

Assessment answers/feedback are tutorial material separate from researcher CaseKeys.
Keep grading rules server-side so completion is consistent, although synthetic answer
feedback can be shown for learning. Validate author-defined codes, step/question IDs,
referenced facts and accessibility-equivalent task paths at publication. Reject a
production v2 publication missing required lessons/questions/resources; empty practice
or a built-in client fallback is insufficient.

Implemented mutation routes, using existing transactional mutation conventions:

| Route | Semantics |
|---|---|
| `PUT /study/setup` | v2 setup draft/explicit submit; version-dispatched as defined above. |
| `PUT /study/tutorial/progress` | Save lesson position, typed practice answer, partial assessment response and completed interaction predicates; bounded, version-scoped, no scored IDs. Draft coalescing only within the same lesson or assessment item. |
| `POST /study/tutorial/assessment` | Submit one item attempt identified by question/attempt IDs; validate code and compute correctness/feedback, record first and subsequent attempts. Not coalescible. |
| `POST /study/tutorial/complete` | Verify acknowledged mandatory lessons and eventual pass of every core item, then allocate atomically if other prerequisites are satisfied. |

Expose resumable progress in StudyState: current lesson, saved practice answer, partial
assessment answers, completed requirements, attempt receipts, outstanding items,
help/completion status and version. An assessment draft never counts as an attempt or pass.
Separate the latest draft from append-only attempts and first/final outcomes. Permit
unlimited pedagogical retries, while bounding payloads and applying ordinary request-rate
limits; do not conflate rate limiting with an assessment failure cutoff. An incorrect
attempt does not allocate, erase progress, close the link or change condition.

Server completion checks validate typed action predicates and answers rather than a
client `passed=true` or `completed_tutorial_steps=[…]`. UI action evidence is still
client-reported and cannot prove human attention; do not advertise tamper-proof learning
or rely on optional click telemetry as the authoritative gate. Assessment results are
server-evaluated. Duplicate completion/attempt IDs return their previous result; stale
revision/session/version and impossible step IDs fail without corrupting progress.

All participants complete the same training before allocation, including baseline and
explanation practice. Completion is durable across refresh, device changes, process
restart and invitation reissue under the existing policy. Pausing/reopening completed
synthetic help does not reset a pass, create another assignment or append scored answers.
Export first/final responses, attempts, eventual completion and separate tutorial duration
as descriptive preparation data. Never interpret them as a validated expertise score.

## B4. Consultation granularity, drafts and reporting — C13, C15

Keep the existing unit: one source plus five candidates, keyed by session/case/presentation
and frozen consultation-form version. Reports follow ranking submission in both conditions.
Do not introduce a study-wide method lock or mandatory candidate-level reports. A participant
can use multiple methods and change methods on the next case; setup/background does not
determine a report. Methods apply to the set of candidates considered, not necessarily all
five individually. Downloads, events and visibility must never synthesize an answer.

Retain existing stable codes `protege`, `other_editor`, `plain_files`, `other_resource`.
For v2 add `queries_scripts`, `reasoner`, `other_method`, with clear non-exclusive labels.
Other covers inspection approaches not named in the list. Optional tool-name fields have
bounded length and no request for identities/URLs. Optional `resource_scope` values:
`supplied_only | different_or_additional | unsure`; null means unanswered, not supplied-only.
Do not demand a scope answer merely because a participant used an unfamiliar tool.

Add `PUT /study/cases/{case_id}/consultation/draft` for partial state and preserve
`PUT /study/cases/{case_id}/consultation` as final commit/advance. Bind both to the current
presentation (explicit in v2 payload). Draft allows `consulted_external_ontologies=null`
and incomplete Yes without methods; final requires Yes with one or more unique allowed
methods, or No with none. Reject contradictions/unknown codes and non-current presentation.
On No, irrelevant method/name/scope fields are cleared from current canonical answers;
backend and frontend must agree whether stale non-null values are rejected or normalized
(freeze one behavior and fixtures at B0). Preserve revision history under the declared policy.

Draft saves do not advance, modify the committed ranking or count as completed consultation.
Final save advances exactly once. Resume exposes current draft separately from a committed
receipt; distinguish unanswered, draft, submitted-No and submitted-Yes. Revisions and
retry identity follow the existing queue; stale writes cannot overwrite another tab.
Timing after ranking commit is consultation time, never appended to ranking-task time.

Exports bind codes and labels to versions, preserve method combinations and optional
resource-scope missingness, and redact unnecessary identifying free text. Include typed
optional names only in an explicitly authorized researcher export if required; default
analysis exports remain de-identified. Do not infer candidate-level attribution, durations,
compliance or causality from a case-level method set. Report resource-scope deviations under
a frozen analysis rule; no automatic post-hoc exclusion based on outcome or tool choice.

## B5. Explicit questionnaire ordering — C05

Keep the canonical serializer's sorted keys for hashing/idempotency. Add an explicit ordered
array of option codes to each v2 question, and ordered matrix-row codes where applicable;
question order itself remains an array. Existing code→label maps can remain for lookup.
Validate that each order is a duplicate-free exact permutation of its keys, including
special choices. Empty/unknown/missing orders fail v2 publication, not arbitrary UI sorting.

Freeze ordinal ordering for years, familiarity, confidence, helpfulness and effort; place
non-substantive options deliberately. Preserve the declared order through storage,
canonicalization, HTTP JSON, client rendering, export and schema generation. Stable answer
codes remain independent of position/label/localization. Version legacy forms explicitly;
do not reinterpret historical answers. Keep conditional/pruned/skipped/not-answered
semantics and optional text behavior, including matrix rows and exclusive options.

## B6. Evidence fidelity, paging and portability — C03, C04, C17, C18, C20

Preserve the fixed subject-scoped identity of shared original axioms and comparison packet
subjects. Resolve fact citations outside first context pages and for original label facts
not displayed in a card. Preserve fact origin, selection/projection provenance, missingness,
capabilities and unsupported-rendering reasons. Referenced property/filler labels must be
admitted separately and policy-filtered; unavailable labels remain honestly unavailable.
Do not replace declared source facts with inferred relations during normalization.

Typed AST/original payloads remain the lossless source for rendering. Document the current
canonical-byte/base64 versus human-readable syntax distinction; never label base64 as a
human-readable axiom. Empty `display`/`semantic_terms` may be a documented capability,
but linked source facts must provide a working fallback. No fabricated feature pairing
for channel-level graph bridges. Inferred hierarchy is available only with its real recorded
reasoner/provenance; full inference is not a prerequisite for this correction.

Enable URI interpretation explicitly on the portable export output connection before
attaching `file:…?mode=ro&immutable=1` sources. Validate read-only source behavior on SQLite
builds with different default URI settings, paths containing spaces/non-ASCII characters,
relocation and concurrent failure cleanup. Do not resolve this by copying unfiltered bytes
or modifying source context databases.

For candidate continuation, either offer bounded rank-ordered paging with a stable tie-break
and query-bound cursor, or retain pair-ID paging with explicit order/scope metadata and
usable continuation. Sorting a capped client subset must not claim global top-N rank order.
Preserve original candidate ranks, saved scores and provenance; never renumber production
ranks to suit UI layout. Study display positions remain frozen contiguous 1–5 separately.

## B7. Telemetry, timing and recovery — C16, C23

Add typed tutorial/help context to events without fake case/presentation IDs. Validate
allowed action types, scope, component IDs, sequence/page-instance IDs and build/version.
An open tab is not a hierarchy expansion; a link click is not reported external use.
Do not put free-text answers, tool URLs or external application observations in events.
Assessment/completion and consultation are authoritative mutations, independent of optional
event delivery. Maintain bounded retry/idempotency for both types.

Preserve readiness-based case timing, raw elapsed/observed segments, explicit breaks and
unknown gaps. Late/offline segments after a transition must either be safely admitted
against the recorded previous stage interval or explicitly reported as unavailable; do
not silently declare full timing coverage. Freeze that policy at B0 and test delayed
submit/receipt, reload and restart. Do not block answer acknowledgement indefinitely on
telemetry. Tutorial and consultation durations are separate; opening synthetic help while
solving a case remains part of its task time unless explicitly paused. Tab invisibility
continues to mean unknown/external activity, not automatically a break.

Retain expected revision, exact-payload idempotency, session identity headers, row-lock
allocation, cross-tab/session rejection, immutable first submission and independent
consultation progression. New draft routes do not permit editing submitted rankings or
escaping pause/closed/revoked/expired states. Include negative tests, not only happy paths.

## B8. Library and researcher support — C19, C21, C22

Expose bounded library metadata needed to identify a bundle without opening all data.
Add local-only deletion of an owned library copy for spec 10 acceptance, with clear active
selection behavior, confirmation in the UI and safe conflict handling during import/use.
Do not accept arbitrary deletion paths, follow symlinks outside the library or delete the
original run/input/export; reject deletion of a CLI-mounted external package as a library
operation. Preserve a usable current bundle on failed import/delete. Hosted profiles deny
these operations on the server regardless of UI controls.

Provide a bounded authenticated revision listing/summary if the frontend needs a selector,
including protocol/form/tutorial/software versions and lifecycle/test status. Participant
endpoints cannot list studies, invitations or keys. Preserve invitations/reissue/revoke,
closure and export controls, including removal of credentials/private links from responses
that do not need them. Frozen publication validation should explain actionable missing
training/policy/resources, without leaking secret values in errors/logs.

Exploration review decisions remain explicitly browser-local unless separately contracted;
do not claim missing ReviewDecision persistence is already implemented or make a new
database mandatory for local read-only exploration. Document its retention/export limits.

## B9. Required acceptance tests and handoff

| IDs | Required backend/integration acceptance |
|---|---|
| C01, C02 | Search/context/hierarchy beyond first page in a frozen case; same permitted facts as main app. Deny another session, baseline case, future/past case metadata, invalid cursor/policy and revoked cookie. Tutorial help remains synthetic. |
| C03, C04, C18 | Shared axioms under multiple subjects, hidden label citations, unsupported AST, inferred/projected provenance, filtered/unresolved/missing states and sparse context survive wire round trip and inspect correctly. |
| C05 | Explicit option/matrix ordering survives sorted canonical storage; reject duplicate/missing/unknown order codes; versioned exports preserve labels/codes/order. |
| C10, C11, C14 | Save part of training, restart service, resume, wrong-answer feedback/retry, duplicate attempt and stale revision. Forged checkbox indices/client pass alone cannot allocate; valid eventual pass allocates once. |
| C12, C13, C19 | Setup passes with no named tool or installation claim; downloads available in both conditions. Accept mixed methods and changes across cases; no external use remains valid. Validate versions/roles/scope without method lock. |
| C15, C23 | Incomplete consultation persists without advancing; No and mixed-method Yes commit exactly once. Wrong case/session, conflicting tab and offline retry do not alter committed rankings. |
| C16 | Correct action types and separate tutorial/help scope; external tab work remains included; timing retries/late gaps never create duplicate or invented time. |
| C17 | Portable export works with URI-default-off and URI-default-on SQLite, relocates and leaves source unchanged after success/failure. |
| C20 | More than 500 candidates with ties and non-rank pair IDs; honest ordering/continuation with no dropped or repeated candidates. |
| C21, C22 | Authenticated revision listing and exports; local library-copy removal cannot escape its root or remove original sources; hosted mutations denied; active/import conflicts safe. |
| C25, C26, C27 | Legacy and v2 publications coexist with immutable hashes; existing historical fixes retained; real-package and PostgreSQL/HTTPS recovery plus built-image/redeploy/restore evidence. |

Retain all current study/resource/preparation/serving security tests. Add schema-generated
client conformance and shared end-to-end tests with the frontend owner; C06–C09/C24 are
frontend-owned but backend integration must not prevent them. Run PostgreSQL concurrency,
crash-before/after-commit, migrations and restore against a disposable database. SQLite
test success is not production database evidence. Test cold start and minimal study image;
never deploy the loopback Origin-rewriting harness.

Deliver code/migrations, regenerated runtime OpenAPI/models, contract fixtures, versioned
tutorial/form examples, export dictionary, exact serving commands and redacted receipts
with build/package/policy versions. Update operational guides and the status ledger only
to the level demonstrated. Remaining parser/data-admission gates, real case adjudication,
owner information/consent, accessibility/usability, duration pilot and Render readiness
must remain explicit. No new participant publication or recruitment is implied by passing
the technical tests.
