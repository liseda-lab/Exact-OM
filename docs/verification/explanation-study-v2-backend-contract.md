# Corrected study protocol: backend version boundary

This document describes the implemented additive `exact-study/2.0` contracts. It does
not authorize recruitment, certify the duration pilot, or migrate active human sessions.
The schemas generated from `exact_inspect.study.models.Publish` and the HTTP OpenAPI are
authoritative for field constraints. The independent protocol versions are frozen in
`StudyDefinitionV2.protocol_versions`: setup, tutorial, assessment, forms, information,
resource policy, software, and analysis export.

## Compatibility and migration

The additive runtime extension is `integration_contract: "study-integration/1"` on fresh
authenticated v2 state. Frozen publications remain `exact-study/2.0`. Historical mutation
receipts retain their original shape on replay; refresh state after replay to discover
current runtime capabilities. See the [integration handoff](explanation-integration-backend-handoff.md)
for implementation commits, generated schemas and actual serialized examples.

| Publication | Read/resume | Mutation contract | New publication |
|---|---|---|---|
| `exact-study/1.0` | Existing frozen state and assignment continue through the legacy adapter | Original setup, question order, consultation, status codes and receipts | Synthetic regression fixtures allowed; new live v1 revisions rejected |
| `exact-study/2.0` | Durable corrected setup, tutorial attempts, drafts and assignment | Explicit versioned v2 payloads only | Requires frozen tutorial, admitted synthetic resources, complete scoped workspaces and download metadata |

V1 model defaults, stored canonical publication bytes, hashes and canonical questionnaire
maps remain unchanged. Historical idempotent publication still returns its original hash.
Legacy checkbox completion is never relabeled as a comprehension pass. There is no automatic
migration, relabeling, or reset of active sessions. Existing JSON state/history tables store
v2 fields only for sessions issued against a v2 revision; no SQL rewrite is required.
Invitations retain their existing session-generation, revocation and reissue rules.

Every mutation keeps `idempotency_key`, `expected_revision`, CSRF and session binding.
An identical acknowledged idempotency key returns its original result before revision
checks; reused keys with changed content and stale revisions return 409. Wrong invitation
generations fail authentication. No tutorial retry is coalesced or interpreted as a scored
answer. The participant state excludes tutorial grading rules and all researcher case keys.

## Setup and preparation

`PUT /api/v1/study/setup` accepts v2 `setup_version`, two acknowledgements,
`resource_access` (`not_checked`, `available`, `needs_help`), optional bounded
`familiar_methods`, and `submitted`. Drafts remain at setup even when complete.
An explicit submission requires both acknowledgements and `available`; the server also
verifies the admitted downloads still exist with their frozen hashes. It advances to
background. Submitting background advances to tutorial, with no scored allocation.
Protégé experience is optional descriptive background in v2.

For example, a complete setup can be saved without advancing by sending this body
with the current session/CSRF headers to `PUT /api/v1/study/setup`. Set `submitted`
to true in a new mutation to advance; use the revision returned by the previous save.

```json
{
  "idempotency_key": "setup-draft-example",
  "expected_revision": 1,
  "setup_version": "setup/2",
  "instructions_acknowledged": true,
  "external_inspection_optional_understood": true,
  "resource_access": "available",
  "familiar_methods": [],
  "submitted": false
}
```

`PUT /api/v1/study/tutorial/progress` accepts a version, lesson position, optional practice
answer, optional assessment draft, `help_opened`, and typed `actions`. Each action has a
frozen `requirement_id` and `action` code, plus the corresponding typed evidence:

- Candidate inspection/return: `candidate_id` from the synthetic candidate set.
- Search/navigation/copy: `entity` with ontology version, IRI and kind.
- Citation/axiom/evidence-list inspection: `fact_id` resolved in admitted tutorial material.
- Ranking controls: `response_type` and `ranked_candidate_ids` in synthetic scope.
- Consultation practice: allowed unique `methods` (multiple or none as required).
- Download discovery: the synthetic ontology `asset_ids`.

Corrected clients send a complete `position` object: either
`{"view":"lesson","lesson_id":"<frozen lesson ID>","question_id":null}` or
`{"view":"assessment","lesson_id":null,"question_id":null}`. An assessment may name a
frozen question instead of null. Position-only saves are valid and durable. Omission
preserves position; explicit null, unknown IDs and inconsistent shapes return 422 atomically.
The deprecated `current_lesson_id` alias is derived from position. A non-null legacy lesson
updates position, while a null or omitted legacy lesson preserves it. If both fields are
sent they must agree. Action-scoping `lesson_id` does not navigate.

Old progress without position is normalized on read: a valid old lesson wins, otherwise
drafts/attempts select assessment landing, otherwise the first lesson. This fallback does
not recover historical screen focus. The next successful mutation persists it; reads leave
stored bytes intact. Old mutation hashes, saved receipts, completion and assignment remain
unchanged. Clients must wait for acknowledgement and use the returned revision for queued
saves; a stale competing write gets the existing 409.

`completed_requirements` is a cumulative acknowledgement echo: it may only name requirements
already acknowledged or satisfied by typed actions in the same request. A checkbox list or
client `passed` flag cannot complete the tutorial. Action reports remain self-reported UI
observations, not evidence of human attention. Actual assessment correctness is evaluated
by the server against the frozen rules.

`POST /api/v1/study/tutorial/assessment` accepts the tutorial version, question ID, unique
attempt ID, and a response using `choice`, `choices`, `matches`, or `part_b`. Typed fields,
row IDs, codes and complete response shape are checked before grading. Draft answers are
never attempts or passes. Incorrect attempts retain previous work and permit unlimited
pedagogical retries under ordinary request limits. Receipts preserve the response,
correctness, specific feedback and lesson to revisit. All five core items must eventually
pass; first and final attempts remain available for descriptive export.

`POST /api/v1/study/tutorial/complete` checks mandatory interaction predicates and all
five passes, then allocates once in the same transaction. Reopening synthetic help retains
the pass, assignment, scored ranking and task timer. Tutorial versions and assessment
versions cannot be republished with different frozen content.

## Consultation and ordered forms

Both consultation routes require the current explicit `presentation_id` and `form_version`.
`PUT /api/v1/study/cases/{case_id}/consultation/draft` permits an unanswered report or an
incomplete Yes report; it never advances or edits the committed ranking.
`PUT /api/v1/study/cases/{case_id}/consultation` requires Yes plus one or more unique methods,
or No with none, and advances exactly once. The stable v1 codes remain available, with
`queries_scripts`, `reasoner`, and `other_method` added in v2. `resource_scope=null` remains
unanswered. Current drafts and the previous committed receipt are separate state fields.

The frozen No policy is **reject stale methods, names, and resource scope**. The frontend
clears these values before sending No; the server rejects contradictory payloads with 422.
Wrong current presentations/form versions return 409. Draft and final histories remain
in the existing transactional history ledger. Optional names are omitted from default
analysis exports, including draft exports; there is no automatic tool-use inference.

V2 form questions carry explicit `option_order` and `row_order` arrays. Each is validated
as an exact duplicate-free permutation of the corresponding lookup-map keys (empty only
when the question has no options or matrix rows). Sorted-key canonical JSON does not
change these arrays. Existing answer codes and conditional/skipped semantics are preserved.

## Reproducible synthetic package and verification

Prepare a new local package with a Python environment containing the repository's study
and native preparation dependencies:

```sh
python -m tools.build_explanation_v2_fixture /tmp/exact-study-v2-fixture
python -m pytest tests/explanation_study_v2_test.py
```

The generator refuses a nonempty output directory and writes `publication.json`, original
admitted downloads/receipts, complete filtered context indexes, frozen explanation/evidence
resources and workspace indexes. It performs no provider request, matcher run, publication,
or participant contact. Lessons and assessment wording originate in the frontend's authored
synthetic tutorial and are checked in as `tools/fixtures/study_tutorial_v2.json`. Original
facts are regenerated from its actual OWL documents and rebound to real content identities;
placeholder fact IDs/hashes are not admitted. Prepared excerpt fallbacks retain their
honest generation status and provenance.

Core tests cover explicit setup submission, server grading, draft recovery, reissue/restart,
synthetic scope denial, allocation exactly once, consultation conflicts, exports and v1
canonical preservation. They also exercise a separate process exiting before and after a
transaction commit. Set `EXACT_STUDY_TEST_DATABASE_URL` to run the same core state-machine
checks on a dedicated PostgreSQL database. The opt-in v2 backup/restore test additionally
requires `EXACT_STUDY_TEST_PGBIN`, `EXACT_STUDY_TEST_PGDATA`, and the marked synthetic cluster
used by the existing recovery harness. It restores to a separate database and compares
completed tutorial state, consultation draft and immutable export.

V2 JSON/CSV exports support `exact-study-analysis/2` and `exact-study-csv/2` for historical
compatibility, and `exact-study-analysis/3` and `exact-study-csv/3` for corrected timing.
The new fixture CLI explicitly freezes version 3; model and test-helper defaults remain 2
so previously canonicalized publications retain their hashes. They retain
consultation method combinations and optional resource-scope missingness, first/final
assessment attempts, eventual completion, and separate observed tutorial/consultation
seconds. Late unavailable timing intervals are explicitly retained and excluded from
observed durations. Preparation is not a validated expertise score; method sets do not
establish candidate-level use, time attribution, compliance, or causal effects.

The admin create-export route accepts optional `analysis_schema`. For v2 choose 2 or 3;
omission selects the publication's frozen source export version. V1 only supports 1.
Unsupported combinations return 422. A researcher can derive analysis 3 from an existing
analysis-2 v2 publication without rewriting it: `manifest.schema` describes the derivation,
`manifest.source_protocol_versions` and `data.protocol_versions` preserve the source.
Revision inventory exposes `source_export_version` and `supported_analysis_schemas`;
the older `export_version` remains the source-version alias. Saved export retrieval and
CSV conversion use the saved manifest and retain its original content/hash.

Analysis 3 reports `page_observation_seconds`, sorted `per_page_observation_seconds`,
`page_instance_count`, `coverage_status`, `coverage_reason`,
`unique_elapsed_coverage_seconds`, `unobserved_elapsed_seconds` and `active_duration_known`.
The same summary appears in session `tutorial_timing` and case `consultation_timing`.
Page sums count only accepted bounded, deduplicated, non-overlapping eligible intervals.
Unavailable intervals remain in raw records with their reasons, excluded from sums/counts.
The three old observed-seconds fields are aliases of their corresponding raw page sums.

Coverage is always `not_established`, and both coverage/unobserved seconds are null,
including for one page. Reasons distinguish zero eligible clocks, one unmapped clock and
multiple clocks. Two pages observing 60 seconds each yield 120 page-seconds even if server
elapsed time is 100 seconds: no clamping, subtraction, inferred concurrency or attention.
`raw_elapsed_seconds` runs from the first usable-content server receipt for that
presentation to submission; subsequent ready events do not reset it. Tutorial/consultation
time stays outside scored time. JSON/CSV dictionaries define every field and null meaning.
Nested CSV JSON preserves null, while top-level missing scalar cells are empty, never zero.
Historical analysis-2 coverage arithmetic is not corrected elapsed-coverage evidence.

## Authorized workspace and readiness

Fresh v2 explanation current-case responses include strict `workspace: {"scope_id":"…"}`.
Use its URL-encoded ID with the same-origin `/api/v1/study/workspace/{scope_id}/…` routes.
The descriptor exposes no filesystem locations. Baseline returns null and no explanation
refs. Missing or unusable required frozen resources produce typed 503, distinct from an
explicit prepared `not_exported` result. Capabilities describe the physically filtered
frozen scope; they do not authorize unrelated contexts or imply unrestricted completeness.
Every read retains session, generation, consent, stage, condition and presentation checks.
Paused discovery is available but scoped reads remain denied; consultation keeps its
existing current-workspace permission. Tutorial help grants no scored-baseline access.

`case_ready` is a client assertion, not proof of rendering or attention. Each page must
report its own ready event before its case observations are eligible. The frontend still
owns the six-entity required-content readiness state machine and retry/focus behavior.
S1 backend tests do not establish those pending browser requirements.

## Errors and missingness at the wire boundary

| Situation | Response or preserved value |
|---|---|
| A mutation, current-case or scoped-workspace request sends another session's `X-Study-Session` | 409; the operation does not run under the replacement session |
| Baseline attempts to read its scored explanation workspace | 403; opening synthetic help grants no scored access |
| A malformed continuation is supplied | 422 `invalid_cursor`; no partial result is returned |
| A continuation from another query, scope or revision is reused | 409 `stale_cursor`; no partial result is returned |
| Final No includes stale consultation methods, names or scope | 422 validation error; the previous canonical answer is unchanged |
| Setup acknowledgements or resource access are incomplete on explicit submit | 422 validation error; no allocation occurs |
| Tutorial completion has outstanding mandatory interactions or core items | 422; progress and attempts remain resumable |
| The session exceeds the short tutorial request limit | 429 with `Retry-After: 60`; there is no lifetime retry limit |
| Optional consultation resource scope is unanswered | JSON null, retained as missing in analysis |
| A v2 timing segment arrives after its stage ends | Acknowledged as unavailable and excluded from durations |

The late timing acknowledgement has this shape (the ID echoes the submitted segment):

```json
{
  "acknowledged_segment_id": "late-segment-example",
  "availability": "unavailable",
  "reason": "stage_no_longer_current"
}
```

Original axiom ASTs and canonical-byte/base64 payloads remain lossless source data.
Base64 is not human-readable OWL syntax. Unsupported renderings preserve their reasons
and resolve to original fact/axiom inspection; no human-readable reconstruction or
inferred hierarchy is claimed merely because a structural projection is available.
