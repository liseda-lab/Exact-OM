# Corrected study protocol: backend version boundary

This document describes the implemented additive `exact-study/2.0` contracts. It does
not authorize recruitment, certify the duration pilot, or migrate active human sessions.
The schemas generated from `exact_inspect.study.models.Publish` and the HTTP OpenAPI are
authoritative for field constraints. The independent protocol versions are frozen in
`StudyDefinitionV2.protocol_versions`: setup, tutorial, assessment, forms, information,
resource policy, software, and analysis export.

## Compatibility and migration

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

V2 JSON/CSV exports use `exact-study-analysis/2` and `exact-study-csv/2`. They retain
consultation method combinations and optional resource-scope missingness, first/final
assessment attempts, eventual completion, and separate observed tutorial/consultation
seconds. Late unavailable timing intervals are explicitly retained and excluded from
observed durations. Preparation is not a validated expertise score; method sets do not
establish candidate-level use, time attribution, compliance, or causal effects.

## Errors and missingness at the wire boundary

| Situation | Response or preserved value |
|---|---|
| An old tab sends another session's `X-Study-Session` | 409; no read or write runs under the replacement session |
| Baseline attempts to read its scored explanation workspace | 403; opening synthetic help grants no scored access |
| An unknown or cross-query continuation is reused | 422 domain error; no partial result is returned |
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
