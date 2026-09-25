# Study backend operation and frontend handoff

The ranking study runs as an isolated FastAPI service with PostgreSQL. It mounts
only `/api/v1/study`, researcher routes under `/api/v1/admin`, and health/readiness.
It does not mount exploration, bundle import, matching, generation, or legacy study
routes. It also serves the first-party participant and researcher pages from the
frontend export: `/participate/` (the `#invite=` fragment is exchanged by POST and
removed from the address bar) and `/admin/` (the researcher token is kept in the tab's
memory only). `/` redirects to `/participate/`; exploration pages are 404. Set
`EXACT_STUDY_FRONTEND_DIR` to override the bundled export in `exact_inspect/static`.
Study pages forbid inline scripts other than the export's hashed bootstrap and forbid
inline styles. All examples and recovery checks in this implementation use synthetic
sessions; no invitations were sent, participants recruited, live study published, or
full experiment run.

## Configuration and publication

Install the `study` extra, or the small dependency list at
`deploy/render/exact_study_requirements.txt`. Set `EXACT_STUDY_DATABASE_URL`,
`EXACT_STUDY_SIGNING_SECRET`, `EXACT_STUDY_RESEARCHER_TOKEN`, `EXACT_STUDY_ORIGIN`
(an HTTPS origin without a trailing path), and `EXACT_STUDY_ASSETS_DIR`. Keep the
signing secret stable across deployments. Start the service using:

```sh
uvicorn exact_inspect.study.api:app_from_env --factory --host 127.0.0.1 --port 8000 --no-access-log
```

TLS terminates at the reverse proxy. Never enable request-body, cookie, or
Authorization logging. Responses and failures use `private, no-store` and
`no-referrer`. The origin check applies to participant mutations. Researcher
requests instead require the separate bearer token. Validation failures omit
Pydantic's raw input field, which could otherwise echo an invitation secret.
The service limits invalid exchanges to 30/minute per worker without recording
IP addresses; configure a proxy limit across workers before exposure to traffic.
Hosting logs may independently contain IP addresses and user agents: verify and
record the selected hosting/log retention arrangement before claiming anonymity.

`exact_inspect.study.models.Publish` is the complete publication input model.
`StudyDefinition` holds only participant-safe cases, pinned resources, frozen
forms/instructions, information and consent versions, four or more crossed
schedules, software identity, information-policy hash, and analysis plan. The
owner supplies consent wording; the software provides none. Real publication
requires explicit owner-supplied launch approval references. `CaseKey` is a
separate field in the researcher publication envelope and persists in the
`researcher_case_keys` table, never a participant package or participant response.
Protect database access and the admin credential accordingly.

A case has exactly five distinct candidates in contiguous initial positions.
Publication verifies the adjudicated IDs, all original ranks, natural-top-five
order, constructed-set provenance, resource bytes/hashes, case uniqueness,
transfer groups, complementary forms and both block orders. Prepared explanation JSON uses the strict `ExplanationResource` allowlist,
including original fact identities and grounded claims. Admission binds each resource's focal entities, profile/comparison scope and evidence candidates to its own case;
a resource shared by several cases must satisfy each case independently. OWL expression
fields use the frozen public constructor schema in `exact_inspect/study/owl_ast_schema.json`;
regenerate it with `tools/generate_study_ast_schema.py` when intentionally upgrading
pyowl-core. Admission validates exact constructor fields/types, predicate categories,
nested annotation visibility and strict original-document byte spans. Literal and IRI
facts preserve their permitted `qualifiers`, including definition citations and synonym
term/source metadata; expression qualifiers remain in the complete axiom AST. Unadmitted raw
syntax is rejected; faithful structured expressions remain available without loading
an OWL parser in the study service. Every publication, even
a synthetic one, supplies its full immutable `VisibilityPolicy`. Ontology assets
require a checksummed admission receipt from
`exact_inspect.context_resources.export_ontology_resource`: the exporter physically
filters prohibited annotations, preserves all declared logical axioms, flattens
the declared import scope, and renders a portable Functional Syntax ontology.
Publication verifies the receipt, ontology identity, policy, and resource bytes
without parsing or generating anything. Semantic blinding still requires content
review: a verified annotation policy cannot establish that every natural-language
label or axiom is suitable for a particular adjudicated case. Both conditions receive the same ontology files;
only the current explanation assignment can download its explanation assets.
Large ontology downloads are hashed incrementally and streamed in bounded memory.
Store assets on read-only storage or bake a verified package into a private
container image. Never mount researcher keys in the asset directory.

The CLI writes its result to a newly created file with mode 0600. It does not
send messages or distribute links:

```sh
python -m exact_inspect.study publish --publication frozen-publication.json --output publication-receipt.json
python -m exact_inspect.study invitations --study synthetic-study-v1 --count 8 --output private-test-links.json
python -m exact_inspect.study progress --study synthetic-study-v1 --output progress.json
python -m exact_inspect.study export --study synthetic-study-v1 --include-test --output analysis.json
```

Invitations default to test records; synthetic studies cannot issue live records.
Do not put recipient names/emails into this application. Preserve private links
for recovery; losing both link and cookie has no identity-based recovery path.
Researcher reissue keeps the same session and history and rejects both old links
and old cookies. `/close` freezes access without deleting study data. Published
study definitions, instructions, case content, forms and adjudication cannot be
rewritten: publish a new revision and follow the exposure/repair protocol.

## Participant API contract

Authoritative runtime request and response schemas are generated by FastAPI at
`/openapi.json` and checked in as
[study.runtime-openapi.json](../../specs/explanation-framework/protocol/study.runtime-openapi.json).
The [prepared explanation schema](../../specs/explanation-framework/protocol/study-resource.runtime-schema.json)
contains the strict original-fact, grounded-claim, hierarchy and evidence envelope.
The older `study.schema.json` and seed examples remain design-time documents;
runtime contracts add typed receipts, full frozen policy, resource admission and
separate draft/submitted states. Strict models reject ownership and condition
overrides. Use `/api/v1/study/state` as the acknowledged
revision authority. Mutation bodies contain `idempotency_key` and
`expected_revision`; identical retries return the original acknowledgement even
after progression. Reusing a key for different bytes or writing a stale revision
returns 409. Save the local retry queue before transmitting; preserve unacknowledged
changes on a conflict. Server acknowledgement is the recovery boundary.

The client exchanges `/participate#invite=...` through `POST /study/session`,
then removes the fragment from the address/history. Send the deployment Origin
and credentials. The secure HttpOnly SameSite cookie is a replaceable access
credential; the original invitation is reusable. A page preview or exchange does
not allocate cases or start time.

Progression is consent → setup → background → practice → ranking → consultation
(repeated) → final questionnaire → explicit completion. Decline closes the
session. Setup failures persist without allocating a condition. Call the setup
endpoint again during practice with all tutorial step indices acknowledged.
Allocation reserves the next counterbalanced schedule transactionally, shuffles
only within blocks using a persisted seed, and never rerandomizes on refresh.
Assignments from started sessions remain reserved, including dropouts. The
initial implementation uses global counterbalancing; it does not infer expertise
strata from questionnaire responses. Freeze a stratification design before
adding stratified allocation.

Ranking begins empty, supports an explicit partial list or explicit none or
insufficient evidence, and freezes on first submission. Submitted cases have no
revision API. Consultation is saved separately after submit; its time is outside
ranking duration. Forms include ordered wording/codes, exclusive options and
conditional branches, including the separate consultation wording. `show_if`
uses field equality or `{"contains": "option_code"}` for a multi-select condition;
the consultation yes/no keys map to the API's boolean field. Publication rejects
a form-version reuse with changed wording or component options. Hidden current values are cleared; mutation history is
append-only. Optional free text never blocks completion and is removed from the
anonymous analysis export.

Record `case_ready` only when the content is usable. Event IDs/sequence numbers
are independent from answer revisions. Save bounded observed monotonic intervals
through `/study/timing`, with a new page instance after reload. Visibility changes
never pause time. Explicit `/pause` and `/resume` are revisioned mutations; on
unexpected return, ask whether the gap represented external work, a break or
unknown activity and send `gap_activity` through `/resume`. Raw wall time,
observed intervals, explicit breaks and uncertain gaps remain separate. Missing
telemetry never blocks answer submission, and unknown/overnight gaps are never
silently counted as known active work. Cross-device overlapping activity is raw
self-report, not a reliable measure of attention. The backend does not observe
actions inside desktop Protégé.

Exports are immutable JSON manifests plus data dictionaries, frozen form labels,
assignments, scores, response/consultation states, events and raw/derived timing.
They exclude test records by default, invitation/cookie secrets and optional
identifying text. First/final submitted-response fields are equal because later revisions
are disabled; both are null before submission, with recoverable drafts in
`draft_response`. Unsubmitted cases remain missing. Pooled and per-condition
summaries include original-system MRR, delta RR, Hit@1, omission, response depth,
uncertainty, and natural-positive top-1 improvement/deterioration counts. Positive MRR and negative
none/endorsement/uncertainty denominators are separate. `--include-keys` enables a
researcher-only adjudication join. Summary metrics are descriptive; clustered
condition-effect models and pilot/power analysis belong to the frozen analysis
plan and are not estimated from synthetic data.

## Database migrations and recovery

`migrations/001_initial.sql` runs transactionally under a PostgreSQL advisory
lock, recording schema version 1. `/api/ready` checks database connectivity and
migration version. Future changes require a numbered, reviewed migration and a
backup before execution. Never use SQLite or ephemeral container storage for
participant state. SQLite is available only with an explicit test-only factory
argument.

Run the complete synthetic HTTP fixture client without a frontend:

```sh
python -m pytest tests/explanation_study_test.py::test_complete_synthetic_http_client_flow -q
```

The development fixture is retained at `data/explanation-framework/study-fixture`:
`assets/` contains participant-safe native ontology/explanation resources;
`publication.researcher.json` is the separate private publication envelope. Its
closing date is explicit, so regenerate it for later developer sessions. It
contains synthetic adjudication only; no invitation credentials are generated.

Run small tests against a dedicated PostgreSQL database:

```sh
EXACT_STUDY_TEST_DATABASE_URL=postgresql://user@127.0.0.1:5432/synthetic_test \
  python -m pytest tests/explanation_study_test.py tests/explanation_study_resource_test.py -q
```

The opt-in recovery test additionally needs `EXACT_STUDY_TEST_PGBIN` and
`EXACT_STUDY_TEST_PGDATA`. It permits restarting only a dedicated cluster under
`/tmp` or `data/explanation-framework/postgres`, carrying the explicit
`.exact-synthetic-test-cluster` marker. It uses `pg_dump` then `pg_restore` into a
new database, verifies the original
invitation/session/assignment/submission/export, and drops only that temporary
restore database. Do not point it at a shared or production database. Run tests
serially while this explicit restart test is enabled. Optionally set
`EXACT_STUDY_TEST_RECOVERY_DIR` to `data/explanation-framework/postgres` to retain
the private logical dump and a redacted receipt. The receipt verifies persisted
assignment, ranking, drafts/history, acknowledgements, events, timing and immutable
exports; it contains no invitation or database credential.

The checked-in [verification receipt](../verification/explanation-study-postgres.json)
records the current synthetic database verification. SQLite checks exercise the
same transaction logic for quick development but are not a persistence claim.
`test_complete_synthetic_http_client_flow` exercises the complete first-party API
without implementing frontend presentation or recruiting participants.

## Render preparation and launch gate

`deploy/render/study.render.yaml` describes a separate study service and paid
managed database, with independent secrets and automatic deployments disabled.
The Dockerfile builds the frontend in a Node stage and copies only the static export;
the runtime has no Node, model or exploration dependency. Supply the prepared
asset image/storage and HTTPS origin before publishing synthetic test content.
The blueprint is preparation; it has not been deployed or measured on Render.

As checked on 2026-09-24, Render documents continuous recovery for paid Postgres,
a three-day Hobby workspace window and seven days on Pro or higher, and seven-day
retention for logical backup exports. Free compute has no managed recovery.
Confirm the actual selected plan and dashboard recovery window before launch;
these details can change. See [Render backup documentation](https://render.com/docs/postgresql-backups)
and the [Blueprint reference](https://render.com/docs/blueprint-spec).

Before participant launch, record actual hosting region/access-log retention,
resource sizing and concurrent fixture-client latency. Rehearse a real Render
redeploy with acknowledged and unacknowledged synthetic requests, reauthenticate
with the original link, and verify unchanged assignments/submissions. Perform
PITR or a logical restore into a separate database and compare invitations,
assignments, histories and immutable export hashes before switching connections.
Record the retained backup location, schedule, retention and access controls.
A paid plan declaration does not prove successful recovery.

Remaining launch work includes frontend checks against the real study package
(keyboard/touch, screen reader, offline queue), browser/proxy cache isolation, real adjudicated cases, information
policy review, duration pilot, approved consent and analysis arrangements,
measured Render capacity and staged recovery rehearsal. The implementation does
not authorize participant contact or a public launch.
