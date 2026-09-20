# Anonymous study service, persistence and Render — B3/B5/F1

Implement a complete first-party study service, not an iframe wrapper around a third-party survey. Reuse frozen ontology/context packages; do not parse ontologies, match or call OpenRouter during participation. Questionnaire content is in 13; case/metric semantics are in 11.

## Resources and state machine

Versioned resources: `StudyDefinition`, `QuestionnaireDefinition`, researcher-only `CaseKey`, `Invitation`, anonymous `ParticipantSession`, `Assignment`, `RankingResponse`, `CaseConsultation`, `QuestionnaireResponse`, `ConsentReceipt`, `TimingSegment`, `InteractionEvent` and `ExportManifest`. Server ownership binds study revision, assignment, package, case/presentation order and permitted resources. An ordinary application `ReviewDecision` is not a study ranking response.

Participant progression: welcome/information and consent -> setup check -> background questionnaire -> tutorial/practice -> assigned case/ranking -> per-case consultation -> next case -> final questionnaire -> completion. Save each step. Declining or leaving is not a completed response. Finish only after required steps are acknowledged; optional text may remain empty. No correctness feedback or score against the answer key is returned to participants.

Researcher workflow: validate/freeze study configuration and assets; preview both conditions with test sessions; generate invitation links in a batch; inspect aggregate progress; revoke/close links when needed; export versioned analysis data. Use researcher authentication separate from participant bearer links. An authenticated CLI plus minimal administration UI is sufficient; do not build a general account/recruitment platform. No automatic sending of invitations is required or authorized.

## Invitation and resume contract

A link carries an opaque cryptographically random secret (at least 128 bits; recommend 32 random bytes), not a sequential participant number, email, name, condition or case assignment. Store only its digest plus a separate internal study-scoped participant ID. The secret is a bearer resume capability: anyone given that link can access that session. It must remain reusable across refreshes, browser closure and new-device opening until the study's declared close/revocation, not be consumed as a one-time link.

Preferred URL: `/participate#invite=<secret>`. The first-party client exchanges the fragment through an HTTPS POST body for a secure HttpOnly SameSite session cookie and removes the secret from the displayed URL/history entry. Fragment format avoids putting the secret in ordinary HTTP URL logs; also suppress exchange bodies, cookies and secrets in application diagnostics. No third-party scripts/analytics on this entry page. Use a no-referrer policy for participant pages and external links. Validate origin/CSRF on mutations and rate-limit invalid exchanges. Equivalent designs must prove they do not leak credentials through proxy logs, referrers or telemetry.

An initial page GET/link preview must not allocate cases, mark participation started or start timing. Exchange/claim operations and allocation transitions are transactional and idempotent. Researcher token reissue/revocation preserves the same session/history and explicitly invalidates the old capability and cookies issued under its invitation generation; do not issue a blank replacement participant. Every authenticated request checks current revocation/status/generation, so a previously issued cookie cannot bypass reissue or revocation. Test the old cookie as well as the old link.

The original invitation remains the durable recovery mechanism, independent of local storage/cookies; ask users to retain it. Refresh with the cookie resumes directly; the original link recreates access if cookies are lost. Do not require account creation, names, email, device fingerprinting or remembered answers. Losing both link and browser session cannot be recovered by identifying the participant; explain this before starting. A distinct return code is optional, not a hidden extra identity requirement. Issuance exports may contain links for distribution, but analysis exports never contain invitation/session secrets. Do not store a recipient-name/email-to-link table in this application.

Call participation anonymous in the sense that identifying details are not requested; longitudinal records are technically pseudonymous under the random session ID. Do not promise that a unique link alone guarantees anonymity. Keep hosting/security log treatment and retention explicit, avoid raw IP/user-agent in research events, use broad background categories, and caution against identifying free text. Hosting access logs and external tools may have separate records; document actual deployment handling without claiming universal non-observation. No identity-based duplicate-person detection is promised.

## Proposed API surface

All routes are target additions under `/api/v1`; B0 generates final models/OpenAPI. Participant routes infer the owner from authenticated session, never trust a supplied participant ID to select a different record.

| Route | Purpose |
|---|---|
| `POST /study/session` | Exchange reusable invitation for session; uniform invalid/closed handling |
| `GET /study/state` | Restore current stage, version, assignment progress, draft and acknowledgements |
| `PUT /study/consent` | Persist supplied information/consent version, accept/decline and acknowledgement time; gate all later research steps |
| `PUT /study/questionnaires/{form_id}` | Versioned, idempotent background/final answers |
| `PUT /study/setup` | Installation/openability/readiness and practice completion |
| `GET /study/cases/current` | Only authorized current case and condition-filtered assets |
| `PUT /study/cases/{case_id}/draft` | Autosave explicit partial answer with expected revision |
| `POST /study/cases/{case_id}/submit` | Validate/freeze ranking and atomically advance to consultation |
| `PUT /study/cases/{case_id}/consultation` | Save resource-use answers; atomically advance to next stage |
| `POST /study/events` | Bounded typed event batches with deduplicated acknowledgements |
| `POST /study/pause`, `POST /study/resume` | Explicit breaks and restored timing segments |
| `POST /study/complete` | Idempotent completion after required final answers |
| Researcher-only `/admin/studies`, `/invitations`, `/exports` | Create/freeze/preview, issue/revoke and export; exact subordinate routes defined in B0 |

Run/pair/explanation/axiom/ontology-download routes also enforce the same session assignment and condition. A client query flag, guessed ID or switching to an old case cannot unlock baseline explanations, answer keys or another participant. Session exchange/state, answers, individualized case responses and researcher exports use `Cache-Control: private, no-store`; no shared CDN caching of these responses. Immutable public static assets may be cacheable. Test session/condition separation through browser/proxy caches as well as application caches. Declined consent is a durable closed outcome; no questionnaire/task/event collection begins without the required consent receipt. Minimal invitation/access handling is separate from research telemetry.

Local bundle import routes are not mounted in study or demo deployments. Authenticated researcher preview uses separate test records, excluded from analysis.

## Persistence, retries and repairs

Use a durable PostgreSQL database for invitations, study definitions, assignments, drafts, questionnaires, responses, timing and events. Immutable prepared ontology/explanation resources can live separately in verified read-only storage. Never use process memory, browser storage or Render's ordinary filesystem as the authority for responses. Render defaults to an ephemeral filesystem; see [persistent storage](https://render.com/docs/disks).

Mutable answer, consent, setup and progression requests carry a unique idempotency key and expected record revision. Initial invitation exchange/claim uses transactional uniqueness and replay-safe lookup without requiring an unknown state revision. Append-only events use per-session event IDs, not the current answer/session revision. Commit a response and its progression transition in one transaction before acknowledgement. Bind each idempotency key to its payload hash: identical retries return the same result; reuse with different content fails. Conflicting stale writes return an explicit conflict and current version, without silent last-writer-wins. Preserve first submitted answer and explicit later revisions separately. An old draft cannot overwrite a submitted response. Two open tabs/devices cannot accidentally double-submit or move the same session twice. Preserve unacknowledged client changes in a bounded local retry queue; display saving/saved/offline/conflict honestly. Browser-loss guarantees apply to server-acknowledged data, not unsent edits.

Deduplicate telemetry by session/event ID independently from answer transactions. A telemetry failure never prevents saving an answer; retain gap/loss indicators and retry where possible. Assignment is created exactly once transactionally after readiness, with schedule/seed/presentation identities. Restart, refresh and copied/restored database must preserve it. Cookie/signing configuration survives deploys or the original link can reauthenticate without creating a new session.

Freeze active study versions. Preparation bug repairs follow 06, but participant state uses database migrations, backups and append-only response/version history rather than copying anonymous package directories. Do not swap case content/prompts/condition instructions under an exposed case ID. Publish a new study/content version, record affected exposure, and pause affected participation until compatibility is resolved. UI-only compatible fixes may proceed with build identity recorded. A closed or revoked invitation receives a clear status, never a new blank session.

## Timing and interaction events

Record setup, tutorial, questionnaire and scored-task time separately. A case timer begins when required content is usable, not when navigation/network loading begins; record render/loading delays. Freeze its end at ranking submit, then time the post-case consultation questionnaire separately. Actual ontology inspection before submission remains part of task time. Keep raw elapsed wall time, explicit breaks and observed active segments; do not silently infer exact working time from mouse movement.

Tab invisibility does not pause task time: it can indicate Protégé or file inspection. Record visibility transitions descriptively. Heartbeat/segment IDs, a page-instance ID (monotonic origins reset on reload), and client monotonic duration plus server receipt time help reconcile reloads and clock changes. Explicit Pause marks a break; unexpected disconnect/browser closure creates a gap of unknown activity. On return ask whether they continued external work or took a break, retain the raw gap and report an estimate/flag without pretending exact recovery. Prevent duplicate segments on retry. Overnight browser closure must not silently become many hours of ordinary task time.

Use a typed allowlist: case ready, candidate inspected, rank add/remove/move/keep-initial-order, response-type change, hierarchy expand/collapse, definition/axiom open, evidence/table/graph/comparison open, graph zoom/fit, external resource link, pause/resume, visibility, submit and revision. Events bind case/presentation, component/element ID, sequence, time and build version. Log meaningful actions rather than continuous pointer movement, screenshots, keystrokes, raw questionnaire/free-text contents or visited external URLs. Record coarse device/viewport class only where useful. A displayed/opened component does not prove it was read or understood; no claim of tracking actions inside desktop Protégé.

## Hosting, exports and operational acceptance

Provide a separate Render study service configuration, managed persistent database binding, migrations, health/readiness, secure secret handling and deployment runbook. Existing demo configuration is not sufficient evidence of study persistence. Test real staging redeploy/restart with synthetic sessions, then database backup/restore with invitations, assignments and responses intact. Render recovery depends on the selected database offering; document and verify the actual backup/retention arrangement rather than assuming free-tier recovery. See [Render backups](https://render.com/docs/postgresql-backups). Resource sizing and concurrency are measured on the deployment profile; no GPU required.

Exports contain pseudonymous session IDs, background/form versions, assignment/order, candidate sets and original scores, first/final responses, consultation, raw/derived timing, events, missingness and software/package versions. Provide separate researcher-only answer/adjudication joins and a reproducible scoring export; no credential, recipient contact or hidden key in participant downloads. CSV/JSON exports include schema/data dictionaries and an immutable export manifest. Persist test/synthetic-session markers and exclude them by default. A claimed anonymous analysis export must omit invitation secrets and unnecessary identifying free text.

Required tests: reuse link after refresh/closed browser/cookie loss; pause/unknown-gap recovery; expired/revoked/completed states and old-cookie rejection after reissue; consent gating/decline; no shared response caching; refresh never rerandomizes; lost acknowledgement retry; draft/submit race; two-tab conflict; event retry/gap; baseline bypass denial; no secret in logs/referrer/export; restart/migration/restore; answer-positive/negative/uncertain scoring; no public demo access to study cases. B5 proves APIs and deployment preparation using synthetic sessions against real test PostgreSQL, including restart/restore; a local or private test deployment is sufficient at this gate. Actual Render staging rehearsal remains required before participant launch. Actual public launch follows F1 end-to-end validation and a separate launch checklist; it is not implied by this spec-writing task.
