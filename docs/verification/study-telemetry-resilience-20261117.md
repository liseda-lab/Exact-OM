# Study logging correction — ready for activation after the current session

The active study remains `exact2-ncit-doid-production-20261117-traces`. This correction does not republish it, alter the cases or consent, reassign sessions, replace invitations, or change the public demo. At the last read-only pre-activation check, one real participant was in progress and two invitations remained at welcome. The live study PID and deployed commit were unchanged. This release is staged, not activated.

## Failure and correction

The newly introduced Decision trace and Scores tabs emitted `tab_open` targets missing from the backend allowlist. The optional-event endpoint rejects a bad batch atomically. The previous browser retried the same batch indefinitely and uploaded timing only after the event queue drained, so one rejected observation could stop later clicks, case-ready markers and timing. Answer mutations remained independent and continued saving. The first real participant's earlier missing observations must remain marked incomplete; they cannot be fabricated from answer timestamps.

The earlier backend correction admits both tabs only when the frozen publication and current workspace explicitly admit their components. Baseline and the synthetic tutorial remain protected. This release includes that correction and adds:

- A durable outbox for events and timing in local browser storage, isolated by session and keyed by immutable record IDs. Reload/reconnect replay preserves the original page clock, sequence and presentation. Acknowledgements remove only their own records; retries remain deduplicated by the server.
- Rejected batches are split to identify individual bad observations. Those records are retained separately for investigation/retry. Valid observations continue, and each timing interval waits only for its own pending case-ready event. One rejected timing record also cannot block later intervals.
- Network, server and authentication failures retain pending records. Requests time out, flush work is bounded, and pending data is persisted before sending. Page hiding/closing attempts a keepalive flush; reconnecting retries. No claim is made that every browser shutdown can finish a request.
- An activity-recording notice appears for delayed, rejected, lost or non-durable observations, separately from answer-saving status. It provides a retry action. Storage is bounded to 2,500 records / approximately 2 MB per session; overflow is counted explicitly, not silently truncated. A storage-disabled browser falls back to memory and shows a warning.
- A case-transition correction: after submitting a case, the timer resumes only when both the stage and presentation are unchanged. A transition from one `case` to another must not restart the previous case's timer. The full browser run exposed this additional issue; a regression test covers it.
- Server rejection diagnostics log only internal event/session identifiers, event type and status, never answer text, invitation secrets or ontology URLs. A read-only operator audit identifies submitted cases missing accepted readiness or timing.

The backend's atomic validation and authorization remain intact. Late timing remains explicitly unavailable where its relation to a completed stage cannot be established; hidden-tab time is not automatically treated as attention or a break. This change does not add eye tracking, click coordinates, arbitrary click capture, screen recording or external-site tracking. The existing participant information already covers these study-page actions.

## Verification

- 50 backend tests passed across telemetry admission, corrected HTTP contracts and timing analysis.
- 162 frontend tests passed, including 15 durable-delivery scenarios and the case-to-case timing regression. TypeScript and the production static build passed.
- Two complete browser journeys passed against an isolated copy of the actual published study: 20 distinct case presentations, opposite condition orders, five tool and five baseline cases each, consultation and final completion.
- Fault injection covered an invalid event, an invalid timing record, temporary 503 failures, reload with pending records, and a server commit whose acknowledgement was lost. Valid records continued, the failure notice appeared, replay retained IDs and clocks, and retries did not duplicate saved records.
- An independent read-only SQLite check verified that every one of the 20 cases had an accepted case-ready event, ranking/submission events and observed timing. Every tool case recorded trace and score tab openings; baseline cases recorded neither. Accepted page sequences were contiguous and observed timing intervals did not overlap. Both browser outboxes were empty on completion.

Private test evidence is on AppVM under `~/exact-study/telemetry-resilient-20261117/private/`: `browser-report.json`, `persisted-verification.json`, build logs and the isolated validation database. The staged release reuses existing dependencies and ontology assets. It does not duplicate a full ontology bundle.

## Activation and follow-up

The owner-facing command will be `ssh -t AppVM '~/exact-study/activate-study-telemetry-complete-20261117.sh'` after the current participant finishes. It requires the owner's sudo authentication. The script checks the saved acceptance evidence and exact release files, refuses any real session still in progress, backs up the database, updates only the study backend/frontend, and checks public readiness. It verifies that invitation identities and the frozen study hash are unchanged. Startup failure restores the preceding clean code and frontend export.

The two waiting participants keep their existing links. Ask them to refresh an already-open study page after activation and before starting so they load the corrected delivery code. Do not refresh the current participant's old page to try to recover its memory-only queue; an old open page may retry some queued events after the backend correction, but recovery is not guaranteed and earlier missing timing remains unknown.

For a non-mutating health check after a participant starts:

```sh
cd ~/exact-study/releases/telemetry-resilient-20261117
~/Exact-OM/.venv/bin/python -m tools.audit_study_telemetry \
  --environment ~/exact-study/study.env \
  --revision exact2-ncit-doid-production-20261117-traces
```

The audit prints no answers, invitation secrets or raw session IDs. It reports event/timing counts, timestamps and per-case missingness. It never creates an export or changes a session. A healthy early-case check should show a case-ready event and recent activity; timing intervals normally flush roughly once per minute, plus transitions/visibility changes. Idle participants need not continuously produce click events.

Implementation: `telemetryQueue.ts`, `telemetry.ts`, `timing.ts`, `ParticipantApp.tsx`, backend `study/telemetry.py`. Reproduction: `tools/verify_study_telemetry_browser.cjs` plus `tools/verify_study_telemetry_records.py`; both acceptance tools operate only on explicitly isolated test sessions. Deployment: `deploy/appvm/activate_study_telemetry.sh`.
