# Hosted spending amendment — 2026-10-04

The user authorizes continuing the remaining campaign without the previous aggregate
request/token allowances. **100,000,000 cumulative tokens is a one-time notification
threshold, not an admission limit, pause or deadline.** This supersedes the phase request
and token caps and protected token reserves in earlier plans. It does not enlarge scientific
populations, change model selection or authorize another paid retry of an ambiguous request.

The immutable operational authorization is bound by both path and SHA256 through
`EXACT_HOSTED_SPENDING_POLICY_PATH` and `EXACT_HOSTED_SPENDING_POLICY_SHA256`.
Absent that verified binding, the existing capped behavior remains. The current binding is
`data/experiments-v2/supervisor-maintenance-20261004/notification-only-spending-01/hosted-spending-policy.json`.
Its SHA256 is `3bb8c4462219df7351e3968fee1f52104ef5ec04aab47747e92cc43a25e27a94`.

- Keep historical budget limits and every completed/failed/unknown charge intact. New work
  records the operational amendment; every new transmission retains its policy in SQLite.
- Campaign workers remove aggregate wire caps under this policy. Explicit numeric wire caps
  still apply to cache-only/reporting scopes. Each request retains a finite positive output
  bound, unknown-paid retry protection and concurrency controls.
- Notification accounting follows the newest cumulative lineage and adds current work's
  measured ledger delta from its durable admission baseline. Inherited ledgers are not summed.
  Missing legacy baselines are reported as incomplete rather than invented. Uncertain paid
  exposure uses retained conservative reservations; it is not a provider billing statement.
- Resource and storage guards remain enforced. Rationales remain off. Private final references
  remain excluded from optimization. Package integration, especially pyHermiT, remains gated
  on the user's publication confirmation.

E12 resumes both original retrieval arms using verified inputs and exact cached responses.
E25 continues its 5,070/5,842-pair checkpoint under an explicit source-hash migration that
rejects changes outside the reviewed accounting modules. Its input/configuration/role/seed,
package identity, ordered completed IDs, predictions, explanations and gate contents must match.
Only the inner checkpoint's two gate paths and dependent fingerprints are relocated. Cached
routing rows are retained; their path-only prepass metadata is regenerated. The old attempt,
including its uncheckpointed suffix, remains untouched. New runtimes copy the newest **closed**
cumulative account at dispatch, never a running account during preparation.

Implementation: core `4df8397b`, supervisor `1bf3e485`/`026f0560`, checkpoint continuation
`9f9a9881`/`a903af28`. Focused checks: 256 campaign/request/worker tests; 216 supervisor and
notification tests (also on the isolated deployed supervisor); 52 checkpoint/worker/recovery
tests. Frozen-worker and real-artifact preflights are recorded in the recovery handoffs.

## Deployment and handoff

At 18:50 UTC the registry atomically queued `E12-spending-recovery-06` and
`e25-spending-recovery-03`; E12, E25-oracles, E25-trust, E04 and E23 pending launchers
were rebound to the amendment. E23 remains reporting-only with both wire caps explicitly
zero. Future preparations use isolated source `769681bf` and the new bound environment.
The supervisor runs in Slurm step `14372.197`; active E21 step `14372.195` and the
interactive shell were preserved. Existing ready priority and one GPU/spending lane remain.

E25's definitive recipe and handoff are in
`data/experiments-v2/e25-spending-recovery-03/revision-02/`; earlier unqueued drafts are
historical only. Frozen source `47984c9c` includes the path relocation fix `927860fd`.
Seventeen migration regressions verify actual trainer acceptance, rejection of changed
scientific gate settings, and cached prepass reuse without scorer calls. Actual artifact
preflight verifies all 5,070 completed pairs and 13 retained numerical routing files.
E12's isolated source `6f1b39c3` passed 94 tests and exact original-config/input checks.
E04/E23 passed 45/40 focused tests plus two/four-cell metadata preflights. No scientific
qualification or new hosted call was made by these checks.

Operational receipts are under
`data/experiments-v2/supervisor-maintenance-20261004/notification-only-spending-01/`:
`publication.json`, `final-verification.json` and `supervisor-deployment/verification.json`.
Old spending alerts are resolved with the user's authorization; their sent or ambiguous
delivery records remain intact. The monitor currently observes 23,996,379 closed-account
tokens. Legacy active E21 lacks a live admission baseline, so its unfinished delta is
explicitly incomplete until normal finalization. Fresh amended workers record that baseline.
If E21's unchanged old process later reaches its cap, prepare a policy-bound recovery from
saved work under this authorization; do not reset charges or seek another spending decision.
