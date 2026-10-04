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
package identity, ordered completed IDs and saved output bytes must match. The old attempt,
including its uncheckpointed suffix, remains untouched. New runtimes copy the newest **closed**
cumulative account at dispatch, never a running account during preparation.

Implementation: core `4df8397b`, supervisor `1bf3e485`/`026f0560`, checkpoint continuation
`9f9a9881`/`a903af28`. Focused checks: 256 campaign/request/worker tests; 216 supervisor and
notification tests (also on the isolated deployed supervisor); 52 checkpoint/worker/recovery
tests. Frozen-worker and real-artifact preflights are recorded in the recovery handoffs.
