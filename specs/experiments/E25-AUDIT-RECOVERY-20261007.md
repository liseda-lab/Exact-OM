# E25 audit recovery and supervisor repair — 2026-10-07

This is an implementation and operational recovery, not a new scientific amendment or
a completed E25 result. The approved benchmark-reference and binary-trust scope is unchanged.

## Failure and repair

Both original comparisons (`14372.261` and `14372.262`) finished primary and selector
scoring, then failed because candidate-audit event hashing received a policy containing
a `Path`. The candidate writer now receives the same JSON-safe policy already used by
the source audit. No score, threshold, candidate, routing or extraction rule changes.

All five latest source checkpoints retain complete 5,842-pair primary and selector results.
The importer verifies source completion, latest artifact/checksums, implementation and
scientific identities, complete populations and identical gate bytes. It relocates only
paths and dependent fingerprints; selector rows remain unchanged. The shipped trust arm
uses checkpoint `00000005`; the other trust arm and three oracle arms use `00000004`.
No full inference or new hosted calls are requested by this output-only continuation.

The earlier automatic repair incorrectly pointed `superseded_by` at a pending batch,
which invalidated both health checks and dispatch. The repaired dispatcher accepts a
queued `pending_recovery`/`recovery_for` pair and promotes it only after a matching
numeric Slurm receipt. Registry publication remains locked, atomic and fail-closed.

Disabling a predecessor must not hide its cumulative ledger during successor startup.
An explicit `retain_accounting` flag now preserves that account for `latest_account`.
The short initial preparation `14372.263` was stopped after detecting this visibility
problem. Its two preparation rows and all 295 previously retained rows were reconciled
into 297 rows, with original closed charges unchanged. The interrupted reservation was
closed from terminal Slurm evidence; no science was repeated to reconstruct accounting.

## Verification

- Dev: 89 audit/recovery/worker tests and 243 registration/supervision/account tests passed.
- Frozen controller: the same 243 focused tests passed in 4.89 seconds.
- Combined frozen worker: 50 audit/account tests passed in 6.69 seconds.
- Mypy passed for the two changed library modules; targeted Flake8 and whitespace checks passed.
- Actual-data checkpoint preflight passed for all five arms. The final accounting backport
  changed no numerical implementation bytes; both launch descriptors were revalidated
  against the current storage/prompt guards (37 trust bindings, 39 oracle bindings).

These checks establish compatibility and launch readiness, not successful final outputs.
The existing full-population completion checks still have to pass during execution.
The cache-only transport guard, native processing, rationale suppression, scientific
scope, token pauses/alerts and storage protection remain in force.

## Deployment and continuation

At 22:37 UTC on October 7, the replacement supervisor is live as `14372.264`.
Trust recovery is live as `14372.265`; oracle recovery is queued behind it. The
interactive allocation, shell and external tmux server remain intact.

Source commits:

- Dev audit fix/importer: `b8c7b7eb`; two-arm trust support: `eb13d2c3`.
- Dev recovery registration/accounting: `795bfd9e`.
- Frozen worker: `8fdb7595`; frozen supervisor: `759426ff`.

Operational evidence is under `data/experiments-v2/e25-audit-queue-repair-20261007-01/`.
The reconciled account retains 97,270,717 cumulative tokens. The two original E25
replay attempts and this repair made no new paid calls. Historical charges remain counted.

Use the supervisor registry for current state; this document is a dated snapshot.
Both recovery directories contain `HANDOFF.md`:

- `data/experiments-v2/e25-trust-audit-recovery-20261007-01/`
  uses `pending-batch-accounting.json` and the suffixed accounting recipe/entry files.
  Earlier unsuffixed preparation files are preserved and are not the live descriptor.
- `data/experiments-v2/e25-oracles-audit-recovery-20261007-02/`
  uses `pending-batch.json`. Its numerical source is original failed step `.261`; its
  operational predecessor is the stopped preparation `.263`.

The controller dispatches oracle automatically after trust releases capacity, inheriting
the latest cumulative ledger then. Do not manually launch another copy. Subsequent
E13-enrichment/G4/E17 preparation remains with the authorized supervisor.
