# Run-once execution amendment — 2026-09-29

This records the user's approved operational optimization. It supersedes conflicting
runtime admission and qualification instructions in older plans. It does not change the
scientific hypotheses, candidate/query populations, training examples, epochs, seeds,
folds, controls, numerical precision, selection rules or held-out reference boundaries.

## Execution and accounting

- Each complete scientific cell runs once. A complete measured execution becomes that
  cell only when its input, extraction and evaluation artifact identities match exactly.
  Original fit files, mappings, metrics, source manifests and measured time remain bound
  and auditable. Finalization forbids a scientific subprocess; missing or incompatible
  evidence is an explicit error, never an invented result or a silent rerun.
- Historical execution and interrupted-attempt costs remain cumulative. Active work
  intervals and retained Slurm allocation time (including idle time) are recorded
  separately. A repaired directory never resets consumption.
- Family time envelopes and node-hour totals are forecasts and warnings. Unknown
  estimates are marked unknown; an underestimate cannot kill or refuse a valid task.
  WorkEstimate's safety multiplier is forecasting only. Do not introduce campaign
  timeout timers or Slurm time limits to approximate these forecasts.
- RAM, VRAM, disk, allocation ownership, corrupt-input and scientific-readiness checks
  remain protective. Request/token reservations retain API spending protection; all
  newly hosted branches are serialized until they share one atomic authoritative ledger.
- Historical ledgers and campaign locks stay immutable except their existing append-only
  accounting protocol. New operational amendments identify source and policy bytes.

## Reuse and throughput

Use one heavy GPU worker on liseda-03 (RTX 4090, six CPUs, about 125 GiB RAM).
Small offline preparation and reporting can overlap it. Native ontology loading and
projection remain mandatory; rationales remain disabled. Inputs for final tracks may be
prepared, but private final references cannot influence optimization or selection.

Prepared ontology data, embeddings and raw numerical evidence use shared, content-bound
caches across compatible cells. Roles, seeds, input bytes and numerical dependencies
remain in identities. Changes to downstream fusion/calibration/extraction reuse only the
upstream evidence they cannot affect. Query-dependent features keep their query context.

The 2026-09-30 capacity amendment sets the campaign's shared numerical-cache allowance to
128 GiB of payloads (`EXACT_NUMERICAL_CACHE_MAX_BYTES=137438953472`); SQLite overhead is
additional. Bind this setting through the supervisor registry's current successor environment
for prepared and future batches. Preserve existing entries and the uncached-computation
fallback when full. This is a disk-cache allowance, not a RAM allocation or task cutoff.
Already-running workers retain their original environment until they finish.

Bounded encoder prefetch, resident embedding rows and bulk tensor transfers remove
avoidable GPU synchronization and repeated copies. Verified local scratch staging reduces
NAS reads while durable inputs, checkpoints, receipts and results remain on NAS. Cache
loss permits recomputation; it never changes the scientific population or invalidates
already committed results. Compatible historical embedding migration additionally binds
the original source SHA and verifies unchanged encoder/pooling implementations.

No production rewrite or unmeasured speedup claim is implied. Forecasts should be updated
from useful scientific work; do not launch a second full qualification matrix merely to
measure the first matrix again.

## Dispatch

The authoritative remaining-scope inventory is `remaining-work.yaml`. All applicable
initial comparisons remain required; only previously declared conditional extensions may
be skipped when their gate fails. Missing input/recipe bindings must be identified and
prepared, not relabeled as scientific failures or completed cells.

Priority is: finish E19 once; E18 and independent ready controls; transfer and label-budget
comparisons; remaining hosted/NIL/retrieval/typed feature branches as their genuine inputs
become ready; G4; freeze; final cases/seeds; track mapping exports and reporting. Keep
resource-aware pending successors registered. Every submission retains a successor or an
explicit terminal/decision disposition, so an empty dispatch queue cannot silently imply
that the scientific programme is complete.

A roughly one-week target is advisory: first 12 hours for integration and remaining
bindings, hours 12–72 for screens, 72–84 for G4/freeze, 84–144 for final study and 144–168
for recovery/reporting. These are scheduling targets, not allocations or deadlines.
Full final populations are materially larger than current development samples; no claim
that the whole scope fits one week is justified until useful-run throughput supports it.
Do not cut samples, seeds or published controls to satisfy the target without approval.

## Supervision and alerts

Run all workers and the supervisor as detached numeric Slurm steps inside retained
allocation 14372. Preserve its interactive step. Poll deterministically every five minutes;
invoke gpt-6-astra with xhigh reasoning only for repair/planning or email delivery.
No daily repair ceiling applies. After two unsuccessful attempts at the same cause,
request human intervention instead of retrying indefinitely.

Keep failure detection, automatic repair progress and recovery in local receipts.
Email only when human action is required: a decision, approval, resource change, or
exhausted same-cause repair attempts. Apply this filter to both newly recorded events
and undelivered outbox messages; preserve sent and uncertain delivery evidence.
Transient supervisor check errors stay local; the same error persisting for at least
three checks and fifteen minutes requires a human check and produces one deduplicated alert.
The separate email worker uses the reviewed ChatGPT/Codex Gmail connection and requires
an actual send receipt. Safe failures retry with bounded backoff while supervision and
repair continue. Uncertain delivery becomes an actionable reconciliation incident rather
than a blind resend loop. Config drift cannot disable deterministic health checks or
outbox processing. Email still depends on the node, supervisor, Codex and Gmail being
available; local status/receipts retain undelivered evidence across restarts.

## Prepared handoffs

`tools/prepared_batch.py` prepares immutable development recipes from reviewed binding
packages. The worker verifies completed dependencies, imports their original selections,
retains declared policy inheritance and bounded training populations, and executes the
whole scientific comparison once. It copies only the small hosted request ledger and the
latest cumulative accounting; large numerical/native caches remain shared. Resume keeps
its own advanced account and checkpoints. A divergent lineage or unfinished reservation
requires reconciliation rather than a reset. Time forecasts remain advisory.

The supervisor checks prepared launch descriptors every 15 seconds without a model call.
Health and repair checks remain every 300 seconds. Descriptors bind the worker, recipe,
source, inputs, environment and a unique launch nonce. The dispatcher reserves ownership
under `registry.json.lock` before launching and registers only a verified numeric Slurm
step or matching terminal receipt. A controller restart reconciles the existing launch.
Science launchers belong to an existing tmux server in the allocation's extern cgroup;
replacing the supervisor step cannot terminate them. Keep a dedicated idle tmux session
in that server for the allocation's lifetime.

Pending recipes bind upstream results at dispatch, not before those results exist. They
retain one heavy GPU/hosted spending lane; native enrichment still requires its actual
materialized inputs, optional scientific decisions stay explicit, and G4/final work needs
the verified frozen selection. An input-blocked branch does not stop independent ready
work. Cold encoder prefetch remains off. This removes routine model-mediated handoffs;
it does not promise zero scheduler or verification overhead.
