# Combined batching and CUDA graph validation

**Completed:** step `14372.283` exited successfully at 2026-10-13 02:44:57 UTC.
Algorithmic invariants and cache checks passed. Batched scoring reached 39.995
pairs/s (125.017 seconds for 5,000 pairs); the graph-enabled arm reached 36.927
pairs/s with zero captures/replays. Graph acceleration remains unvalidated on
this natural workload. The same 202 final mappings were retained in this bounded
panel, with score/evidence/ranking drift reported and no paid calls. The user
subsequently approved [G4 with batching and graphs off](G4-BATCHED-RESTART-20261013.md).
The original launch-time authorization and handoff below are retained as history.

The October 13 authorization covers implementation, short regression checks and
one bounded offline validation. G4 and every other scientific run remain paused.
The user reviews the results before approving any further execution.

## Implementation and acceptance

The shared encoder path combines the existing staged context regrouping with an
optional model-local CUDA graph. Graphs preserve the current model, dtype,
tokenization, pooling and mathematical operations. The pinned BERT attention-mask
branch is specialized locally; no library global or model method is patched.
Repeated eligible shapes can be captured, checked against their same-shape eager
output and replayed with copied inputs and independent outputs. Model mutation,
unsupported execution, training, memory pressure and retention limits retain the
eager path. Graphs stay default-off and restricted to isolated qualification roles.

This comparison treats sequential output as a compatibility reference. Small
numerical changes and their effects on selected evidence, ranks or mappings are
reported, not automatically declared invalid. Complete candidate groups, fixed
method/configuration, finite outputs, genuine evidence and cache correctness remain
required. Existing strict-parity failures are retained as historical evidence.
See the amended [acceptance contract](SCORING-THROUGHPUT-200.md).

## Single authorized validation

Artifact root: `data/experiments-v2/g4-combined-optimization-20261013-01`.

The existing reference-free fitted G4 panel contains 5,000 distinct pairs in 250
complete candidate groups. Three fresh scorers use the same saved fitted artifacts:
sequential execution, staged encoder grouping, and that grouping with graphs.
Matrix arithmetic stays `mm` on device; tokenizer identity normalization stays off.
There is no fitting, reference use, hosted generation or scientific output promotion.

Native ontology loading and public feature preparation happen once and are timed
separately. Each variant starts with isolated empty encoder/numerical caches;
600-pair prefixes also check warm, replayed and partly cached execution. Counters
distinguish graph captures/replays, fallback, encoder calls and actual scored pairs.
Preparation-inclusive estimates and fitted extraction times accompany scoring rates.
This panel is not full cold-feature or multi-dataset qualification of 200 pairs/s.

The immutable recipe, source manifest, numeric Slurm step and dispatch nonce bind
the result. Bounded local caches and independent whole-tree storage guards protect
both local disk and home storage. No deadline is imposed on scientific work.
The finite worker requests six CPUs, 112 GiB RAM and the RTX 4090 in allocation
`14372`. Its graph retention cap is 2 GiB; the library default remains 512 MiB.
Local/durable tree ceilings are 32/16 GiB, each with a 64 GiB free-space reserve.

## Monitoring and stopping point

The supervisor gets one explicit validation ID/nonce allowlist while `PAUSE` stays
in place. It can notify the user about a terminal result needing review, failure,
or a twice-observed lost step; this hook cannot dispatch or repair anything. It
uses the existing deduplicated notification outbox and transport. No successor or
pending scientific batch is added. The G4 STOP file remains intact.

The supervisor still depends on its node/allocation and the configured mail
transport. It cannot deliver alerts while it is itself offline. An unreceipted
lost step is reported only after this instance has observed it running, followed
by two absent observations at least a minute apart.

## Verification and handoff

Implementation commits:

- `2c0fd931`: guarded, bounded model-local encoder graphs and runtime/cache admission.
- `0220f274`: three-variant fitted comparison with correctness and drift reporting.
- `e4c93a38`: watch-only notifications for one explicitly admitted paused validation.

The frozen validation source is `2c0fd931cb036408b2e9d9e2716418ae2a2bc2d5`.
The reviewed recipe SHA256 is
`7d351410644791ebcf7fd5c48423effe9f78c1797611ae32f2432fa314e38d8e`.

Verification before submission:

- 511 CPU regressions passed, with 11 existing warnings, in 24.75 seconds
  (`cpu-regressions.xml` and `.log` in the artifact root).
- All five import contracts passed; scoped typing passed the six changed runtime
  modules and two benchmark tools. This is not a new full-repository typing pass;
  the earlier 25 errors in nine unrelated files remain outside this change.
- Real context and label encoder smoke `14372.282` passed 32 logical calls,
  including both mask branches, four captures and 12 replays. Every tested output
  was bitwise equal to its same-shape eager result; released graph residency was
  zero. `encoder-smoke-2g.json` binds the tested source bytes.
- The retained `14372.281` smoke reached its 512 MiB graph retention ceiling and
  therefore missed its minimum replay-coverage check. Its output comparisons all
  passed. The final test recipe uses the explicit 2 GiB cap; no numerical check was
  relaxed to obtain the later smoke pass.
- Frozen-supervisor compatibility smoke verified one deferred review notification
  across repeated checks/state reload, with no dispatch, repair, live mail or pause
  mutation. The implementation uses the existing notification transport; receipt
  delivery is not newly proven by this offline smoke.

The single long validation was submitted as **`14372.283`**, starting at
2026-10-13 02:15 UTC. The replacement supervisor is **`14372.284`**; its first
check at 02:17 UTC recorded this validation `RUNNING` and healthy and persisted
`seen_running=true`. Only the new ID/nonce is allowlisted for paused monitoring.
All seven existing pending batch records are unchanged and remain blocked by the
retained campaign pause. No additional batch or successor was queued.

At handoff the worker is loading the native ontologies for the first variant.
Results will be written to `validation/results.json`; operational progress is in
`status.json` and `validation.log`, with the owned outcome in `completion.json`.
Full-panel throughput and drift have **not** yet been measured. Implementation and
short verification are complete; finite validation, user review and any scientific
admission are separate later states. Historical checkpoints and spending remain
unchanged; no generic G4 resume or mixing of old and new predictions is authorized.
