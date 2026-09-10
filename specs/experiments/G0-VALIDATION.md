# Detached G0 validation

The user authorized bounded validation on 2026-09-10, with monitoring handed back rather than
waiting through the job. This does not authorize launching the long screen/confirmation campaign.
The detached job uses only NCIT–DOID train/development inputs; no private test references.

## Work and limits

The launcher measures full original ontology loading/projection and cold/warm 64-source
scoring with the pinned SapBERT/BGE encoders, followed by at most 20 hosted source groups and
one 64-training-source selector fit. The local resource probe explicitly disables generation;
the hosted probe and later production-method replay retain the frozen hosted settings.

After a measured forecast with a 1.5 safety multiplier, it may run the 300-source global/local
operational replay, deliberate interruption, relocation/resume, and completed-cache checks.
The fit population is explicitly bounded for validation, so this is not a full-training-quality
result or a component-selection experiment. A failed or unaffordable phase stops the job.

One GPU worker, two numerical CPU threads, at most 56 GiB process RAM, 2,000 hosted requests and
3.2 million hosted tokens are permitted. The relaunch is limited to 11h45 with a 30-minute
checkpoint margin, retaining headroom for the earlier failed startup within the 12-hour
foundation envelope. Unknown hosted deliveries are not automatically retried.

## Monitor and stop

Current output: `data/experiments-v2/g0-validation-02/`.

```console
tmux attach -t exact-g0-02
.venv/bin/python data/experiments-v2/g0-validation-02/monitor.py
```

Detach from tmux with Ctrl-b, then d. `status.json` reports the current phase, process and
worker log paths. Follow its `stdout`/`stderr` paths for detailed progress. `launcher.log`
contains driver-level failures; `exit-code` appears when the driver exits. The monitor window
can remain open after completion, so session existence alone does not mean work is running.

Request a cooperative stop:

```console
touch data/experiments-v2/g0-validation-02/STOP
```

The parent forwards STOP to the active worker, including during ontology loading. Completed
artifacts and partial checkpoints remain available. Do not remove STOP and overwrite an old
attempt; use an explicit new attempt with verified reusable artifacts after reviewing the cause.

## Review results

`report.json` is the final outcome, per-stage `*.measurement.json` files record timing/resources,
and `budget-plan.json` contains the measured operational forecast. The shared hosted ledger
retains requests, raw responses and reported usage. A successful report does not automatically
mark all experiment families ready; unmeasured feature cases and campaign cost forecasts still
need admission review. Final output is reference-free submission mappings after development
selection, as specified in [LABELS-AND-SUBMISSIONS.md](LABELS-AND-SUBMISSIONS.md).

The updated repository `api_key` passed the bounded hosted capability/cache probe at
`data/experiments-v2/hosted-profile/probe-04/result.json`. The launcher reads that file into the
worker environment; no credential value is written into configs, logs or commits.

Attempt `g0-validation-01` failed before model work because CUDA telemetry was reset before
CUDA initialization. Its logs/report remain intact. The startup ordering was fixed before
launching the new attempt; do not interpret the failed attempt as a resource measurement.
