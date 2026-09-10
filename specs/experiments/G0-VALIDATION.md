# Detached G0 validation

The user authorized bounded validation on 2026-09-10, with monitoring handed back rather than
waiting through the job. This does not authorize launching the long screen/confirmation campaign.
The detached job uses only NCIT–DOID train/development inputs; no private test references.

## Work and limits

The launcher measures full ontology-closure loading/projection and cold/warm 64-source
scoring with the pinned SapBERT/BGE encoders, followed by at most 20 hosted source groups and
one 64-training-source selector fit. The local resource probe explicitly disables generation;
the hosted probe and later production-method replay retain the frozen hosted settings.

After a measured forecast with a 1.5 safety multiplier, it may run the 300-source global/local
operational replay, deliberate interruption, relocation/resume, and completed-cache checks.
The fit population is explicitly bounded for validation, so this is not a full-training-quality
result or a component-selection experiment. A failed or unaffordable phase stops the job.

One GPU worker, two numerical CPU threads, at most 56 GiB process RAM, 2,000 hosted requests and
3.2 million hosted tokens are permitted. The relaunch is limited to 11h30 with a 30-minute
checkpoint margin, retaining headroom for earlier failed attempts and bounded loader diagnostics within the 12-hour
foundation envelope. Unknown hosted deliveries are not automatically retried.

## Monitor and stop

Current output: `data/experiments-v2/g0-validation-04/`.

```console
tmux attach -t exact-g0-04
.venv/bin/python data/experiments-v2/g0-validation-04/monitor.py
```

Detach from tmux with Ctrl-b, then d. `status.json` reports the current phase, process and
worker log paths. Follow its `stdout`/`stderr` paths for detailed progress. `launcher.log`
contains driver-level failures; `exit-code` appears when the driver exits. The monitor window
can remain open after completion, so session existence alone does not mean work is running.

Request a cooperative stop:

```console
touch data/experiments-v2/g0-validation-04/STOP
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

Attempt `g0-validation-02` stopped during DOID loading, before scoring or hosted requests.
The original root omits two annotation declarations; its `ext.owl` import relies on three
annotation declarations in sibling documents. The strict parser requires them locally.
`tools/prepare_doid_annotations.py` produces separate, byte-preserving derivatives with explicit
provenance. Original files remain untouched; no asserted annotation or logical axiom is deleted.
The three import declarations already occur in the same pinned closure.

Revision 06 binds all 15 imports from the original DOID `v2026-05-30` release, commit
`3a4023833a9d7048c7ad110b061b851344957fc6`, using local files with verified checksums.
The full closure passes strict native loading: 16 documents, 199,429 effective axioms,
305,921/305,921 root RDF triples consumed, no dropped triples or diagnostics, in 15.43 seconds.
Evidence: `data/experiments-v2/ontology-normalization/doid-611355c44553/strict-load-report.json`.
The runtime uses `import-map.normalized.json`; the unnormalized import map is diagnostic history.
This successful loader check is not yet the full G0 throughput/recovery result.

Attempt `g0-validation-03` stopped after 355.98 seconds while building DOID's entity-kind index,
before scoring or hosted calls. The shared core's retained `SignatureView` reference-count
traversal raises `BackendProtocolError: retained signature traversal found an unindexed entity`.
Exact only needs typed entity enumeration, so its facade now caches the public
`OntologyView.signature(include_builtins=True)` tuple and preserves lexical IRI ordering per
kind. The complete snapshot, import closure, axioms and typed/punned entities remain unchanged;
no dependency monkeypatch or permissive parser fallback is used.

The actual repaired DOID entity-kind index passes: 19,546 classes, 47 object properties and
61 annotation properties (19,654 distinct matching IRIs). The targeted check took 17.99 seconds
including strict loading. Evidence is under `data/experiments-v2/g0-signature-repair/`.
Tiny native/Python closure and overlay regressions also exercise imports, ontology-only
annotations, undeclared references, punning, cached enumeration and lexical ordering. The
fresh attempt uses the unchanged revision-06 campaign inputs and matching configuration.

The broader DOID preflight reached its 180-second cap while traversing class labels, after the
formerly failing index had passed. It does not establish full label/projection throughput;
those measurements remain part of detached G0. The focused regression suites pass (38 existing
ontology integration tests, plus 11 signature/import checks), as do scoped static, documentation
and import-boundary checks.
