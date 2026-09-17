# Detached G0 validation

The user authorized operational validation on 2026-09-10, a fresh post-native attempt on
2026-09-14, a restart without a wall-time limit on 2026-09-15, and repair/recovery of the
output-writing failure on 2026-09-17. Monitoring is handed back rather than waiting through
the job. This does not authorize launching the long
screen/confirmation campaign. G0 uses only NCIT–DOID train/development inputs; no private
test references.

## Work and limits

The launcher measures full ontology-closure loading/projection and cold/warm 64-source
scoring with the pinned SapBERT/BGE encoders, followed by at most 20 hosted source groups and
one 64-training-source selector fit. The local resource probe explicitly disables generation;
the hosted probe and later production-method replay retain the frozen matching settings.
Generative posthoc rationales are off by default for all experiments and require explicit
opt-in. Matching decisions, scores and exact numerical explanation traces remain enabled.

After a measured forecast with a 1.5 safety multiplier, it may run the 300-source global/local
operational replay, deliberate interruption, relocation/resume, and completed-cache checks.
The fit population is explicitly bounded for validation, so this is not a full-training-quality
result or a component-selection experiment. A failed or unaffordable phase stops the job.

Attempt 09 retains `--no-time-limit --requests-cap 100000 --tokens-cap 32000000`.
The user removed G0's wall deadline and approved finite hosted headroom for fitting and
replays. One GPU worker, two numerical CPU threads, at most 56 GiB process RAM, cooperative
STOP and the request/token caps remain enforced. Unknown hosted deliveries are not
automatically retried. Runtime forecasts remain reported as conservative estimates; they
are not wall-deadline gates. Request/token admission still applies.

The recovery plan retains the verified cold64, warm64 and fit64 artifacts originally measured
in attempt 06 and attempt 07's successful rationale-free hosted20 probe. Attempt 09 adopts
attempt 08's complete, content-addressed global300 extraction artifact and reruns reporting
without a model worker. Identity checks limit the intervening change to posthoc relation
normalization; scoring, selector fitting and saved mappings are unchanged. Completed model
work is not a new measurement. Historical time and hosted usage stay charged, and previous
attempt files remain unchanged.

This supersedes attempt 06's 41,400-second hard/39,600-second soft allowance and its
2,000-request/3.2-million-token caps for the new G0 continuation only. Attempt 06 and all
previous reports remain unchanged. Cumulative historical plus new G0 time, requests, tokens
and reported cost must remain visible even though the new wall allowance is unlimited.
Scientific campaign budgets and protected final/recovery reserves are unchanged, and the
campaign remains unadmitted.

## Monitor and stop

G0-09 recovery is configured for `data/experiments-v2/g0-validation-09/`, using
`/tmp/exact-native-candidate/bin/python` and tmux session `exact-g0-09`. This records the
restart configuration, not a claim that the job has launched.
`status.json` and `report.json` are authoritative for runtime status. The installed native
identity, adoption evidence and resource amendment belong with this attempt's plan.
Neither completion nor an ETA is claimed.

[Native T4 validation](../native-optimization/IMPLEMENTATION.md#completed-t4-result) passed
128 exact ordered feature rows across NCIT–DOID. Attempt 06 then measured the changed native
fingerprints and evidence schema 3 in a fresh cold64 run. Subsequent attempts preserve that
measurement; none substitutes attempt 04's older implementation or counts an artifact
replay as a new cold/warm timing.

After launch, monitor with:

```console
tmux attach -t exact-g0-09
/tmp/exact-native-candidate/bin/python data/experiments-v2/g0-validation-09/monitor.py
```

Detach from tmux with Ctrl-b, then d. `status.json` reports the current phase, process and
worker log paths. Follow its `stdout`/`stderr` paths for detailed progress. `launcher.log`
contains driver-level failures; `exit-code` appears when the driver exits. The monitor window
can remain open after completion, so session existence alone does not mean work is running.

Request a cooperative stop:

```console
touch data/experiments-v2/g0-validation-09/STOP
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


Attempt `g0-validation-04` completed cold64 successfully: 25,927.18 worker seconds, 13.57 GB
peak process RSS and 2.77 GB peak reserved GPU memory. Its 64 source decisions, 1,245 scored
pairs, 35 protected exact pairs and committed extraction artifact reconcile. No reference labels
or hosted requests were used. The outer controller then tried to publish an empty evaluation
artifact even though evaluation was disabled. The fix omits that stage, including evaluator
callbacks, while retaining the requirement that every completed artifact has durable outputs.

`--resume-from data/experiments-v2/g0-validation-04` verifies the saved campaign, configuration,
input/output artifacts and successful worker evidence, then adopts cold64 under its original
implementation identity. The original files remain unchanged; the old outer manifest still
records the interrupted bookkeeping. The new `cold64.adoption.json` and stage measurement record
that recovery, with zero new worker calls and the original cold wall time. Preparation without
`--execute` verifies the evidence without copying shared caches; execution uses SQLite backups.

Attempt 05 carried forward 26,003.85 seconds of active validation time, leaving at most 4h16m36s
of the original 11h30 allowance, or 3h46m36s before the soft stop. Downtime does not reset spend.
The saved caches contain candidate tables and embeddings, but no durable ontology graphs or
raw entity features. A fresh warm worker would rebuild those expensive structures: the cold
source projection alone took approximately 3h42m, before annotation indexing and scoring.
A completed-output replay therefore cannot be reported as a measured warm run.

Attempt 05's forecast used the original cold-reload term, requiring 43.21 hours with its
safety factor. Its controller recorded `blocked_budget` before launching warm64, hosted20,
fitting or production300. That historical outcome remains intact. The native validation and
explicit resource amendment permitted fresh attempt 06. Its outcome is recorded below;
G0 is not yet passed. The launcher never replaces cold timing with the near-zero cost of
adopting its outputs.


## Attempt 06 measurements and restart policy

Attempt 06 finished `blocked_budget` after all four probes completed: cold64 **848.92 s**,
warm64 **581.41 s**, hosted20 **756.06 s** and fit64 **1,749.54 s**. These are complete stage
wall times, including setup. Warm64 encoded zero new texts, confirming embedding-cache reuse;
it still rebuilt ontology/graph state. Total active attempt time was **3,936.42 s**.

Actual hosted usage was **522 requests, 377,208 tokens and USD 0.09027675**, with no unknown
or unpriced requests. The forecast, rather than actual exhaustion, blocked continuation:
17.55 hours total versus the 11-hour soft allowance, 24,012 requests versus 2,000, and
17,351,568 tokens versus 3.2 million. The whole-probe time extrapolation repeatedly includes
fixed setup; it is conservative planning evidence, not observed remaining runtime.
The 300-source global/local and interruption/replay checks did not run.

Within those observed calls, the non-rationale roles account for **133 requests, 19,779 tokens
and USD 0.00401265**, including one validation-only probe. This is a role-accounting subtotal,
not a new rationale-free run or a guaranteed future cost. The user explicitly requested the
new rationale-free default and sufficient hosted headroom; attempt 07 measures the resulting
hosted behavior rather than relabeling attempt 06. All prior rationale costs stay in history.


## Attempt 07 failure and attempt 08 recovery

Attempt 07 reused the completed cold/warm/fitting probes and passed hosted20 with rationales
off. Global300 completed scoring and selector processing for **5,842 candidate pairs** and
wrote **300 final mappings**, then failed while writing `source_decisions.json`. Its dataset
contains both `Scores` (protected exact confidence) and `Score` (candidate metadata). The
audit writer renamed both to `S_final`, producing duplicate columns and a pandas
`InvalidIndexError`. This was an output failure; no time, RAM or hosted cap stopped the job.
Local300 and the remaining recovery checks did not complete.

Commit `1ed45c3` makes the audit writer select `Scores` before `Score`, matching the existing
prefilter/extraction decision semantics, without renaming both fields. It also accepts
consistent/complementary entity aliases and rejects conflicting source/target identities.
The scoring and selection algorithms are unchanged. The focused export/replay suite passes
**23 tests**. Replaying the actual saved global300 artifacts reproduces the old failure and
then produces **300 source records and 6,000 candidates**, including **158 protected exact
pairs**, with every final score preserved and the same 300 emitted mappings. The replay made
no model calls and used no reference labels. Hashed evidence is retained under
`data/experiments-v2/g0-audit-repair-01/`.

Attempt 08 recovered the complete global300 checkpoint after the input/configuration and
writer-only AST checks. It completed audit export and builtin evaluation with **zero new
encoded texts and zero hosted requests**, then failed in posthoc E00 attribution. The public
development reference uses `<=` and `>=`, whose canonical meanings `<` and `>` are already
specified in [IMPLEMENTATION-CLARIFICATIONS.md](IMPLEMENTATION-CLARIFICATIONS.md). Attribution
rejected those valid aliases. The repair uses the shared relation normalizer and preserves
all source/target identities and directionality; it changes evaluation only.
The real saved-output postprocessing check passes attribution, builtin evaluation and replay
comparison with zero mapping-score/metric drift across all 300 mappings and no model calls.
Hashed evidence is in `data/experiments-v2/g0-postprocessing-repair-01/`.

Attempt 09 is configured to adopt the complete global300 extraction artifact, repeat
reporting without a worker, and continue the remaining local/global recovery checks. Attempt
08's failure, completed outputs and cumulative historical charges remain intact. G0 remains
unpassed until the new report confirms all prescribed checks.
