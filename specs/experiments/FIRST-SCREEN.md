# First bounded retrieval screen

The first attempt ran inside interactive allocation **14212**, as Slurm step **14212.2**,
from 17:59 to 18:48 UTC on 2026-09-18. All four E05 cells finished dataset preparation but
failed when the difference-scoring channel reached LLM relation-template generation. The
launcher had not supplied the updated API key. The ledger retains 28 rejected HTTP 401
attempts; no cell completed and no E05 result or selection was produced. Rationales remained
disabled, and the interactive shell and VS Code access survived.

The user authorized repair and resubmission. Revision 08 preserves every resolved scientific
configuration and corrects operational admission for the relation verbaliser. The authenticated
retry started at **19:58 UTC** in Slurm step **14212.3**, with tmux session `exact-screen-02`,
inside the existing interactive allocation. Its records are in
`data/experiments-v2/first-screen-02/`. This attempt was interrupted by the node reboot
around 20:24 UTC; Slurm closed step 14212.3 at 20:25:45. A second boot occurred at 20:35.
Its unclosed running marker is stale. The baseline retained a verified checkpoint of
512/5,842 scored pairs; no arm completed and no new hosted requests occurred. The earlier attempt remains intact in `first-screen-01/`.
The user authorized resuming that checkpoint in the current interactive allocation **14220**.
The continuation started at **22:39 UTC** as Slurm step **14220.1**, detached in
`exact-screen-03`, using the same revision 08 campaign and outputs in `first-screen-03/`.
All four arms completed by **00:30 UTC on 2026-09-19**, with saved scoring outputs.
Final selection then stopped because cached datasets did not restore the development
reference or raw retrieval diagnostics. The node stayed up. The user authorized recovery
in `first-screen-04/`: regenerate only raw retrieval lists, require their original exact
fingerprints, and evaluate the saved predictions. Earlier attempts remain intact.
The full scientific campaign remains gated by later results.

## Completed E05 decision and next batch

E05 recovery completed at **11:08 UTC on 2026-09-19**, step **14220.2**, exit 0.
It reused all four saved scoring outputs with zero new scored pairs or hosted requests.
The frozen decision retained **top-k 20**: baseline recall was 250/253; SapBERT reached
251/253 but missed the required 0.005 improvement; RRF reached 245/253; adaptive-k
retained 250/253 while exceeding the matched mean-pool-size guard. These are development
results for this 300-source sample, not final benchmark claims.

The next immutable declaration is
`data/experiments-v2/prepared-campaign-09/campaign.lock.yaml`: **12 new D0 cells**, comprising
all eight E26 channel-quality arms and all four E06 lexical/string arms, seed 17, global
mode. It imports the signed historical E05 decision without rerunning retrieval selection.
Optional supervised E20 is explicitly deferred pending measured training throughput;
this is an operational disposition, not a negative empirical result. All other readiness
blocks remain in place. Method settings and selection rules remain unchanged.

Validated prepared data, raw retrieval diagnostics, 121 relation templates and compatible
encoder embeddings are reused. Shared prepared data is isolated by reference bindings and
experiment role, with content hashes and native/schema checks. Scoring outputs are computed
for the new treatments. The complete prior budget and hosted-request ledger are retained.
The batch has a **13.8-hour conservative admission allowance**, including a 1.5 safety factor;
this is not a wall-time limit. It runs one heavy GPU worker with two numerical CPU threads.
Matching calls, fitting and rationales remain disabled; private test references are excluded.

Validation: **95 campaign/scoring tests and 51 cache/runtime tests passed**, with formatting,
lint and focused type checks. All 12 resolved configurations passed a separate admission
check. The candidate-quality guard now validates the scored source pool after exact-match
prefiltering; baseline pipeline snapshots may repeat unchanged legacy values, while changed
experimental controls still require the canonical configuration surface.

The channel batch started at **20:31 UTC on 2026-09-19** as **Slurm step 14220.3**
in the existing unlimited interactive allocation, detached in tmux `exact-channels-01`.
The interactive shell remains step 14220.0. Startup source/native/cache verification passed;
this records launch, not completed experimental outcomes.

```bash
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/channel-screen-01/launcher.log
cat /home/pgcotovio/Exact-OM/data/experiments-v2/channel-screen-01/launch-status.json
```

Progress and cell results are under
`channel-screen-01/runtime/exact-om-focused-v2/screen/`; `exit-code` appears when the
launcher exits. Detaching the interactive shell does not release the allocation. To stop
only this batch, cancel step `14220.3`, not allocation `14220`.

## Frozen work

Preparation: `data/experiments-v2/prepared-campaign-08/campaign.lock.yaml`.
Current recovery output and launch records: `data/experiments-v2/first-screen-04/`.
Completed scoring outputs: `data/experiments-v2/first-screen-03/`.
Evaluation repair scripts/evidence: `data/experiments-v2/first-screen-evaluation-repair-01/`.
Verbaliser repair evidence: `data/experiments-v2/first-screen-repair-01/`.
Node-reboot recovery evidence: `data/experiments-v2/first-screen-reboot-01/`.

Only E05's baseline, SapBERT, reciprocal-rank fusion and adaptive-k arms are admitted:
300 NCIT–DOID development sources, seed 17, global mode, target-label-free components.
The frozen primary endpoint is candidate recall with the declared matched-mean-k guard.
No fitting, hosted matching, or rationale generation is enabled. Relation verbalisation
retains its original LLM mode: it supplies scoring evidence and is independent of matching
and rationale flags. No private test references are accessed. E00 imports verified G0
operational acceptance at its actual 300-development/64-training-source scope, with no new
cells, treatment winner, fitted policy, or claim of current-code prediction compatibility.
E24/E26 and later work retain their dependency, capability and resource blocks.

The revised **8.28-hour conservative admission allowance** is not an expected duration or
wall-time cutoff. It retains the original cold/warm, candidate-pair and fresh retrieval-index
allowances, with a hosted contingency. Matching cache fingerprints allow all four arms to
reuse G0's 121 relation templates. The contingency reserves 2,904 wire attempts, 5,947,392
tokens and a USD 4 admission allowance if generation is needed; these are conservative
allowances, not expected usage or quoted pricing. Final-study and recovery reserves remain
protected. No new template requests are expected when the validated cache is reused.

One GPU worker runs at a time with two numerical CPU threads and the full 16-CPU allocation.
The persistent native interpreter and four native package fingerprints are pinned. Worker
commands use the current Python interpreter. The campaign CLI explicitly loads the repository
`api_key` through `--api-key-file`; the secret stays in the process environment and never
enters command arguments or artifacts. Source hashes are verified before execution.

## Repair and validation

The original preflight checked the matching gate and rationale flag but missed the inherited
relation verbaliser. The repair adds explicit credential-file loading, with missing/empty files
rejected before expensive work. Five CLI tests cover early failure, worker inheritance, and
absence of the secret from output/artifacts. Fourteen focused relation-template, model-revision
and numerical-control tests pass, including the actual difference-channel call and cache replay.

A live probe repeated one previously rejected public NCIT relation request with the updated
key. The pinned model/provider returned a valid template: 135 tokens, USD 0.00003015. This
usage is separately retained and charged. All four real resolved configurations are checked
against the failed attempt; G0 template bytes are accepted only through normal fingerprint
validation. Cached difference-channel checks forbid hosted calls and model loading.

Recovery retains the original input artifacts and dataset caches in explicit checkpoints
with zero scored pairs. Verified templates are added through a new checkpoint version.
No old campaign selection or progress is imported into the forecast-amended suite. The
original numerical identities must match before dataset restoration; normal dataset/native
cache validation remains active. Ontology objects still load where needed for inference.
SQLite backups preserve compatible encoder vectors and all failed request records. The
budget retains G0, failed E05 work, measured launcher overhead and the authentication probe.

Reboot recovery combines the latest 512-pair baseline checkpoint with the other three arms'
prepared boundaries, preserving signed progress for the unchanged campaign. The interrupted
SQLite database and its journal are copied before recovery; SQLite rolls back only the copy.
Original database bytes and prior attempts remain intact. The interrupted reservation is
settled at the 1,618-second Slurm allocation interval, explicitly a conservative bound including
preparation and possible controller delay after node loss. It is not a measured compute time.
The original forecast and all prior hosted usage remain recorded.

The earlier sampled-local evaluator repair reproduced G0's 300-source/310-query population
and saved metrics without rescoring. Its 24 runtime tests, 39 campaign tests and 15 launcher/
configuration tests passed. The separate 52-fixture recovery audit and two reporting guards
retain the limitations in `first-screen-preflight/recovery-fixtures.json`.

Cached evaluation now restores and scopes the development reference on both paths. New
caches also preserve raw candidate ranks and exact rows in a checksum-pinned reporting
sidecar. Audited legacy caches fail before inference when that sidecar is unavailable.
Global evaluator replay requires the saved sampled references. The focused suites pass
63 tests, including cold/cache recall parity, exact matches outside retrieval, empty-source
denominators and missing-reference rejection.

The current repair shares native ontology objects across retrieval arms and uses production
label access without full projection. A tiny native parity check passed. Raw candidate hashes
must match the historical pools before any metrics are accepted; no thresholds or k values
are revised. Pair scoring, hosted calls and rationales are disabled. Finalization imports the
historical extraction artifacts unchanged, publishes new evaluation artifacts with explicit
repair lineage, and preserves original timing measurements and cumulative charges. Current
code is not claimed to be prediction-identical to the earlier extraction implementation.

## Handoff

Keep allocation **14220** and its interactive shell alive for experiment and VS Code access.
Future launches use Slurm steps inside the existing allocation, as specified in
[NODE-SETUP.md](NODE-SETUP.md). Detached tmux hosts `srun`; detaching does not release the
allocation. The repair started at **10:56 UTC on 2026-09-19** as Slurm step **14220.2**, in `exact-screen-04`. Earlier tmux sessions were lost with the reboot.

```console
tmux attach -t exact-screen-04
squeue --steps -j 14220
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/first-screen-04/launcher.log
```

Press **Ctrl-b**, then **d** to detach. `interactive-launch.json` records the actual Slurm
launch; `launch-status.json` records launcher state. Runtime `screen/progress.json`, per-cell
`experiment_manifest.json` and eventual `screen/selection.json` record scientific progress.
Detailed worker output is in each cell's `exact.log` and `experiment.stderr.log`. `exit-code`
appears when the launcher exits. The allocation has unlimited wall time. This interactive
run has no automatic batch requeue; persistent checkpoints allow explicit recovery after a
node reboot, but tmux cannot survive a reboot or allocation cancellation.

The repair has no pair-scoring checkpoint loop. To interrupt it, cancel only the repair's
Slurm step (the step ID is in `launch-status.json`), preserving allocation 14220 and step 0.
Do not overwrite frozen declarations or historical attempts. Development selections do not
authorize final reporting claims or changes to production defaults.
