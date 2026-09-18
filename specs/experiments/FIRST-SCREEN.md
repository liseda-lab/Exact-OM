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
`data/experiments-v2/first-screen-02/`; `launch-status.json` records current launcher state. The earlier attempt remains intact in `first-screen-01/`.
The full scientific campaign remains gated by later results.

## Frozen work

Preparation: `data/experiments-v2/prepared-campaign-08/campaign.lock.yaml`.
Output and launch records: `data/experiments-v2/first-screen-02/`.
Repair and recovery evidence: `data/experiments-v2/first-screen-repair-01/`.

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

The earlier sampled-local evaluator repair reproduced G0's 300-source/310-query population
and saved metrics without rescoring. Its 24 runtime tests, 39 campaign tests and 15 launcher/
configuration tests passed. The separate 52-fixture recovery audit and two reporting guards
retain the limitations in `first-screen-preflight/recovery-fixtures.json`.

## Handoff

Keep allocation **14212** and its interactive shell alive for experiment and VS Code access.
Future launches use Slurm steps inside the existing allocation, as specified in
[NODE-SETUP.md](NODE-SETUP.md). Detached tmux hosts `srun`; detaching does not release the
allocation. The new session is `exact-screen-02`; the failed `exact-screen-01` pane is retained.

```console
tmux attach -t exact-screen-02
squeue --steps -j 14212
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/first-screen-02/launcher.log
```

Press **Ctrl-b**, then **d** to detach. `interactive-launch.json` records the actual Slurm
launch; `launch-status.json` records launcher state. Runtime `screen/progress.json`, per-cell
`experiment_manifest.json` and eventual `screen/selection.json` record scientific progress.
Detailed worker output is in each cell's `exact.log` and `experiment.stderr.log`. `exit-code`
appears when the launcher exits. The allocation has unlimited wall time. This interactive
run has no automatic batch requeue; persistent checkpoints allow explicit recovery after a
node reboot, but tmux cannot survive a reboot or allocation cancellation.

Cooperatively stop at the next supported boundary:

```console
touch data/experiments-v2/first-screen-02/runtime/exact-om-focused-v2/STOP
```

The launcher preserves an existing STOP file. Review its reason before intentionally resuming.
Do not overwrite frozen declarations or historical attempts. Development selections do not
authorize final reporting claims or changes to production defaults.
