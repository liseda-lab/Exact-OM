# XR-2 reproducible batches

The [batch plan](../../specs/exact-repair/protocol/batches.json) covers XR-E00–XR-E09:
smoke preparation, matched model controls, smoke studies, pilot preparation and
training, generated studies, independent real-input preparation, transfer, and
reporting. Smoke and pilot remain exploratory. Installed component qualification
does not establish model quality or complete biomedical reasoning support.

The repair batch launcher reuses `tools/supervise_experiments.py`. Its only supervisor
extensions are continued deterministic observation during a Codex intervention and
opt-in `pending_batches` prerequisites/resource admission. Existing registries keep
their previous behavior. Other experiments' frozen supervisor deployments are untouched.

Freeze committed source and resolved protocol before submission:

```bash
python -m tools.repair.batch freeze batch-spec.json /absolute/new-batch-directory
python -m tools.repair.batch submit /absolute/new-batch-directory/batch.json /absolute/supervisor-directory
```

The specification names the existing allocation, campaign/ledger paths, interpreter,
protocol source, experiment capacity, and jobs. Each job has a stable logical ID,
maximum cumulative seconds, numeric CPU/GPU/RAM requirements, Slurm `gres`, and
command argv arrays. `{python}`, `{code}`, `{protocol}`, `{work}`, and `{campaign}`
are substituted without a shell. The first deployment is under
`data/exact-repair-xr2-20260926`; `handoff.json` identifies its authoritative paths.

Each batch contains the source commit/archive and file hashes, resolved protocol,
installed package list, source/native ontology dependency identities, commands and
submission receipt. Each attempt has independent logs, startup/status/completion
JSON, exit code, output hashes and the actual Slurm step ID. Logical work directories
retain completed teacher caches, study results and atomic optimizer/RNG checkpoints.
Run IDs are linked across replacements so retry limits survive replacement jobs.

The phase ledger retains allocated CPU/GPU/RAM time, measured CPU and sampled
worker-tree RSS, including failures. Its 3600-second smoke and 86400-second pilot
caps sum worker elapsed time conservatively under parallel execution. Inner stage
and per-call caps remain in force. Lost work is charged its outstanding reservation;
a Slurm accounting reconciliation must be explicit. Sampled RSS may overshoot and
counts shared pages per process. Slurm also limits each step's requested resources.
Neither these records nor GPU allocation time claim measured GPU utilization.

Training resumes at the next saved minibatch with optimizer and CPU/CUDA/Python RNG,
the selected development checkpoint, and unchanged scientific settings/input hashes.
Changed implementation or training inputs require an explicit compatibility review.
Teacher preparation checkpoints after each case; held-out cases remain unlabelled.
Study resumption does not replenish its budget. A failed model-selection criterion
remains a failed arm, and unavailable real inputs stay in the reporting denominator.

The supervisor uses a 300-second interval, gpt-6-astra with xhigh reasoning, the
existing ChatGPT login, no daily intervention cap, and two attempts per persistent
error. Routine checks invoke no model. Connected Gmail alerts are sent when a
decision/intervention is required and retain delivery evidence. Codex usage is
separate from experiment compute, with no invented subscription dollar estimate.
See the [repair handoff instructions](../../specs/exact-repair/SUPERVISOR-INSTRUCTIONS.md).
