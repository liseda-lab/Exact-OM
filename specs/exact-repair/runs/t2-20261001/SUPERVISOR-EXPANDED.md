# Authorized continuing XR-2.1 studies on allocation 14408

On 2026-10-03 the user explicitly instructed: "as long as work as being done we
have no time limit. Lets advance and start running stuff use our allocated
machine". They also require an email when the experiment queue finishes and the
supervisor becomes idle so they can approve what comes next. This supersedes the
initial smoke's 48-hour ceiling and its deferral of larger generated studies.

Read the supplied registry's `plan`, `handoff`, `time_limit_amendment` and
`expanded_program` first, then `docs/project/repair-xr21-expanded-studies.md`.
The old root `plan.json` and smoke reports remain immutable historical designs;
they do not restrict the new studies or supply new training/test splits.

## Autonomy and machine ownership

Work in this campaign's maintenance checkout and artifact directory. Routine
preparation, adapters, fixes, meaningful tests, commits, frozen batches and
compatible checkpoint recovery are already authorized. Prepare and submit the
next eligible registered study without requesting approval for routine details.
An intervention must finish with a concrete registry/handoff update; return the
supervisor's required JSON. Stop after handing off a submitted batch.

Use detached Slurm steps in allocation 14408 on liseda-t2 only. Preserve step
14408.0, interactive access, the extern-owned tmux server and all other campaigns.
Never cancel the allocation, use sbatch, start another allocation, or modify
`data/experiments-v2` or the stopped historical XR-2 campaign.

The allocation has one RTX 2060 SUPER with 8 GiB VRAM, eight CPU threads and
61952 MiB RAM. Aggregate experiment admission stays at six threads, one GPU,
49152 MiB RAM. Normally run one four-thread/32768-MiB GPU worker and one independent
two-thread/16384-MiB CPU worker. The supervisor reserves one thread/8192 MiB.
Use the existing dispatcher, nonce-bound receipts and tmux socket. Register all
changes under `supervisor/registry.json.lock`. Validate descriptors and actual
live steps before launch; never duplicate running work.

This deployment includes the user-authorized idle-notification improvement.
Do not edit deployed frozen supervisor source; fixes are tested and committed in
maintenance and deployed in a separate versioned controller export when needed.
The supervisor checks every 300 seconds, dispatches prepared jobs deterministically,
and invokes Codex only for confirmed failures or eligible preparation. Preserve
explicit gpt-6-astra, xhigh reasoning, ChatGPT login and no API-key fallback.
No daily repair limit; stop after two unsuccessful repairs of the same error
across replacement descendants, preserving all attempt and supersession links.
A quiet log is not a failure. Confirm disappeared steps on separated checks and
inspect actual receipts, descendant ownership and checkpoint compatibility.

## Time, accounting and scientific budgets

There is NO overall campaign worker-hour, elapsed-time, CPU-hour or GPU-hour
ceiling. The ledger's `limit_worker_seconds: null` is bound to the user amendment.
Charge all work and all failed attempts; never clear costs or reservations.
Preserve the historical ledgers, immutable cost reports and original smoke caps.
Those finished smoke caps are not a resource allowance for this larger program.

Finite worker slices and per-call deadlines prevent hangs; they are not permission
barriers to continued productive work. For new studies, choose operational job
slices from development profiling and checkpoint within them. If a healthy job
uses its slice with saved progress, autonomously prepare its next compatible
slice or record a larger logical-job operational ceiling. Preserve previous
costs and identify this as continuation, not a software-repair attempt. Keep
scientific per-case comparison budgets fixed across matched arms. Increasing
those budgets or grammar bounds creates a separately named development/resource
experiment; never adjust them using test outcomes. Runtime projections are
estimates, not user-imposed spending limits. API expenditure remains zero.

## Design and dependencies

Follow current XR-2.1 specs. Start native qualification, the historical 90-case
engineering regression, and development profiling. Do not declare G0-G2 solely
from passing unit tests: bind the requirement/test matrix, live backend scope,
actual generation coverage and remaining failures. Missing gates limit dependent
claims; continue independent preparation and eligible studies.

The expanded program authorizes fresh-case frozen-model evaluation, a larger
six-arm/three-seed training study, robustness, bounded grammar/search scaling,
and preparation/controlled corruptions of small real-ontology modules. See its
numeric targets and sequencing. No full factorial of every optional feature.
Freeze settings, schedules, source code, runtime, split identities and controls
before each experiment. Train/development/test parents must be genuinely distinct;
renamed or mirrored siblings inherit one split. Audit generator alias families
and structural fingerprints to prevent falsely independent parents. Exclude
historically exposed parents from new held-out evaluation. Store new study
checkpoints separately from the completed pilot; warm starts require an explicit
compatible arm, never silent continuation under different data identities.

Use symbolic rich-action controls with the same accessible semantic evidence,
action language and resource accounting. Fixed-inventory regret is diagnostic;
generated-pool verified semantic quality, coverage and effort are primary. Keep
unavailable, partial and unknown rows in every denominator. Preserve both original
results and all new revisions. Compare learning efficiency only after its gates.
Real ontology preparation and controlled synthetic corruption are authorized;
actual production matcher outputs remain deferred until user release. LLM labels
remain disabled. Dataset editions follow the specs, with separate release hashes.

## Completion and notification

Fill registered preparation slots and continue their eligible independent branches.
Do not leave an empty queue with `remaining_work_status: pending` after completing
only a convenient subset; carry forward all remaining registered study stages.
If a stage is scientifically inapplicable or terminal after the repair limit,
record why and continue independent work, keeping its scheduled denominator.

When every authorized study stage is completed or explicitly accounted for,
publish cumulative costs and a machine-readable scope-completion receipt, set
remaining_work_status to terminal, and leave pending_batches empty. The controller's
`notify_when_idle` policy queues ONE durable `approval_needed` email to
pgcotovio@gmail.com through connected Gmail for that completion episode, without a
diagnostic model call. Repeated checks and restarts must not resend it. Await the
user's approval for an additional research program. Do not send duplicate manual
completion emails. Genuine blockers still notify through the same mechanism,
while independent eligible work continues.
