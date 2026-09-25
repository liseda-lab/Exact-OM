# Experiment supervision

The lightweight supervisor checks the declared Slurm steps every five minutes. Healthy checks run
ordinary Python/Slurm commands and use no model allowance. A confirmed failure or a completed
queue needing another batch starts one fresh `codex exec` intervention with the
[bounded instructions](SUPERVISOR-INSTRUCTIONS.md). Codex uses the existing ChatGPT login;
API credential overrides are removed and API billing fallback is disabled.

The supervisor runs as its own detached numeric step within the retained allocation. It
does not keep this chat running. It survives closing the editor/terminal while the allocation
remains alive. It stops with the node or allocation; recovery from that requires another
persistent host and is outside this node-local setup.

State, registry, logs and intervention reports are under
`data/experiments-v2/hourly-supervisor-01/`. `status.json` is the latest check/action;
`health.json` records per-launch findings. `registry.json` tracks launches and applicable
experiment STOP files. A repair updates the registry when it supersedes a launch. Completed
runs require completion evidence, not merely a zero launcher exit code. Missing steps need
two observations; quiet logs are not classified as failure.

There is no daily repair limit. The supervisor allows two repair attempts for the same
unresolved error, even if a repair changes the run directory or Slurm step. Different
errors remain eligible independently; one blocked experiment does not prevent another
independent experiment from being repaired. Each Codex intervention has a 30-minute
bound; scientific runs have no supervisor time limit. A finished intervention is checked
again after at most one minute. These controls do not bound tokens exactly. The configured Codex model is retained.
The supervisor policy overrides reasoning effort to `xhigh` (extra-high), as requested
on 2026-09-25; both repair and Gmail notification invocations use this override. Each intervention
saves its prompt, JSON events, final report, exit status and token-usage fields. A needs_user
result or exhausted retry allowance leaves a visible blocker instead of an endless retry.

The deployment receipt binds the supervisor source, instructions, policy and successful
permission smoke. Changes to the user Codex configuration require review before further
interventions. Automatic approval review remains enabled; rejected actions are reported.
No changes are made to system cron, global authentication, cluster configuration or running
experiment snapshots.

From the repository, inspect or pause using:

```bash
cat data/experiments-v2/hourly-supervisor-01/status.json
tail -f data/experiments-v2/hourly-supervisor-01/launcher-02.log
touch data/experiments-v2/hourly-supervisor-01/PAUSE
```

PAUSE prevents new interventions; a current repair observes the instruction to stop before
further mutations/submissions. Removing PAUSE resumes checks. STOP exits the supervisor
and interrupts its Codex invocation, leaving detached experiments running:

```bash
touch data/experiments-v2/hourly-supervisor-01/STOP
```

Do not launch a second supervisor or reset its state to clear a blocker. The exclusive lock
prevents duplicate supervisors. Before manual experiment repairs, pause it and check that
no intervention is still active. Interventions interrupted by supervisor shutdown are marked
for review on restart. Resume experiments from their saved runtime and preserve accounting.


Interventions that need a scientific decision, exhaust their same-error attempts, or are
interrupted produce a persistent alert in `alerts/`. Configured email delivery sends one
message per incident and blocking outcome; failed deliveries are retried and their status
remains visible. The email contains the affected run, reason, repair summary and handoff
path, rather than raw logs or credentials. Delivery is best effort and depends on the
configured mail transport; a local alert alone is not an email confirmation.

The optional `notifications` policy supports a reviewed command receiving an RFC 822
message on stdin, local sendmail, or SMTP with SSL/STARTTLS. The Gmail command adapter
uses the existing connected account through a separate, bounded Codex invocation only
when an intervention alert needs delivery. It requires a successful Gmail tool receipt,
retains delivery evidence and token usage, and refuses to resend an uncertain delivery
until it has been inspected. These rare notification invocations use subscription
allowance; normal health checks still use no model calls. SMTP credentials are read from
a named environment variable, never stored in policy. The deployment handoff records
whether transport has been verified. `max_agent_runs_per_day` may be omitted or null;
only an explicitly configured positive integer adds a rolling daily cap.

After resolving a recorded user decision, retain its history and explicitly record the
resolution before making the incident eligible again. Do not reset the whole state to
clear a blocker. The user can change these policies; automatic repair agents cannot.


Email replies are not automatically treated as experiment instructions. Give the required
decision in the Codex conversation so it can be applied and recorded before resuming.
