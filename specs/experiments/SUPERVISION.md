# Hourly experiment supervision

The lightweight supervisor checks the declared Slurm steps every hour. Healthy checks run
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

Defaults are two interventions per incident, four per rolling 24 hours, and 30 minutes per
Codex intervention. These limits control agent usage, not experiment duration. They do not
bound tokens exactly. The configured Codex model/effort are retained. Each intervention
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
tail -f data/experiments-v2/hourly-supervisor-01/launcher.log
touch data/experiments-v2/hourly-supervisor-01/PAUSE
```

PAUSE prevents new interventions; a current repair observes the instruction to stop before
further mutations/submissions. Removing PAUSE resumes hourly checks. STOP exits the supervisor
and interrupts its Codex invocation, leaving detached experiments running:

```bash
touch data/experiments-v2/hourly-supervisor-01/STOP
```

Do not launch a second supervisor or reset its state to clear a blocker. The exclusive lock
prevents duplicate supervisors. Before manual experiment repairs, pause it and check that
no intervention is still active. Interventions interrupted by supervisor shutdown are marked
for review on restart. Resume experiments from their saved runtime and preserve accounting.
