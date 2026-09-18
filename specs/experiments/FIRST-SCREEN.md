# First bounded retrieval screen

The user authorized fixing the remaining issues and submitting this wave on 2026-09-18.
The wave started at **17:59 UTC** inside the existing interactive allocation **14212**,
as Slurm step **14212.2**, in detached tmux session `exact-screen-01`. This preserves the
interactive shell and VS Code access. The earlier pending batch job **14217** was cancelled
before execution when the user clarified this requirement; its submission record is retained.
The wave **failed at 18:48 UTC** with exit code 2; none of the four E05 cells completed.
All four finished dataset preparation, then the difference-scoring channel reached the
inherited relation verbaliser (`dataset.verbalization_mode: llm`). This separate generation
path was missed by the zero-hosted-call preflight. The launcher did not supply the updated API key,
and the ledger records 28 rejected HTTP 401 attempts with no returned usage. Rationales
remained disabled. The interactive allocation and shell survived. G0 acceptance remains
valid; no E05 selection or quality result was produced. The full campaign remains gated.

## Frozen work

Preparation: `data/experiments-v2/prepared-campaign-07/campaign.lock.yaml`.
Output and launch records: `data/experiments-v2/first-screen-01/`.

Only E05's baseline, SapBERT, reciprocal-rank fusion and adaptive-k arms are admitted:
300 NCIT–DOID development sources, seed 17, global mode, target-label-free components.
The frozen primary endpoint is candidate recall with the declared matched-mean-k guard.
No fitting, hosted matching, or rationale generation is enabled. No private test references
are accessed. E00 imports G0's verified operational acceptance at its actual 300-development/
64-training-source scope; it creates no new cells, treatment winner, fitted policy, or claim
of current-code prediction compatibility. E24/E26 and other steps retain explicit dependency,
capability and resource blocks; this wave does not complete the G1 pool freeze by itself.

`E05-budget-plan.json` records an **8.03-hour conservative admission allowance**, not an
expected duration or a wall-time cutoff. It charges four cold process setups, candidate-pair
work including adaptive-k's maximum 50 candidates, a fresh SapBERT retrieval-index allowance,
and a 1.5 safety factor. Shared deterministic embeddings are cached, but the forecast does
not assume durable ontology-graph reuse. The first wave planned zero hosted requests/tokens;
the unexpected rejected attempts above remain recorded in its ledger.
The prior G0 ledger retains all 4,405 requests and USD 0.4454526. A 12-hour historical foundation
reservation covers its measured lower bound and an explicit unmeasured allowance without
inventing wall intervals. The missing interrupted tail remains unknown. Implementation/test
work stays separately recorded. Final-study and recovery reserves remain protected.

One GPU worker runs at a time with two numerical CPU threads and the full 16-CPU node allocation.
The persistent validated native interpreter and four native package fingerprints are pinned.
Worker commands now use the current Python interpreter instead of resolving `exact` on PATH.
A SQLite backup of G0's compatible encoder cache seeds the persistent campaign cache; model,
tokenizer, role and input identities still govern each reused vector. Source files are hashed
at submission and checked at launch so a queued job cannot silently execute changed code.

## Validation

The sampled local evaluator repair reproduces G0's original 300-source/310-query population
and metrics from saved outputs without rescoring. The 24 runtime, 39 campaign, and 15 launcher/
configuration tests pass. A separate audit passed 52 recovery/ledger/population/replay and tiny
CPU training fixtures plus two reporting-amendment guard checks. Integrated ontology/raw-evidence
recovery retains the limits recorded in `first-screen-preflight/recovery-fixtures.json`.
All four real E05 configurations passed model-free materialization/preflight with pinned model
revisions, label-free resolved components, hosted gate off, and rationales off. These checks
did not exercise the inherited relation-template generation reached by the difference channel.
A repair must validate that path before restart. Switching it to deterministic verbalisation
would change model inputs and requires an explicit revised configuration, not a silent fallback.

## Handoff

Keep allocation **14212** and its interactive shell alive: they carry the experiment and
VS Code access. The failed experiment used a separate overlapping Slurm step within that
allocation. Its tmux pane is retained after exit; no experiment worker is currently running.
To inspect the retained session from liseda-01, attach with the command below; press **Ctrl-b**,
then **d** to detach. Future runs must also use Slurm steps inside the active interactive
allocation, as recorded in [NODE-SETUP.md](NODE-SETUP.md).

```console
tmux attach -t exact-screen-01
squeue --steps -j 14212
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/first-screen-01/launcher.log
```

`interactive-launch.json` records the active launch command and allocation; `launch-status.json`
records the launcher state. The runtime `screen/progress.json`, per-cell `experiment_manifest.json`,
and eventual `screen/selection.json` record experimental progress. `exit-code` appears when the
launcher exits. Logs append and launch attempts have separate records. The allocation has
unlimited Slurm wall time. This interactive run has **no automatic batch requeue**: tmux survives
detachment but not a node reboot or allocation cancellation. Persistent checkpoints permit
explicit recovery after restoring access; they do not prevent the unresolved host resets.

Cooperatively stop scheduling/work at the next supported boundary:

```console
touch data/experiments-v2/first-screen-01/runtime/exact-om-focused-v2/STOP
```

The launcher preserves an existing STOP file on restart. Review the reason and remove it
only when intentionally resuming. Do not edit the frozen campaign or overwrite historical
attempts. This submission does not authorize changing production defaults or making reporting
claims from development selections.
