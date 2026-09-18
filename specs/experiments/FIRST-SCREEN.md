# First bounded retrieval screen

The user authorized fixing the remaining issues and submitting this wave on 2026-09-18.
Slurm job **14217** (`exact-e05-screen`) was submitted at 17:33 UTC. It waits for the existing
interactive allocation **14212** to finish. Submission is not evidence that the experiment
has started or succeeded. The full scientific campaign remains gated by later results.

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
not assume durable ontology-graph reuse. The first wave plans zero hosted requests/tokens.
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
revisions, label-free resolved components, hosted gate off, and rationales off.

## Handoff

Release allocation 14212 when finished with that interactive session. From the cluster login
node, `scancel 14212` releases it explicitly; this terminates its interactive processes.
The queued batch job can then acquire liseda-01 without an attached terminal.

```console
squeue -j 14217
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/first-screen-01/slurm-14217.out
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/first-screen-01/slurm-14217.err
```

`batch-status.json` records the launcher state; the runtime `screen/progress.json`, per-cell
`experiment_manifest.json`, and eventual `screen/selection.json` record experimental progress.
`exit-code` appears when the launcher exits. Logs append across Slurm requeues and batch attempts
have separate records. The job has unlimited Slurm wall time and requeue enabled; persistent
stage checkpoints permit resume, but do not prevent the unresolved host resets.

Cooperatively stop scheduling/work at the next supported boundary:

```console
touch data/experiments-v2/first-screen-01/runtime/exact-om-focused-v2/STOP
```

The batch wrapper preserves an existing STOP file on requeue. Review the reason and remove it
only when intentionally resuming. Do not edit the frozen campaign or overwrite historical
attempts. This submission does not authorize changing production defaults or making reporting
claims from development selections.
