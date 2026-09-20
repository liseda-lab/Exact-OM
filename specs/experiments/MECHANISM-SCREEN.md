# E02 mechanism screen

The channel screen completed at **00:39:58 UTC on 2026-09-20**, Slurm step
**14220.4**, with exit zero. All eight E26 and four E06 configurations are complete;
three cells were reused and nine ran in that continuation. Neither family selected a
replacement. E05's top-k20 pool and both channel controls remain the frozen outputs.
These are 300-source development screens, not final benchmark results.

## Next admitted comparison

`data/experiments-v2/prepared-campaign-10/` prepares the complete six-arm E02 screen
on D0 NCIT–DOID, seed17, 300 development sources, global alignment. It imports the
original E05/E06/E26 decisions without rescoring them. The six treatments, their
supervision labels and selection rules remain unchanged.

Trusted anchors are selected deterministically from the public training positives:
3,845 distinct pairs, 576 excluded because either endpoint has competing mappings,
3,269 eligible pairs, and a fixed 2,000-anchor sample. All anchors are one-to-one and
disjoint from the entire 619-source development universe. The predicted-anchor rule
uses the existing threshold0.95 and margin0.1, declared before execution; it is not
a fitted or outcome-optimized rule. Both noise diagnostics use the same trusted
population. Input hashes and the deterministic builder are retained with the lock.

All six resolved configurations passed admission, alongside 141 campaign/scoring
regression tests. A further regression covers predicted-anchor preparation, disabled
LLM calls during the base pass, fresh-trainer checkpoint reuse, and invalidation
when diagnostic corruption changes. Before any experiment cell, the detached
launcher must verify all anchor endpoints against native ontology class signatures.
A failed native check stops the queue.

Hosted matching retains the baseline analytic gate; **rationale generation stays
off**. Structure supervision does not trigger selector/head fitting in these arms.
No private test reference is used. Native package fingerprints stay fixed.

## Resources and recovery

The queue uses one heavy GPU worker, two numerical CPU threads, and an `srun` step
inside the existing interactive allocation **14220**. The launcher has no wall-time
limit. Its 7.75-hour conservative admission forecast includes cold preparation,
structural rescoring and a1.5 safety factor; it is not a measured E02 duration.

Under the user's instruction to provide sufficient hosted allowances, this wave
records an explicit request planning cap amendment from20,000 to50,000. The32M
token cap,5,000-request/8M-token final reserves,336node-hour cap and every envelope
remain unchanged. E02 reserves19,200 requests and12.288M tokens from a conservative
extrapolation of G0's hosted probe, without assuming cache savings. Historical
spend and failed attempts remain in the cumulative ledger.

The latest channel-screen-02 prepared data and embedding cache are copied and
verified. Its request cache contains28 earlier rejected attempts; the new wave
also imports G0's3,883 verified completed responses, preserving both histories.
Only exact matching request identities can reuse these responses; changed prompts
remain separate. Cache import itself incurs no new hosted usage.

The immutable lock, input provenance, admission, forecast and bootstrap/launch
receipts are under `prepared-campaign-10/` and `mechanism-screen-01/`. The live
launch-status file is authoritative for whether native validation, screening,
completion or failure has occurred.

```bash
cat /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-01/launch-status.json
tail -f /home/pgcotovio/Exact-OM/data/experiments-v2/mechanism-screen-01/launcher.log
tmux attach -t exact-mechanism-01
```

Detach tmux with Ctrl-b then d. Preserve allocation14220 and shell step14220.0.
The launcher receipt records the batch's own Slurm step. Compatible checkpoints
resume within this wave; immutable prior campaign selections remain external
history, and their progress files are not copied across campaign identities.

## Remaining blockers

E01 needs a common frozen scorer/selector/cardinality binding and extraction-only
replay; the current runner would repeat scoring and mix cardinality policies.
E10 needs its internal analytic selection phase before the two acceptance-training
arms; six analytic arms alone cannot complete the eight-arm family. E08 still
needs verified signed-identifier semantics and broader provenance deduplication.
E09 requires native runtime/resource admission on its specified D1 case; D0
cannot replace it. These families remain blocked, not empirically screened out.
