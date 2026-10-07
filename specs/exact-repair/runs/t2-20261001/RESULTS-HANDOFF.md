# T2 preliminary repair results: shutdown and export

The authorized preliminary campaign completed on 7 October 2026 at 03:54 UTC.
The user requested shutdown and publication on 7 October. Supervisor step
`14408.338` was stopped at approximately 22:19 UTC. Only the interactive shell
and Slurm extern step remained in allocation 14408. Campaign `STOP`, supervisor
`STOP`, and supervisor `PAUSE` records prevent accidental dispatch or restart.
Restart requires a new user instruction. No experimental rows were replayed.

## Read these first

In the results archive, open `SUMMARY.md` for the scientific findings and
`tables/corrections.csv` for the 14 correction/qualification items. Some items
describe validated focused fixes or missing study qualifications; they are not
14 new unresolved software defects. `tables/fresh-evaluation-by-arm.csv` gives
denominators beside conditional quality. Never rank arms by a conditional mean
without considering unavailable results, different usable cases, and ancestry.

The final controller receipt is
`campaign/artifacts/expanded-preliminary-closure-001/completion.json`.
It authenticates completion of the local preliminary scope. Its adjacent
`costs.json` is the settled cumulative accounting record. Earlier reports retain
their original snapshots, including a pending closure and provisional costs.
Those historical fields are superseded by this later receipt, not rewritten.

The final evaluations have 18/576 usable generated-pool quality results,
50/576 usable common-inventory results, and 195/2,880 usable robustness results.
Development collection yielded 33 usable unique labels across 13/32 cases;
these labels remain excluded from fitting. These diagnostics do not establish
HGT superiority, the strongest-symbolic comparison, or broad generalisation.
The full training study, production matcher inputs, LLM labels and new cluster
launch remain deferred. API spend in this repair campaign was zero.

## What is preserved

The compact archive contains original final stage reports (including detailed
rows where the reports provide them), pilot training reports, protocols,
attempt outcomes and history, source identities, correction acceptance checks,
closure/accounting receipts, supervisor shutdown/publication records and CSV
summaries. All failed, unknown and unavailable outcomes remain represented.

`configuration/batches.json` preserves launch commands, resource settings,
package versions and frozen input hashes from each batch. Only repeated hashes
of whole code checkouts are omitted; each original batch manifest hash remains.
`configuration/selected-models.json` records the six pilot models and hashes.
`configuration/source-commits.json` lists campaign/batch source revisions.

This is a results export, not a standalone replay or training-resume package.
Raw datasets, unopened held-out payloads, caches, model weights/optimizer state,
raw worker logs and duplicate code snapshots remain on the server. No server
results were deleted. The exported verified-file index identifies supporting
evidence, including files not included in this compact bundle.

Original report bytes and absolute paths are preserved for provenance.
`export-manifest.json` maps each included source path to its portable archive
member. For paths under the original campaign root, the corresponding exported
path starts with `campaign/`. A referenced file outside this selection remains
server-side; do not infer it is part of the export.

## Code and documentation

Repository: <https://github.com/liseda-lab/Exact-OM>

The complete campaign implementation and this handoff are published on
`xr21-t2-20261001`. The separate qualification-test revision is preserved on
`xr21-qualification-005`. Source `795c461e` authenticated closure; `9c9af092`
produced the consolidated scientific report. Later handoff commits do not alter
the experimental source revisions. The main checkout and divergent `dev`
history are left intact. The archive's publication record contains verified
remote commit IDs.

To rebuild the export from retained server evidence, run from the campaign's
maintenance checkout (choose a new output filename):

```bash
/home/pgcotovio/Exact-OM/.venv/bin/python \
  specs/exact-repair/runs/t2-20261001/export_results.py \
  /home/pgcotovio/Exact-OM/data/exact-repair-xr21-t2-20261001 \
  /home/pgcotovio/Exact-OM/exports/repair-results-rebuilt.tar.gz
```

## Download and verify on your computer

Use the same SSH host/alias and VPN or jump-host configuration that you use to
connect to this node. `liseda-t2` below is the node's reported hostname; replace
it with your configured SSH alias if needed. The export operation does not
claim a download has occurred on your machine.

```bash
scp pgcotovio@liseda-t2:/home/pgcotovio/Exact-OM/exports/exact-repair-preliminary-results-20261007.tar.gz .
scp pgcotovio@liseda-t2:/home/pgcotovio/Exact-OM/exports/exact-repair-preliminary-results-20261007.tar.gz.sha256 .
shasum -a 256 -c exact-repair-preliminary-results-20261007.tar.gz.sha256
tar -xzf exact-repair-preliminary-results-20261007.tar.gz
cd exact-repair-preliminary-results-20261007
shasum -a 256 -c SHA256SUMS
```

## Next research decision

Before full training, qualify complete graph schemas and composite mappings,
resolve native worker cleanup, profile candidate generation and verification,
and establish adequate TRAIN-only supervision, stronger symbolic controls and
full optimizer/checkpoint operation. Preserve the frozen splits and accumulated
costs. On the future cluster, keep code in home, active data/results/caches on
parallel storage, and archive completed bundles after verified relocation.
