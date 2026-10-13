# G4 restart with admitted encoder batching

The user approved preparing and restarting G4 on October 13 after reviewing the
combined optimization results. This authorizes G4 and its necessary recovery;
later experiments remain held for a separate review. The retained interactive
allocation is 14372 on liseda-03. CUDA graphs remain off.

## Evidence and execution choice

The completed offline validation in
`data/experiments-v2/g4-combined-optimization-20261013-01` scored the same 5,000
pairs in 250 complete candidate groups with fixed saved models and fitted heads.
Staged batching took 125.017 seconds (39.995 pairs/s), compared with 318.663
seconds for corrected sequential execution. The graph-enabled variant took
135.403 seconds and recorded no captures or replays. Its graph benefit is
unvalidated; the fixed-shape encoder smoke is a separate result.

Algorithmic invariants and cache checks passed. All variants produced the same
202 final mappings on this bounded panel; fitted scores differed by up to
0.00702722 and evidence/ranking drift was recorded. These are reference-free
compatibility checks, not quality estimates or scientific predictions. The
validation made no paid calls. Its result SHA256 is
`65c6c42e4f71cd09788057a24da6448480ae366c30f34b74b61b7dc20cb9dfcb`.

The admitted development recipe is `staged-32x64-mm-v1`: staged scheduling,
encoder regrouping, 32-pair blocks, 64-text batches, device `mm` arithmetic,
original tokenizer identity, and graphs off. Exact numerical reproduction of
sequential execution is not required by the amended acceptance contract.
Method, populations, precision, evidence validity, candidate coverage and fitted
models remain controlled. The recipe and an isolated scientific namespace are
bound to numerical caches, encoder caches and inference checkpoints.

The 200-pair/s target remains unachieved and is deferred as a prerequisite for
this G4 restart. The measured 40-pair/s rate excludes preparation, fitted
selection and live hosted latency; other ontology pairs remain unmeasured.
No further graph tuning or new optimization qualification batch is queued.

## Reuse and comparison

Recompute predictions for all eight G4 cells under the same execution recipe.
The four completed old baseline cells and the stopped 9,728-row core prefix
remain historical artifacts. They must not be inserted into the new predictions
or used as the new baseline timing. This preserves a meaningful paired quality
and inference-cost comparison.

Bind compatible saved learned artifacts and retrieval models explicitly, with
their checksums and original training provenance. Cells with complete fitted
state skip redundant training. Missing fits use the unchanged declared training
inputs. Never substitute validation labels or private test references for
training inputs. Keep the cumulative paid ledger and every earlier charge.

Five cells reuse their completed learned heads; three unfinished core cells fit
their missing heads normally. All four core cells reuse their saved retrieval
models. Compare fresh inference times directly; whole-cell times have different
amounts of previously paid fitting work, which remains recorded separately.

Apply the existing ordinary-inference amendment to turn off controlled synthetic
difference replays. Preserve natural difference evidence, citations and matching
calculations. Historical E24 diagnostics remain intact. Rationales stay off.

Fresh campaign preflight also identified a diagnostic import incompatibility:
E13's existing completed result serializes its absent published policy as
`null`, while the importer required `{}`. Accept either explicitly present empty
representation inside the strict completed-diagnostic checks. Missing or
populated policies, selected overlays and invalid signatures remain rejected.
Retain E13's original completed status and evidence without rerunning it.

## Operational handoff

Operational receipts belong to
`data/experiments-v2/g4-batched-restart-20261013-01`.
Snapshots preserve the previous registry, policy and pause. All pre-existing
pending batches are held before releasing G4. The supervisor may repair or
resume G4 within this recipe and the existing scientific/spending constraints;
it must request intervention for changed methods or allowances and notify when
G4 finishes and results need review. It may not dispatch later experiments.

New disposable numerical and encoder caches use bounded local disk. Compatible
native prepared data and inventory caches remain reusable. Durable scientific
outputs, checkpoints and accounting remain on shared storage. Both filesystems
retain independent free-space and growth guards. Existing token alerts and
hard paid-call limits stay in force; no wall-time kill limit is introduced.

Status: restarted on 2026-10-13 as numeric Slurm worker **14372.286**, registered
under `G4-batched-restart-20261013`. Supervisor **14372.285** runs separately;
the interactive shell **14372.0** and its allocation are retained. All seven
later queue entries are disabled pending review. The old runtime STOP remains
on disk; only its superseded global pause binding was removed.

Execution source is `51ab682750ff3c4c6d4994ea4e572220f184c892`; supervisor source
is `f9e052db524e90dc1b62c395ecfb7982b6199a3d`. Implementation commits are
`7bf9808e` (scientific batching admission), `e083b907` (verified fit reuse),
`f9e052db` (scoped worker and handoff), and `51ab6827` (diagnostic import fix).
No changes were pushed.

Verification includes 240 batching/cache CPU tests, 24 fit/recovery tests,
89 prepared-worker/accounting tests, 112 external-diagnostic/campaign tests,
263 protected runtime tests and 366 supervisor/dispatch/notification checks.
These suite counts overlap; they are not a unique total. Scoped typing,
formatting and five import-boundary contracts passed. Actual metadata-only
preflight resolved all eight G4 cells without configuration identity mismatches,
verified 31 historical producers and passed 42 upstream terminal-status gates.
It performed no scientific freeze or paid generation.

The shared experiment-tree guard permits 1,280 GiB with 100 GiB minimum free
space and 16 GiB growth reserve. New local caches have a 192 GiB tree guard,
64 GiB minimum free space and 16 GiB reserve; numerical cache disk/payload caps
are each 128 GiB. The reported shared filesystem had about 891 GiB free at
launch. An independent user-quota query was unavailable. Existing paid-call
thresholds are unchanged. This startup verification does not establish live
G4 throughput, its scientific result, or final-study readiness.

Operational evidence: `recipe.json`, `launch-validation.json`, `step.json`,
`release.json`, `g4-fit-restart.json`, and
`preflight/operational-final-review.json` in the restart directory. Its
`HANDOFF.md` records the monitor and log paths. The supervisor owns continued
monitoring and compatible recovery; no other experiments are authorized here.
