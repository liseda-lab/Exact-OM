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

Status: preparing the verified fit bindings and launch. Update the operational
receipt and this handoff with actual numeric worker/supervisor steps after launch.
