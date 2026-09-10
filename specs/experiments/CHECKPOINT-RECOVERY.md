# Checkpoint, resume, and bug-recovery contract

**Required new E00 implementation.** Current code has inference/additional-model checkpoints,
atomic/sharded explanation writes, and strict same-cell reuse. It does not yet provide this
stage-level relocation/repair contract. In particular, current harness identity includes commit,
whole specification hash, package set, and output path; those must not invalidate every artifact.

## 1. Three identities

1. Scientific design: task roles, source universe, treatment semantics, selection rule,
   endpoints, negative policy, and final contrast family. Freeze independently of implementation.
2. Numerical artifact identity: relevant normalized parameters, actual input/content hashes,
   split/role/entity kind, parent artifact IDs, model/tokenizer/prompt/decoding identities,
   relevant implementation/dependency hashes, random state/seed where material, and schema.
3. Attempt provenance: commit, full source/spec snapshots, host, environment, absolute paths,
   time, process, parent attempt, and repair record. Preserve all of it, but do not use an
   unrelated documentation edit or directory move as evidence of changed numerical semantics.

Use canonical relative artifact references plus a relocatable content-addressed store. Runtime
paths and modification times are not scientific identities. Hash actual bytes. Stage dependency
maps include transitive shared utilities: changing a common normalization function invalidates
all consuming stages. Unknown dependency impact defaults to conservative invalidation of the
affected scope; do not whitelist a commit or offer an unrestricted ignore-fingerprint flag.

Record relevant package/numerical-backend versions per stage; log the full environment separately.
An environment change is reusable only when compatibility is established for that artifact/schema.
Do not assume GPU, tokenizer, solver, precision, or model-revision changes are cosmetic.

## 2. Artifact dependency graph

| Stage | Durable outputs | Immediate semantic parents |
| --- | --- | --- |
| inputs | locked bytes, role/split manifest | public/licensed source identity and transformations |
| ontology | parsed snapshot, projection/hierarchy, provenance | input bytes, parser/projector/reasoner contracts |
| embeddings | entity/text vectors, tokenization metadata | normalized text, encoder/tokenizer/revision, pooling/precision |
| candidates | complete unlabeled pool per source, retrieval diagnostics | retrieval vectors/index, aliases, policy, entity universe |
| evidence | selected ontology features, raw channel components/top-m similarities, missingness | ontology, candidate pool, embeddings, evidence rules |
| pair_scores | per-channel s/q, fusion, uncertainty, trace | evidence, quality/fusion parameters or fitted artifact |
| llm_requests/responses | canonical request, state ledger, raw response, usage | exact evidence/prompt/candidate order/model/decoding/seed |
| fitted_heads | OOF predictions, model/optimizer checkpoints, fitted parameters | permitted train labels, train pool/features, training recipe |
| decisions | reranking, acceptance, calibration, NIL distributions | scores, applicable responses/heads, source groups |
| extraction/typing | global selected mappings, rivals, typed decisions | decisions, exact anchors, graph/relation policy |
| evaluation | source TP/FP/FN, ranking metrics, diagnostics | immutable predictions, reporting references, evaluator semantics |
| reports | tables, intervals, figures, current result-set manifest | evaluated cells, frozen contrast/statistical definitions |

This is a dependency graph, not a license to force every path through every stage. Train
embeddings/feature artifacts and reporting features have separate role identities. An LLM
response may be reused across routing policies only if the exact requested question and model
identity are unchanged; reapply the new gate/fusion separately. Label-dependent gates and
training artifacts must never acquire access to final references through cache reuse.

Persist sufficient raw components to replay analytic quality/fusion without model calls.
Where evidence selection itself changes, reuse eligible vectors/ontology artifacts but rerun
selection. Do not claim that an aggregate channel score is enough to reconstruct discarded facts.

## 3. Interruption and same-attempt resume

- Default checkpoint target: at most five minutes of work between committed boundaries.
  Checkpoint at completed source groups/pair batches, each optimizer-step interval/epoch, and
  every completed LLM request. Store the exact completed item IDs and the next deterministic
  cursor; row count alone is insufficient.
- Record any non-interruptible unit (ontology parsing, solver component, model loading) and its
  measured maximum duration. The loss bound is one such unit plus the checkpoint interval,
  not an unqualified promise of five minutes. Cap or shard long units where supported.
- Training checkpoints include model, optimizer, scheduler, gradient scaler, RNG states, sampler
  order/cursor, step/epoch, early-stop state, and all parent IDs. Save at optimizer-step
  boundaries; do not resume an altered objective from an old optimizer state.
- SIGINT/SIGTERM and a STOP file stop scheduling, finish or safely abandon the current bounded
  unit, flush a checkpoint, and leave status interrupted. Restart continues missing work.
- Write temporary files, fsync data where required, verify checksum, atomically publish manifest,
  then mark complete. A complete flag without all verified outputs is corruption. Keep the
  previous valid checkpoint until the next one has passed validation.
- Use per-artifact locks/leases and single writers. Concurrent consumers are read-only. Recover
  stale leases explicitly after confirming the owner is gone. Disk-full and partial-shard
  failures must not destroy the previous checkpoint.
- Checkpoint source-group boundaries before global selection; after a global assignment change,
  rerun the affected connected component, not just one pair within it.

For LLM work, persist request planned/sent/completed/unknown states and raw responses before
parsing/aggregation. Reuse completed requests. On ambiguous timeout, use a provider idempotency
key/status lookup if supported; otherwise preserve unknown and charge/record any retry. Do not
promise exactly-once paid calls when the provider cannot supply that guarantee. Cache responses
by exact request identity, not by source/target alone. Save tool-fetched evidence as well.

## 4. Resume from another directory

Planned CLI contract, to implement in the existing runner:

~~~text
--resume
--resume-from OLD_CAMPAIGN_DIRECTORY
--repair-record repair.json
--reuse-plan-only
--stop-after-checkpoint
~~~

Resume uses the latest compatible valid checkpoint. Resume-from imports references to verified
artifacts into a new attempt/output root, leaving the old directory untouched. It supports the
same or a changed checkout and moved storage; it does not require the old absolute path in the
artifact identity. Explicit copies or read-only hardlinks are acceptable; never mutate shared
hardlinked artifacts. A missing upstream path must be resolvable from the copied store/hash index.

Before any repair execution, emit reuse-plan.json: reused stage/artifact IDs, stages recomputed,
descendant invalidations, incompatible/missing fields, estimated incremental cost, and rationale.
The same plan drives execution and is included in final provenance. Reusing a completed cell
requires checksum verification of all consumed outputs, not just existence of a success marker.

For legacy v1 directories, write an import record with original fingerprints and source hashes.
Validate materialized inputs, models, schema, source universe, and role provenance. Reuse only
artifacts whose meaning can be established; missing provenance is not repaired by inventing a
hash or copying the current config over the old one. Rebuild only the unknown/affected stages.

## 5. Repairs invalidate descendants, not the whole campaign

| Change | Preserve if identities still match | Recompute |
| --- | --- | --- |
| README, progress text, output location | All numerical artifacts | Provenance/report packaging only |
| Plot/rounding/display bug | Predictions and authoritative metrics | Affected reports |
| Evaluator bug | Inputs, models, predictions | Evaluation, statistics, affected selections/result sets |
| Threshold/extraction bug | Evidence, scores, applicable fitted heads | Decisions/extraction as affected, evaluation |
| Fusion/uncertainty bug | Ontology, embeddings, candidates, raw evidence | Fusion, changed gates, dependent heads/decisions and evaluation |
| Prompt/probability parsing bug | Exact raw requests/responses if the requests did not change | Parsing/calibration/fusion; new calls only for changed requests |
| Candidate retrieval/alias bug | Valid ontology/encoder artifacts | Pools, candidate-dependent evidence/heads and descendants |
| Ontology normalization/projection bug | Verified input bytes; independent text vectors if proven unaffected | Affected graphs/evidence/heads and descendants |
| Training loss/label split bug | Uncontaminated input/features | Affected fit from compatible parents, not the bad optimizer state |
| Reporting-label leakage | Independently verified clean upstream artifacts | Quarantine all contaminated derivatives; reassess scientific validity |

A repair record contains: bug ID and description; old/new implementation IDs; reproduction;
affected contracts, stages, tasks and arms; dependency-derived invalidation closure; evidence
for reused stages; migration version if any; test results; whether reporting labels were exposed;
whether scientific choices are unchanged; and parent/new attempts. No arbitrary file edit may
mark an incompatible checkpoint compatible.

Replay a repair on synthetic and development cases first. Recompute all affected paired arms,
including controls, under compatible semantics; do not repair only the losing or winning arm.
Freeze the impact/re-run list before seeing repaired reporting outcomes. Retain original results
as superseded, with the reason, not deleted.

## 6. Statistical validity after a bug

Scientific validity and computational reuse are separate. A correct implementation repair can
reuse unaffected expensive artifacts without certifying that already-exposed test data are fresh.

- Before final exposure: repair, recompute affected development selections, and freeze the
  resulting final design; unaffected computation remains reusable.
- After final exposure, unchanged predeclared method/contrast with an objective defect:
  record a corrected evaluation of the same frozen experiment and rerun all affected cells.
  Disclose the correction; do not claim the rerun was a new untouched test.
- If the fix changes treatment, thresholds chosen from reporting, feature choice, task subset,
  stopping, or stack selection: it is a scientific amendment. Development artifacts remain
  reusable where compatible, but a new confirmatory claim requires fresh holdout/outer-fold
  evidence. Never automatically refreeze a more favorable stack on the same test outcomes.
- If an evaluator repair changes a development selection that has already led to final
  exposure, preserve the originally frozen final comparison as a corrected result; evaluate a
  newly selected stack only as exploratory or with fresh final evidence.

One current-result-set manifest selects the approved execution revision for each cell by repair
lineage, never best metric. Aggregation refuses mixed incompatible revisions, duplicate attempts,
stale selection records, missing controls, and invalidated descendants. A repair to an unrelated
component must not mark the entire scientific programme invalid.

## 7. Acceptance tests and operational proof

E00 is not ready until fixtures establish all of:

1. Interrupt during ontology/evidence/scoring, training, LLM ledger, extraction, and evaluation;
   resume and compare with uninterrupted execution within declared tolerances.
2. Crash between shard write and manifest publication; recover the last valid checkpoint.
3. Move/copy the output root and resume from the previous directory with no extra completed
   encoder calls or completed LLM requests.
4. Change only docs/output path: reuse numerical outputs; record new provenance.
5. Change fusion: reuse encoder/evidence counters, recompute fusion/descendants and affected
   pairs/heads. Change evaluator: zero new model calls.
6. Change model, pool, tokenizer, role, or training labels: reject inappropriate reuse.
7. Import a legacy directory with missing metadata conservatively; report what cannot be reused.
8. Simulate duplicate writers, stale locks, disk-full, corrupted checkpoint, ambiguous API
   timeout, and budget exhaustion; preserve prior valid work and honest cost.
9. Paired controls and source denominators remain identical after selective repair.
10. A post-reporting method amendment cannot silently regenerate a confirmatory selection.

Also perform one short real development interruption/resume and a harmless relocation replay
before the unattended campaign. This proves the actual models/filesystem path, not benchmark
quality. Keep an operator-readable recovery summary and exact continuation command in every
interrupted or failed attempt.
