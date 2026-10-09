# Exact-OM scoring throughput contract: 200 cold unique pairs/second

Recorded: 2026-10-09. Status: **specified_not_implemented_not_deployed**.

Inspected checkout: `328b9c5a`. Running G4 source:
`035a1a569c73620ce7c9fdbd41758ebd9eb8aa7f`, frozen under
`data/experiments-v2/runtime/g4-freeze-code-20261008-02/`. These are separate
execution identities. This document changes neither source tree used by G4 nor its
configuration, checkpoint, spending allowance, supervisor or queue.

This specifies implementation work needed to retain the approved final scientific
objectives while reducing repeated computation. **200 cold unique pairs/second is
an engineering acceptance target, not a measured result or a completion promise.**
It is not permission to reduce populations, seeds, candidate coverage or controls.
The experiment correction and deployment record must identify which scientific
cells consume this implementation before any affected execution starts.

Read with [E17](E17-promoted-stack-integration.md),
[run-once execution](RUN-ONCE-EXECUTION.md),
[checkpoint recovery](CHECKPOINT-RECOVERY.md),
[labels and submissions](LABELS-AND-SUBMISSIONS.md),
[hosted spending safety](HOSTED-SPENDING-SAFETY.md), and the cross-cutting
[performance policy](../03-performance.md). This specification supersedes no
scientific treatment or spending policy. Its explicit reuse contracts refine the
conservative cache identities described by older execution documents; unknown
dependencies remain a cache miss.

## 1. Observed evidence and problem boundary

The completed G4 baseline cells record 0.254–0.276 seconds per inference pair in
their `timings.json` files. Their four fitting stages total approximately 13.4
hours, within approximately 30.5 hours of combined execution. Fitting time is
therefore material, but those timings alone do not prove which fits are identical
or which implementation routine dominates inference.

The following findings are source inspections, not claimed profiler percentages.
Line numbers refer to the inspected checkout; use the named functions after edits.

| Finding | Inspected location | Consequence to measure |
| --- | --- | --- |
| `forward()` scores structural channels in a Python loop over pairs. Selected combined contexts are encoded two at a time. | `exact/impl/models/pair_adaptive_scorer.py:1732`; `pair_adaptive_evidence.py:689–704` | Many small encoder launches and repeated synchronization despite an outer batch of 512 pairs. |
| Existing evidence prefetch covers pool texts, explicitly excludes pair-selected combined contexts, and is off in the G4 environment. | `pair_adaptive_evidence.py:615–673`; G4 `environment.json` | Enabling existing prefetch alone cannot establish the target throughput. |
| Hosted-enabled `forward()` exits the numerical-cache wrapper before establishing its outer transaction and channel-scope context. | `exact/experiments/numerical_cache.py:288–323` | Nested channel calls can repeat full configuration hashing and commit separately. |
| Persistent encoder-cache access opens a connection, sets pragmas, looks up rows individually, commits and closes for each missing-text call. | `exact/experiments/runtime.py:635–782` | One or two cold texts can incur database setup, synchronous write and reread overhead. |
| General numerical-cache scope includes the whole configuration, inputs, dataset, role and seed. | `numerical_cache.py:59–153` | Identical primitives may be recomputed across task modes, fits and treatment arms. Removing identity fields indiscriminately would be unsafe. |
| Training aggregate lookup exists, but the completed fitting path writes only its local aggregate; no production caller publishes rows through `shared_training_scores(..., rows=...)`. | `exact/impl/trainer/fitting.py:250–379`; `numerical_cache.py:356–382` | A subsequent run cannot discover a newly completed aggregate through that shared lookup. The previous no-duplicate-storage correction must be preserved. |
| Example selection scans the edge list separately for each relation. Its cache omits the requested count and domain/range exclusion flag. | `exact/core/entities/ontology.py:261–311` | Up to O(relations × edges) work before template generation; repeated requests can return results for different arguments. |
| The current core configuration carries controlled difference perturbations even when the selected formulation is off. | G4 core `config.yaml`; `pair_adaptive_scorer.py:1769–1789`; `exact/experiments/evidence_diagnostics.py:23–89,116–147` | Ten synthetic replay variants, copying, hashing and diagnostic persistence per ordinary pair. |
| Provenance and channel reductions repeatedly transfer small GPU results to Python. | `pair_adaptive_evidence.py:310–357`; `pair_adaptive_channels.py:397–570,730–871,1471–1487` | Per-item and per-pair synchronization remains after encoder caching. |

The October 9 cache snapshot contained approximately 31.6 million numerical entries
and occupied approximately 146 GB including database overhead. The configured
128 GiB limit applies to payloads, not physical database size. This is historical
evidence, not a new allowance or an assertion of current free space.

Native ontology loading and projection remain mandatory. The optimization target
is Exact-owned orchestration, scoring, caching and reporting. No pyowl-core,
projector, pyELK or pyHermiT release is a prerequisite. Existing native projection
already produces canonical edges which Exact materializes through
`exact/ontology/projection.py:427–457`; a relation-example index consumes those
edges without implementing a Python ontology projector.

## 2. Non-negotiable result and execution invariants

- Preserve the three class-pair tasks, global and supplied-pool local modes,
  approved baseline/selected-stack/label-free contrasts, applicable published
  controls and genuinely stochastic seeds. The parent experiment plan owns the
  exact admitted matrix; this throughput target must not silently amend it.
- Global candidates come from the frozen global retrieval rule over eligible
  ontology populations. A local benchmark candidate pool must never substitute
  for global candidate generation. Preserve original local query identities,
  candidate memberships and denominators, including empty or unmatched cases.
- Preserve encoder weights and revisions, tokenizer, precision, pooling,
  truncation, evidence limits, evidence ordering, tie rules, thresholds,
  cardinality, calibration, fitted artifacts and request construction. Changes
  to these are scientific changes, not batching optimizations.
- Private test references remain unavailable to optimization, cache construction,
  selection and fitting. Unknown reference pairs do not become negatives. Label-free
  arms may reuse label-independent primitives, never target-supervised fitted data.
- Preserve exact numerical explanation inputs, selected fact identities,
  provenance links and competition information needed to reconstruct decisions.
  Generated rationales remain off. Diagnostic separation in T07 does not authorize
  deleting ordinary decision evidence.
- Existing completed predictions, evaluation artifacts and historical costs are
  immutable. Shared deterministic computation is measured once and attributed to
  consumers; copied artifacts are not independent stochastic observations.
- Preserve one heavy GPU worker in the retained interactive Slurm allocation.
  Modest CPU preparation may overlap only within the assigned cores and checked
  RAM/disk envelopes. No whole-job wall-clock cutoff is introduced by the forecast.
- Current paid-call limits, reservations, request size checks and deduplicated
  intervention notifications remain enforced. A faster scorer must not create
  unbounded outgoing requests or silently raise campaign/family allowances.

## 3. What the 200-pair target measures

### 3.1 Required measurements

Report three distinct rates; none may be substituted for another:

1. **Cold local scoring:** first computation of unique pair primitives with no
   reusable encoded text vectors or pair/channel results in memory or durable
   caches. This is the **200 pairs/second** target.
2. **Warm local scoring:** the same operation with explicitly declared reusable
   immutable vectors/primitives. Report the hit rate and new work separately.
3. **Campaign throughput:** total wall time for preparation, fitting, retrieval,
   new local scoring, reuse, hosted requests/retries, extraction and durable
   outputs. Only this measure supports the final completion forecast.

Cold local scoring starts before feature gathering for the first measured pair
and ends after all measured local scores and required ordinary evidence are
durably checkpointed. It includes cold text tokenization/encoding, evidence
selection, matrix operations, cache lookup/insertion, identity checks and normal
checkpoint serialization. Lazy feature work triggered inside scoring is included.
Prior native ontology preparation, model loading, candidate retrieval and fitted
head training are separately timed and included in the campaign forecast. Calling
an operation “preparation” does not make its time disappear from that forecast.

Hosted network/provider time is separately measured: the local target is not an
end-to-end claim that includes arbitrary API latency. Throughput qualification
uses recorded compatible responses or deterministic offline fixtures, with no
new paid calls. It retains the actual gate computation and counts which requests
would be required. Such fixtures cannot establish hosted quality or final runtime.
An actual scientific run must use its normal governed request path.

Offline fixtures remain in a separate namespace. They cannot enter scientific
request ledgers, final predictions, completed scientific checkpoints or
complete-scorer caches. Only independently verified, label-independent numerical
primitives may transfer from qualification under an explicit compatibility receipt.
Retaining useful production chunks never means promoting fixture-derived decisions.

Use isolated benchmark namespaces or explicit misses, never erase live shared
caches to manufacture a cold run. Record resident and persistent cache states
before measurement. Count a `(source, target, evidence/numerical identity)` only
once when it is actually computed. Exact-prefiltered pairs, duplicate requests,
replayed checkpoints and expanded reporting copies are excluded from the cold
numerator and reported separately. Repeating one easy pair is not cold work.

### 3.2 Workload and timing discipline

T01 freezes a label-independent workload manifest before measuring the optimized
path. It contains real source groups from each class pair, selected uniformly by
a fixed source hash, with separate declared stress strata for long/uneven evidence,
high candidate counts and missing channels. Preserve each selected source group's
candidate context and natural mix; do not select only cache-friendly or empty
pairs. Keep stress results separate from the natural-distribution aggregate.

Use at least 5,000 distinct model-scored pairs per class-pair workload and at least
three consecutive measured chunks. If the workload is faster than 30 seconds,
extend it to at least 30 seconds of measured cold work. The retained frozen
baseline may supply parity outputs; do not rerun an entire scientific matrix for
qualification. A bounded legacy parity sample must include every stress stratum.
Hardware/model initialization may precede the clock; any warming of measured
texts or evidence is forbidden in the cold measurement. Synchronize the GPU at
timing boundaries and include final asynchronous writes before stopping the clock.

Report elapsed time, distinct computed pairs, already available pairs, distinct
encoded texts/tokens, encoder calls and batch-size distribution, local-stage times,
cache hits/misses, scope rebuilds, transaction counts, bytes read/written, peak
RSS/VRAM, disk footprint, source-group sizes, route fraction and queue backlog.
Record median and worst chunk rates as well as total pairs divided by total time.
Capture source revision, configuration and dependency identities, GPU/CPU model,
assigned cores, device precision flags and input-manifest hashes.

The performance acceptance is **at least 200 cold local pairs/second on each
admitted class-pair workload for each distinct deployed scoring path**, without
changing its natural predictions. A workload below target remains a reported
failure, even if another task lifts the aggregate above 200. Cheap warm or
duplicate paths do not create additional cold qualification obligations; their
actual work and dependency equivalence must be proved.

Keep `throughput_target_status` separate from `execution_readiness`. A measured
180 pairs/second is a missed 200-pair target, not a scientific failure or an
automatic reason to leave a feasible approved experiment idle. Admission still
requires parity, coverage, resource/spending safety and a measured whole-program
forecast within the subsequently authorized execution plan. If those already pass,
a numeric target miss alone does not require a new allowance or scope approval;
report the miss and residual bottleneck explicitly. If the forecast instead implies
the rejected weeks/months-scale schedule, return a revised implementation/resource
plan before admitting that workload. Never kill a running scientific task because
an observed rate drops below the target.

## 4. Work packages and completion criteria

The IDs below are stable for the parent plan and machine-readable status. Every
package remains `specified_not_implemented_not_deployed` until its own evidence
is recorded. File names listed here identify existing implementation owners, not
permission to create parallel scoring pipelines.

### T01 — Instrument and count the actual remaining work

**Owners:** scorer/trainer timing, existing benchmark tooling, final recipe
materialization. Implement first; do not attach an intrusive profiler to live G4.

- First reconcile the tested frozen G4 source with the main checkout. The main
  `328b9c5a` tree predates frozen fixes for native source options, import checks and
  the `public_final_inputs` freeze contract; its older `freeze_final_selection`
  path still expects test references. Preserve subsequent main work while bringing
  forward the reviewed fixes and tests. Do not implement optimization on a base
  that loses these corrections or rebuild an existing public-inference facility.
- Add bounded phase counters around feature gathering, raw matrices, selection,
  combined-context encoding, fusion, numerical-cache identity/I/O, hosted waits,
  diagnostics and output. Sampling overhead must be reported and disabled or
  negligible in scientific execution.
- Materialize a small workload manifest and a final work-count manifest without
  reading final answers or starting a full final experiment. Enumerate consumers
  separately from unique fitted artifacts, candidate pools, text vectors, raw
  primitives and stochastic requests. Where enumeration is not yet possible,
  record an interval and its unknown dependency rather than an exact count.
- Report overlap of global/local and seed-specific candidate pairs, but do not
  assume overlap makes query-dependent features interchangeable. Separate work
  saved by exact reuse from work saved by faster computation.
- Publish a phase breakdown from bounded measurements. Select subsequent tuning
  based on it; a source-inspected hotspot is not a measured dominance claim.

**Done when:** the manifest and timing receipt reproduce counted inputs and
outputs, distinguish cold/warm/replayed work, and expose the numerator and full
wall-time denominator used for the target. Instrumentation must not require LLM
calls, final labels or a second qualification matrix. The reconciled source
inventory explicitly identifies retained public-input fixes and their tests.

### T02 — Keep numerical-cache batching active during hosted scoring

**Owner:** `exact/experiments/numerical_cache.py`, scorer entry/exit.

- Establish a scoped channel-identity context at every inference batch, including
  `use_llm=True`. Full-forward replay remains forbidden when hosted computation
  would otherwise occur; independent channel reuse remains allowed.
- Rebuild immutable scope inputs once per batch and once after a relevant mutation.
  The typed reversal diagnostic can mutate relation interpretation; `tau` and all
  other channel-affecting changes must produce the appropriate new identity.
- Group bounded local cache writes into transactions. Flush and release any write
  transaction before network waits, long training or yielding ownership. A
  per-batch context is not permission to hold a shared SQLite writer through a
  slow hosted request.
- Restore prior nesting/context state in `finally` paths. Do not swallow actual
  scorer exceptions or retry a scorer because cache persistence failed.
- Use in-memory reuse within the bounded batch for identical empty or repeated
  payloads; avoid serializing the same neutral result for every pair.

**Done when:** hosted-enabled and decision-off tests show bounded identity rebuilds
and transactions per chunk, unchanged scores/evidence, no hidden paid-call replay,
and context cleanup after exceptions. A mutated channel setting invalidates its
consumers. Cache failure remains an optimization miss, not a changed prediction.

### T03 — Batch persistent encoder I/O and bounded resident vectors

**Owners:** `runtime.cached_encoder_rows`, `scorer_common._encode_with_cache`,
`_DeviceEmbeddingRows`.

- Retain a process-owned connection/store handle, initialized once per compatible
  path and closed explicitly. Never share a live SQLite connection across a fork.
  Use bounded bulk lookup/insertion within database parameter limits.
- Deduplicate exact missing texts before tokenization; retain a scatter map so
  repeated input texts and original ordering remain unchanged. Return freshly
  computed rows directly instead of reading them back solely to rebuild the
  current result.
- Verify stored checksums, shapes and dtypes. Publish successful cache writes
  atomically. Interrupted cache writes can be recomputed; interrupted paid calls
  continue to follow the separate authoritative request ledger.
- Bound CPU and GPU resident caches by measured byte usage, not only row count.
  Transfer/gather vectors in blocks; a view must not retain an unexpectedly large
  backing allocation. OOM recovery reduces optional acceleration/batch size only,
  never the evidence set, candidate pool, precision or output population.
- Preserve model/tokenizer/revision/precision/hardware/role identity and the
  existing reviewed compatibility check. Cache sharing across roles is admitted
  only through T05's explicitly label-independent primitive contract.

**Done when:** cold/warm outputs retain order and duplicates; corrupted/truncated
rows, connection restart, concurrent readers, interrupted commits, incompatible
encoders and bounded OOM cases are covered. Receipts show fewer connection setups,
round trips and transfers without growing an unbounded resident cache.

### T04 — Two-phase pair-context batching and vectorized reductions

**Owners:** `pair_adaptive_scorer.py`, `pair_adaptive_channels.py`,
`pair_adaptive_evidence.py`, existing encoder helpers.

Implement one bounded execution path around the existing channel semantics:

1. Gather unique entity pools and relevant label/item texts for a chunk. Encode
   missing label/item vectors in bounded batches. Assemble support matrices in
   shape buckets or padded masked batches; padding never contributes to maxima,
   means, entropy, denominators or selection.
2. Apply the existing stable evidence selection to those support values. Preserve
   family limits, relation caps, specificity tie breakers, first-argmax rules,
   deduplication policy and original evidence order.
3. Construct the **exact selected combined context strings** for all pairs and
   channels in the chunk. Deduplicate by encoder identity, exact text and token
   limit, encode their cold misses together and scatter results back. This is the
   missing operation that pool-text prefetch alone does not provide.
4. Finish embedding/support combination, attributes, uncertainty and fusion;
   transfer required output vectors once per bounded block. Compute provenance
   row/column argmax and scores together, preserving link IDs and final sorting.
   Build the same ordinary evidence packets before the unchanged hosted gate.

Do not replace pair-selected evidence with per-entity average context or truncate
evidence to form uniform shapes. Do not use a cross-product of unrelated pairs to
inflate work or memory. Cache normalized vectors only under a numerical contract
that preserves dtype and the existing normalization operation.

Keep a bounded legacy path for parity and difficult numerical cases during
migration. Batched FP16 encoder kernels can differ from small-batch kernels due
to padding/reduction shape. Test real encoder outputs on the assigned GPU, not
only fake encoders. Record absolute/relative score error and every change to
selected evidence, tie ordering, routing, acceptance and final mappings. Numerical
tolerances cannot justify changed discrete decisions. If discrepancies exist,
preserve legacy computation for affected cases under a validated detection rule
or reject the optimization; a guessed epsilon is not a universal parity proof.

**Done when:** adversarial fixtures and real-data parity satisfy section 5, all
source groups remain complete, and measurements show actual larger context
encoder batches and fewer host/device synchronization points. Only observed
throughput establishes progress toward 200; batching is not itself acceptance.

### T05 — Reuse dependency-bound primitives and discover training aggregates

**Owners:** numerical-cache scope, fitting aggregate publication, existing artifact
store/recovery graph. Avoid another general workflow framework.

Replace one oversized identity with small explicit contracts only where actual
function dependencies are understood:

| Primitive | Required identity and boundary |
| --- | --- |
| Native prepared pools | Ontology/import bytes, native implementation/options, entity kind, selection and annotation semantics. |
| Text embeddings | Exact text, pinned encoder weights/tokenizer, token limit, pooling, dtype/precision and relevant execution compatibility. No learned encoder is identified only by its model name. |
| Pair support/evidence | Source/target and ordered feature contents, all consumed embedding identities, selection limits and tie rules, channel options, `tau` where neutral behavior depends on it, and any consumed anchors/templates. |
| Candidate-context features | Complete source/query candidate context and retrieval identity, alongside pair evidence. Same source/target under another pool is not sufficient. |
| Training score aggregate | Ordered training sources/pairs/features, explicit label semantics and content when present, scored-model/channel identities, complete candidate groups, scoring role, and numerical batch compatibility. |
| Fitted head/encoder | Above training dependencies plus fit algorithm, hyperparameters, label budget/folds, seed and supervised/label-free role. Independent stochastic fits remain independent. |
| Hosted response | Existing exact request/model/prompt/sampling/seed identity and authoritative request receipt. Never use a numerical cache to bypass spending or pretend a copied response is a new stochastic seed. |

A primitive may omit a downstream parameter only when that parameter cannot affect
its content. Do not delete `seed`, `role`, reporting-source checks or dataset
identity globally. If a seed acts only downstream, the deterministic primitive may
be shared while the consuming result retains its seed and fit/request lineage.
If evidence depends on anchors, retrieval, generated templates or training, their
actual content identities remain. Unknown dependencies fail closed to recomputation.

For training scores, publish a small immutable **aggregate reference** after the
existing aggregate is durable. It records schema, semantic key, aggregate URI,
checksum, row/source counts and upstream identities. A compatible run verifies and
reads that artifact; it does not copy all rows into a second SQLite payload or
rebuild scores just to create accounting. Use the existing artifact addressing
mechanism where possible; relocation updates a verified location binding, not
content identity. Dangling, corrupt or mismatched references are cache misses.
Retention must protect referenced aggregates until every consumer is complete.

Separate training-content compatibility from application authorization. Every
consumer still checks the disjointness required by its declared scientific role,
permitted label role, reference provenance and required negative policy. Fitting,
OOF and source-disjoint development/evaluation cohorts retain their exact separation
rules. Complete reference-free global deployment necessarily contains some entities
seen in training; do not reject its full ontology population merely for that IRI
overlap. Admit it only through an explicit frozen same-pair deployment manifest,
with no training/reference reads or refitting, and never count training examples as
held-out evidence. Organizer evaluation retains its own reporting denominator.
Sharing raw label-independent features does not share supervision. Reuse an already
fitted artifact only when its full training contract is identical; otherwise refit
the inexpensive head in its separately admitted training phase.

Publish the actual fitted deployment artifacts for each required same-pair,
arm/role and seed before frozen final inference. Do not assume a run's original
`config.yaml` points at artifacts written later under paths such as
`fitting/.../selector.json`. The final matrix driver must collect and checksum
the required selector, calibration/fusion/gate and trained-retrieval artifacts,
then bind them into immutable deployment configurations. Reuse
`tools/prepare_public_inference.py`, `exact/experiments/public_inference.py`,
`exact/utils/frozen_inference.py` and existing submission exporters. Frozen final
workers perform **zero refits and zero training/reference reads**, use
`run_eval=False`, and consume the preserved source-options/import and public-final
population bindings. Missing fitted bindings block that final cell; they do not
activate an implicit training fallback or a weaker model.

**Done when:** compatible global/local and seed consumers reuse only demonstrated
shared primitives; incompatible labels, anchors, pools, seeds, roles and weights
miss. A completed aggregate is discoverable by a second run without duplicating
its payload. Tests prove disjointness checks still execute on hits and report
logical result cells separately from distinct expensive computations. A frozen
final integration test resolves actual fitted artifacts and rejects every
training/reference access or implicit refit.

### T06 — One-pass relation examples from native-produced edges

**Owners:** `OntologyGraph.get_example_triples`, existing graph construction/cache.

- Traverse the existing canonical edge list once, collecting at most the requested
  number of examples per admitted relation. Stop when all admitted relations have
  enough examples. Preserve per-relation edge order and the caller's established
  relation iteration/order semantics.
- Key example caches by effective request count and domain/range exclusion options;
  human-readable conversion remains tied to the same ontology label identity.
  Define nonpositive-count behavior explicitly and test it rather than inheriting
  an accidental first-edge result.
- Avoid a second ontology-sized graph copy. Store only the bounded example index;
  native projection and reasoner semantics are unchanged. Shared verbalization
  templates require their exact ontology/example/prompt/model identities.
- Report separate times for example gathering, domain/range queries and template
  generation so a slow API call cannot be confused with a graph scan.

**Done when:** exact examples match the legacy positive-count contract on fixtures
and real native-produced edge samples, labels/options remain correct, and an
instrumented edge iterator proves at most one scan per uncached request rather
than one scan per relation. This does not require changing an upstream package.

### T07 — Keep synthetic development diagnostics out of full ordinary inference

**Owners:** resolved final configuration, difference diagnostic dispatch and output
policy. This requires a recorded diagnostic-output amendment, not a hidden flag.

- Keep E24's predeclared perturbation experiment and its retained results intact.
  Its ten replay variants remain available in their declared diagnostic runs.
- Disable synthetic replay generation for ordinary G4 successors/final inference
  only under the reviewed execution amendment. Historical G4 configurations and
  already committed records remain unchanged. Do not mutate a running worker.
- Preserve the natural difference formulation and ordinary scores, selected facts,
  fusion/gate inputs, uncertainties and reconstruction evidence. The off treatment
  must remain off; diagnostic removal must not change its meaning.
- Record which output fields are diagnostic-only and prove they are not consumed
  by prediction, fitting or ordinary explanation reconstruction. If a downstream
  consumer needs them, scope a retained bounded diagnostic artifact for that
  purpose rather than silently omitting required data.
- Account for the avoided calls, serialization and bytes. Do not rerun E24 merely
  to prove final inference can omit its already completed diagnostic work.

**Done when:** diagnostic-on/off parity yields identical natural scores, routes,
rankings and mappings; ordinary provenance remains reconstructable; full inference
does not invoke synthetic replay helpers. Diagnostic experiments still produce
their complete declared output and costs.

### T08 — Bounded streaming, quota safety and exact restart

**Owners:** existing trainer/scorer checkpoints, candidate/output shards, extraction,
storage guard and worker resource configuration.

- Work in bounded chunks with explicit source/query group boundaries. A large
  group may be processed internally in tiles, but its candidate-dependent
  statistics, local ranking and listwise decisions become final only after the
  complete original group is available. Never normalize or choose winners over
  an arbitrary tile instead of the declared pool.
- Global extraction remains global. Greedy sorting, mutual-best competitors,
  cardinality constraints and assignment components span chunk boundaries.
  Use bounded indexed/streamed reductions or a stable external merge where
  needed; do not perform independent per-chunk alignments and concatenate them.
  Preserve the existing tie order and component fallback rules.
- Persist ordinary numerical/evidence records in compact bounded shards, referring
  to immutable shared entity/fact data where equivalent. Do not persist a dense
  support matrix solely because an internal helper returned it; first prove which
  future consumers need it, preserve required reconstruction data and version the
  artifact format. Avoid one additional giant JSON/SQLite copy per consumer.
- Bound caches and shard buffers by **actual** memory/disk consumption. Account for
  database/WAL/index overhead, temporary atomic replacements, output/ledger growth
  and delayed cleanup. Keep the existing storage guard and reserve. Exhausting an
  optional cache stops cache growth, not the scientific task; inability to commit
  required results triggers a controlled resource intervention before more paid
  work is admitted.
- Local scratch may hold disposable caches or temporary sort files. Use the
  additional datasets drive only after validating its mounted path, capacity and
  durability requirements. The inspected `/mnt/datasets/pgcotovio` mount is an
  approximately 800 GB NFS resource, not local NVMe; `/tmp` is the local scratch
  candidate and still requires a current capacity/mount check. Do not forecast
  local-storage throughput for the NFS drive. Durable receipts/checkpoints cannot
  exist solely on disposable scratch. Never delete data merely because it is large.
- Commit content/checksum/count manifests atomically after shard durability. Record
  next source/query group, completed group IDs and exact primitive identities.
  Recover partial writes without losing earlier committed groups, repeating paid
  calls with a known response or marking incomplete global decisions final.
- Publish training aggregate references only after durability; remove redundant
  temporary shards only after verification and retention/reference checks.
  Preserve previous attempts and all historical accounting without retaining
  unnecessary duplicate scientific payloads.

**Done when:** interruption at each durable boundary, NAS read/write failure,
cache loss, quota pressure and process restart preserve committed groups, final
global/local decisions, paid-call accounting and source denominators. A projected
full-workload disk forecast fits the measured storage envelope, including overhead.

### T09 — Qualify, migrate and hand off the existing implementation

**Owners:** existing regression/benchmark suites, recovery/admission receipts and
documentation. No deployment is authorized by this specification alone.

- Deliver small logical commits with regression coverage for each changed owner;
  retain public configuration/record compatibility or provide an explicit versioned
  reader and migration. Keep the optimized path isolated from frozen active G4.
- Produce an impact map from changed functions/options to prepared pools, embeddings,
  channels, fits, predictions and reports. Reuse descendants only when compatibility
  is demonstrated; do not invalidate all historical work for an unrelated change.
- Start with immutable compatibility reads. Do not rewrite a live shared database,
  broaden a cache key, delete old artifacts or change a worker environment in place.
  New schemas/namespaces must fit storage capacity before creation. Back up or
  retain old manifests; a rollback restores old execution bindings without losing
  new paid-call usage or completed outputs.
- Run section 5 checks, publish section 6 evidence, then prepare an immutable worker
  snapshot, recipe and resource/spending bindings. The parent rollout plan controls
  approval, admission and restart at a safe boundary. Preserve the interactive
  allocation and use numeric Slurm steps, never cancel the enclosing shell job.
- Mark implementation, fixture parity, actual-GPU validation, throughput acceptance
  and deployment separately. No state advances because code compiles or a test
  double meets 200 pairs/second.

**Done when:** the complete evidence bundle is reviewable, required checks pass,
actual measured throughput and remaining-work forecast are recorded, and the
immutable deployment/rollback package is prepared. A running deployment is not a
prerequisite for T09: the separate C05 rollout step consumes T09's acceptance and
later records its own admission/launch receipt. If performance misses, report the
measured bottleneck and revised forecast; do not weaken parity, change scientific
scope or claim a target was achieved.

## 5. Required regression and scientific parity coverage

Extend the existing suites; use representative integration fixtures instead of
tests that merely repeat implementation formulas. Relevant starting points include
`tests/numerical_cache_test.py`, `scorer_device_cache_test.py`,
`pair_adaptive_experiments_test.py`, `candidate_quality_test.py`,
`grouped_fitting_test.py`, `fitting_recipes_test.py`,
`fitting_storage_retention_test.py`, `training_retention_recovery_test.py`,
`semantic_runner_checkpoint_test.py`, `storage_safety_test.py` and the hosted
spending/recovery tests.

The acceptance record must cover:

1. Uneven, missing and duplicate labels/facts; tied similarities; differing relation
   caps; exact token-budget boundaries; multichannel selection; both label-pooling
   modes; hierarchy overlap; attribute banks and natural difference variants.
2. Identical pair identities under different source/query pools; one source group
   spanning internal tiles; exact-prefiltered sources; duplicate query occurrences;
   empty groups and classes with no admitted candidates.
3. Numerical outputs, selected fact IDs/order, provenance links, evidence packets,
   uncertainty, routing, fitted-head inputs, local ranks and global competitors.
   Compare final mappings and scientific metrics, not only intermediate vectors.
4. Cold versus warm paths; blocked/disabled caches; invalid checksums; semantic
   mutation within a batch; connection exceptions; OOM batch reduction; bounded
   resident storage; no transaction held through hosted network work.
5. Aggregate discovery and retention; incompatible seed/fit/label/role/pool/anchor
   changes; safe reuse across consumers; repeated source-disjointness enforcement;
   no duplication counted as independent stochastic evidence. Include a full global
   population overlapping training entities: frozen deployment may admit those
   entities without reading labels/refitting, while an impermissibly overlapping
   held-out fitting/evaluation cohort must still fail its role-specific checks.
6. Recorded hosted-response replay plus fault injection before reservation, after
   send and before settlement. New requests retain existing limits and uncertainty
   accounting; rationale generation stays off; diagnostics never create paid calls.
7. Resume after shard creation, manifest commit, partial group processing, cache
   loss, output finalization and supervisor restart. Stable ordering and full
   populations survive every admitted recovery path.
8. Real pinned encoders on the assigned GPU, using declared cold workloads and
   adversarial boundary cases. Publish numeric differences and all discrete
   mismatches. Explain any legacy fallbacks and count their actual runtime.

Use exact equality for discrete structures, ordering, candidate identities,
selected evidence and final decisions. Unchanged cache/serialization/index paths
must preserve stored numerical values exactly. Any tolerance needed for changed
GPU batching must be justified from the existing precision and accompanied by
unchanged discrete decisions in the verification workload; passing toleranced
vectors alone is not an acceptance result or a proof for every possible input.

## 6. Evidence, forecast and completion record

Write the eventual evidence under a new immutable throughput-work directory;
this specification creates no such runtime directory. At minimum retain:

- Source/configuration/environment identities and a work-package implementation
  status table linked to logical commits.
- Frozen workload and remaining-work manifests; exact counts plus unresolved
  intervals, input roles and disallowed-reference checks.
- Old/new dependency and cache compatibility map, aggregate reference schema,
  restart/output-format version and retention rules.
- Regression receipts and actual-GPU parity records, including failures, fallback
  frequency, changed/disallowed outputs and remaining limitations.
- Cold/warm stage timings and counters, real disk/RAM/VRAM measurements, request
  counts/token forecast and a full remaining wall-time calculation.
- Prepared deployment/rollback instructions; later admission/Slurm receipts belong
  to a separate authorized deployment status, not the implementation result.

Compute the forecast from measured quantities:

`remaining time = nonshared preparation/fits + sum(unique cold work / measured rate)
+ measured reuse cost + hosted critical path + extraction/output + stated reserve`.

Respect dependencies and resource contention; overlapping stages cannot be added
or subtracted twice. If several final paths need different evidence, count each
unique path. Retain independent stochastic fitting/request work even when immutable
features are shared. Use measured per-task rates, not 200 by assumption.

For scale only, 20 million unique pairs at 200/second require approximately 27.8
hours of local scoring; 60 million require 83.3 hours; 180 million require 250
hours. Preparation, fitting, hosted work, I/O and recovery are additional. These
examples are not asserted final-work counts. A seven-to-ten-day campaign forecast
is acceptable only when the actual union, measured rates and all other phases
support it within the available node and approved spending/storage envelope.

Implementation completion means all necessary work packages and regression checks
are complete. Throughput qualification additionally requires measured cold-work
acceptance and real-GPU parity. Experiment completion requires the declared cells,
predictions and reports. Deployment and organizer-held test evaluation are separate
statuses. None follows automatically from this specification being committed.
