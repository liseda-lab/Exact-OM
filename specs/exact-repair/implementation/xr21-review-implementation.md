# XR-2.1 corrective implementation notes

This document accompanies the [REV-01–REV-14 acceptance map](xr21-review-conformance.json) for the [30 September implementation review](../14-implementation-review.md). It records the corrected execution contracts. Test outcomes, environment, source digests and the implementation commit belong in the separate [corrective validation receipt](xr21-review-validation.json); the mapping alone does not certify a pass. The earlier `xr21-validation.json` and experimental results remain historical evidence.

## Verification and durable recovery

Verification streams use bounded batches and a SQLite WAL journal. A broker validates each event against frozen assignment, theory, policy, obligation and backend identities, then commits its batch durably. The producer receives acknowledgment only after the controller has retained the committed-prefix receipt. A blocked validator, proof check or storage operation remains inside the supervised process tree. The controller continues sampled CPU/RSS and deadline checks.

The final positive report carries a bounded exact-coverage receipt. The complete event history stays in the journal; neither a partial positive stream nor duplicate completed obligations can supply missing coverage. A qualified negative event survives a later hang or broken frame. When a positive final report contradicts such an event, the negative evidence is retained and the discrepancy is explicit.

Search recovery retains journal references and the complete qualified source-exception artifact. The source artifact resolves immutable premises through the supplied input's source hash and binds source/import documents, original-source consistency, exception queries, policy and qualified backend evidence. Matching artifacts can authorize standalone continuation without an external baseline argument. Missing or incompatible legacy evidence triggers explicitly budgeted revalidation; it cannot silently remove obligations.

The durable search ledger reserves a dispatched stage's allowance and cleanup grace before execution. Ledger publication itself carries a conservative persistence reservation because interruption may happen after atomic publication. Successful later publications reconcile earlier reservations to observed elapsed work. An unsettled reservation remains spent on replacement; loading the same state does not renew allowances. The returned in-memory state can contain more recent measured work than the last durable publication after an exhausted budget or failed write. Preserve both the saved ledger and its journal directory when moving compatible saved work.

CPU/RSS limits are sampled across the process tree, not operating-system memory guarantees. Cleanup status is explicit. Storage and event replay run under their own remaining allowance; invalid or unfinished transactions do not become evidence.

## Learning and generation identities

A resolved `EffectivePreparation` binds graph/text limits, retrieval settings and interaction selection. Supplied graphs in new v3 execution must carry the matching resolved identity; an unqualified legacy graph must be rebuilt through that preparation path. Historical v2 execution remains under its declared contract. Risk reads exactly the graph's canonical admitted-support projection. Omitted-support records remain reporting data. Disabling risk bypasses its readout and loss.

A `SemanticTargetSpec` binds the query basis, desired/unwanted weights and weighted-family-mean convention. Initial teachers, acquisition, generated development and resumed collection consume this target explicitly. Semantic benefit, edit cost and feasibility stay separate. Complete qualified query outcomes may be reweighted when their dependencies match; historical labels retain their actual original semantics.

Complete-plan acquisition resolves finite per-stratum attempt quotas before verification. Candidate draws are a separate setting. Scheduled attempts interleave deterministically, keep duplicate origins and preserve unavailable, unknown and unvisited denominators. Quartet budgets count assignment attempts in complete groups of four. A per-draw probability is not presented as a deduplicated inclusion probability.

Training checkpoints retain the exact phase, epoch, minibatch order and offset, optimizer/model/RNG state, acquisition progress, development progress and selection state. An incomplete phase does not advance the epoch or consume completed-development patience. Checkpoints retain cumulative trainer elapsed time under the original total allowance; a replacement may consume only the remaining allowance. Exact continuation requires matching data, target, preparation, configuration and recovery identities. An older checkpoint with an ambiguous processed boundary needs a proven earlier boundary or an explicitly separate warm start with renewed model selection.

Optional support supervision predicts a declared qualified witness violation from permitted pre-decision inputs. Newly discovered proof contents remain evaluator-only labels for the same decision. Higher-order targets retain their joint structure; missing a sufficient support is not a negative witness label. Disabled support mode bypasses the head while preserving available-label counts. Known proof-supported conjunctions continue to produce symbolic cuts directly.

Protected family representatives satisfy the effective grammar and immutable context before a family is counted as covered. Logical emptiness differs from bounded search or compilation exhaustion. Candidate caps and final interventions cannot silently evict protected coverage. Progressive schedules must be nested in action families, typed vocabulary, grammar bounds and filters under unchanged context; incompatible narrowing rejects before execution.

Compiler cache maintenance tolerates entries disappearing during enumeration or deletion, serializes only publication/eviction and reports maintenance resource limits explicitly. Cold, disk-warm and in-memory reuse receipts distinguish measured compilation, serialization, save, restoration and current admission work. Unsupported metrics remain null. Study export discloses originating cold costs separately and never substitutes a warm-load duration for unobserved cold work.

## Annotation and historical compatibility

Semantic-fidelity aggregation uses a frozen schedule and canonical durable request/response provenance. Replayed responses, parser revisions and corrections cannot multiply a scheduled vote. Corrections retain their paid request accounting. Distinct scheduled repetitions remain distinct observations without being described as independent human experts. Training/evaluation reference a validated aggregate revision and the exact unique observations used.

The [corrective protocol](../protocol/xr21-review-conformance.json) and [schema](../protocol/schema-v3.2.json) record the new implementation, preparation, target, sampler, annotation and recovery contracts. Legacy manifests lacking these declarations remain historical inputs; they are not silently reinterpreted as compliant execution configurations.

Dependency-specific reuse is listed per requirement in the acceptance map. Qualified logical proofs can survive changes confined to risk scheduling. Changes to pools, objectives, graph admission, target meanings or selection policy invalidate the artifacts that depend on those identities. Prior experimental rows, paid raw responses, split manifests and cumulative costs are preserved rather than rewritten.

## Scope of qualification

The large-coverage regression exercises the actual verification producer with a controlled complete adapter and more than 10,000 obligations. Native ELK/HermiT coverage is checked separately on small OWL fixtures. These checks do not establish biomedical-scale runtime, memory sufficiency, out-of-distribution generalization or learned superiority over symbolic baselines.

Mock annotation tests make no live judge calls. Support-head gradients establish an implemented capability, not improvement over whole-plan risk with identical symbolic cuts. That comparison requires a separately designed experiment. This corrective task does not restart research campaigns, supervisors or production-matcher experiments.
