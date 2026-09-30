# XR-2.1 architecture and shared records

**Revision:** 30 September 2026. Baseline runtime uses `exact-repair/*/v2`; the additions below are planned version-3 contracts. No new runtime capability is asserted by this document.

## Architecture

~~~text
shared ontology snapshots + provisional alignment + available evidence
  → validated input, edit occurrences and frozen policy
  → four baseline reports + bounded sound conflict detection
  → observable typed graph and retrieved finite vocabulary
  → HGT node embeddings
       object-conditioned attention → proposal logits
       logits + grammar/context circuits → complete candidates
  → candidate syntax encoder + candidate-conditioned attention
       unary benefit + sparse pair benefit; explicit edit costs
  → frozen finite inventory and integer objective
  → MaxSAT produces a bounded high-utility shortlist
       sound detector filters proved violations and supplies reusable cuts
       plan-risk head orders remaining expensive verification calls
  → complete supported policy verification
       feasible: retain incumbent
       infeasible: add proved support cut or exact-assignment exclusion
       unknown: retain pending plan and its upper-bound contribution
  → repeat within the frozen epoch, or start a declared new proposal epoch
  → alignment + ontology patch + logical report + search bound + coverage
~~~

The shared HGT returns a matrix of node embeddings, not a single decision vector. Shared heads query this matrix for each revision object, completed candidate or selected plan. Conflict nodes represent detected supports; they do not own separate trained heads. [09](09-graph-and-neural-model.md) defines neighbourhoods, readouts and tensors.

The LLM annotator reuses the existing OpenRouter profiles/client/request ledger in `exact/llm/routing.py` and `exact/llm/ledger.py`; [13](13-semantic-fidelity-supervision.md) defines the narrow annotation adapter. The symbolic teacher and LLM annotator are **training-time** components. Their future labels never feed an earlier inference graph. A frozen model estimates benefit during inference. The hard verifier is still run before acceptance, irrespective of how high the model scores a repair.

## Stage interfaces and ownership

| Interface | Inputs | Outputs and invariants |
|---|---|---|
| prepare | Shared snapshots/imports, mapping semantics, evidence, eligibility | Occurrence manifest; elementary candidates; validated complete policy; no new parser |
| diagnose | O_s, O_t, their union and T_0; capability and budget | Per-query statuses, available proof supports, immutable exception evidence; streaming failures |
| retrieve/build_graph | Observed input and evidence available by this decision time | Typed graph, context omission masks, finite vocabulary/endpoint menus; no teacher answers |
| generate | Object context, menus, template grammar, immutable proof constraints, fixed logits | Canonical candidate bundles, encoding mass/provenance, per-family coverage and failures |
| score | Frozen pool, shared graph, explicit pair selector and calibrated benefit model | Unary/pair coefficients and structural costs; all recomputed after a pool change |
| select | Frozen objective, policy, logical cuts and scheduling ledger | Exact master proposal/bound, bounded shortlist; risk changes ordering only |
| verify | Reconstructed asserted selected theory, policy and qualified route | Completed obligation events and final complete/incomplete report bound to exact hashes |
| refine | Available new proofs and a provisional plan, remaining generation budget | Optional proposal-context update; changed pool/graph/model starts a new epoch |
| train | Verified exhaustive or sampled plan records and separately sourced semantic labels | Versioned proposal/benefit/risk parameters, target and calibration metadata |

One routine constructs explanation-aware interaction pairs in training, development and inference. One routine materialises endpoint alternatives on both prepared and direct paths. Existing elementary candidates cannot disappear because a circuit or worker fails. A reduced generation pool remains explicitly partial.

## Record boundary

All new persisted records have a canonical serialisation, schema version and content hash. Use `exact-repair/*/v3` for the changed contracts; Python class names may follow repository conventions. Do not write changed semantics under a v2 envelope. Implementation must add strict readers and explicit migration tests before writing v3 results.

| Record | Required fields beyond the existing semantic content |
|---|---|
| RepairInputV3 | Source/target/import content identities; full original alignment and relation interpretation; original and candidate public signatures; occurrence manifest; eligibility/locks; evidence provenance; budgets and declared generation universe |
| RevisionObjectV3 | Stable object and occurrence IDs, kind, original canonical axioms, import/all-emitter provenance, authorship human/generated/unknown, allowed action families |
| ReplacementCandidateV3 | Complete emitted axioms, expression ASTs, selected directions/endpoints, active obligations, action tags, structural costs, generation/source identity; semantic identity independent of scores |
| PolicyV3 | Monitored public signature, strict/source-exception policy and proof IDs, required/prohibited queries, activation definitions, eligibility and policy hash |
| BaselineReportV3 | Separate four theories, obligation masks and verdicts, support scope, exception evidence, reuse/accounting metadata |
| GraphInputV3 | Node/edge schema and feature version, masks for absent/zero/unavailable/truncated data, pair-selection identity, evidence cutoff, menus and omissions |
| ProposalRecordV3 | Grammar/menu/template/constraint hashes, circuit/artifact identity, original encoding and canonical bundle, posterior/mass information, seed/context, exact versus approximate likelihood scope, per-family completion status |
| GenerationReportV3 | Requested/effective language bounds; expansion history; elementary and generated counts; retrieval/grammar/compile/sampling misses; duplicates; resource/circuit telemetry and cache status |
| ObjectiveV3 | Frozen pool, pair set, raw calibrated unary/pair benefits, explicit costs, signed integer coefficients/scale, target-basis/calibration IDs, model and epoch hashes |
| ProofSupportV3 | Violation/obligation type; sufficient asserted support; occurrence origins; activation dependencies; backend/rule proof scope; parent theory/policy; proof/replay identity and completeness |
| VerificationEventV3 | Query identity, completed verdict, optional proof support, theory/policy/backend/capability hashes, monotonic receipt index, work/resource counters |
| VerificationReportV3 | All requested and completed/unknown obligation statuses, full input support, route and versions, final logical status, completed event IDs and resources |
| SearchLedgerV3 | Durable logical exclusions, temporary enumeration/scheduling exclusions, untested/unknown/retry plans, exact utilities, bound provenance, incumbent and epoch identity |
| PlanSampleV3 | Input/epoch/pool/assignment identities, sampler and sampling stratum, evidence-at-decision, verification masks, symbolic semantic vector, source/LLM label provenance, runtime/censoring, split parent |
| RepairResultV3 | Alignment and exact ontology patch or null; selected plan and replay records; logical/search/generation statuses; incumbent value, upper bound/gap; unresolved alternatives/failures; total resource accounting |

A proved support may be sufficient without being subset-minimal. A free-text explanation or LLM rationale is not a ProofSupport record. Proof hashes establish identity; correctness requires the qualified proof or replay procedure.

## Identity and dependencies

Candidate identity includes object, canonical emitted axioms and activation obligations. Multiple derivations of the same candidate are provenance entries with probability mass combined; they are not extra solver choices. An emitted axiom maps to every selected-candidate emitter and any fixed occurrence. Ontology patches identify exact affected occurrences, including imports; no upstream file is edited implicitly.

A compilation cache key covers the full grammar structure, variable mapping/order, templates, bounds, immutable semantic constraint proofs and compiler format/version. Neural logits are not part of structural compilation identity. Alpha-renaming reuse requires a reversible structure-preserving mapping; equal menu sizes are insufficient. A verification cache additionally covers materialised theory, active policy/query set, backend capabilities/version and relevant import/patch dependencies.

A neural neighbourhood is not a sound reasoning module. Logical-module and incremental-session reuse require the preservation conditions in [03](03-module-soundness.md), including deletions and replacements.

Non-class mappings remain in the logical theory. The initial non-class action menu is keep/delete if eligible; richer property/individual revisions require their own explicitly supported grammar. Use (IRI, entity kind) identity for legal OWL punning. Unsupported inputs are recorded and preserved for complete verification, not coerced into class mappings.

## Status and failure contracts

- `logical_status`: VERIFIED_FEASIBLE, VERIFIED_INFEASIBLE, UNKNOWN, INVALID_INPUT, ERROR.
- `verification_scope`: full supported input/query semantics, complete_supported_fragment, or partial_detection, with exact capability details; a broad string alone is insufficient.
- `search_status`: OPTIMAL_IN_POOL, INCUMBENT_WITH_GAP, NO_FEASIBLE_IN_POOL, UNRESOLVED, ERROR.
- `generation_status`: COMPLETE_DECLARED_ENUMERATION, SAMPLED, PARTIAL_RESOURCE_LIMIT, INVALID_LANGUAGE, ERROR, plus each family status and coverage.

A partial pool may yield an OPTIMAL_IN_POOL repair; this never claims an optimum in the requested larger language. VERIFIED_INFEASIBLE normally describes one checked assignment. NO_FEASIBLE_IN_POOL needs complete exhaustion with no unknown or untested alternative. An incomplete detector cannot label a full policy VERIFIED_FEASIBLE.

No incumbent means output alignment/patch and lower bound/gap are null; diagnostic candidate plans are non-authorising. With an incumbent, the lower bound is its exact frozen objective. All deferred and unknown plans still contribute to the upper bound. A finite fallback upper cap is valid but may be loose. A failure after a completed contradiction preserves that negative evidence; a partially completed positive pass cannot authorise acceptance.

## Replay and migration

Safety replay reconstructs the asserted theory and checks the complete recorded policy without the neural model or solver. Optimality replay also validates objective, logical cuts and all unresolved-bound accounting. Same-adapter replay checks reproducibility, not independent reasoner correctness.

A v2 reader remains available for archival reporting; it must label absent v3 data as unavailable. A v2 model cannot be resumed as v3 after adding risk heads, feature masks, pair selection or target definitions without an explicit training migration. Old labels can be reused only if their full input/policy/query identities and statuses remain valid. In particular, do not invent missing proof supports, semantic judgements, exact generation likelihoods or query completeness.

Keep immutable original artefacts and produce separately hashed migration outputs. Runtime repair stays disabled by default, and unrelated matching/frontend contracts remain unchanged. New record and experiment loaders must reject unsupported schema versions clearly.
