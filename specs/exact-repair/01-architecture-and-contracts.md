# XR-2 architecture and shared records

**Schema namespace:** exact-repair/*/v2. **Status:** implementation specification, not installed capability.

## Placement and data flow

Any matcher adapter supplies shared ontology snapshots, a provisional alignment, score provenance and optional supporting evidence. Exact-OM's decomposition and explanations are useful inputs, not required truth labels.

~~~text
snapshots + provisional alignment + soft evidence
 -> bounded baseline diagnosis
 -> typed graph + retrieved vocabulary
 -> HGT node embeddings + per-object attention
 -> proposal heads -> constrained circuit -> complete replacement candidates
 -> candidate syntax encoder + candidate attention -> benefit coefficients
 -> explicit preference costs -> frozen weighted MaxSAT problem
 -> reconstruct selected theory -> policy verification
       feasible: retain verified incumbent
       infeasible: add a sound exclusion and repeat
       unknown: retain pending assignment; continue within budget
 -> repaired alignment + ontology patch + verification/search records
~~~

The encoder supplies node memory to both readouts. No conflict node owns separately trained heads. No single pooled ontology vector is the only input to all decisions. See [09](09-graph-and-neural-model.md) and [10](10-constrained-generation.md).

An optional outer graph/proposal refresh starts a new frozen optimisation round. Logical evidence may be reused only when its dependencies and current axiom-presence conditions remain valid.

## Records

All records have a schema version, canonical serialization and content hash. Scores are finite and accompanied by missingness/matcher identity; scores are not assumed calibrated.

| Record | Required content |
|---|---|
| RepairInputV2 | Snapshot/import identities; original complete alignment; typed evidence; policy; budgets; matcher identity |
| RevisionObjectV2 | Stable ID; kind mapping or ontology_axiom; original canonical axiom set; source/import provenance; human/generated/unknown authorship; eligibility/locks; candidate IDs |
| ReplacementCandidateV2 | Object/candidate IDs; complete canonical emitted axioms; action tags; selected directions/endpoints; expression ASTs; activated queries; provenance; compiler/version; structural cost features |
| BaselineReportV2 | Separate O_s, O_t, O_s∪O_t and T_0 checks; witnesses/support; proof scope; completeness per obligation; unknown causes; fixed exception set with evidence |
| GraphInputV2 | Node/edge types, role labels, feature provenance, retrieved menus, missing/omitted context and diagnosis revision |
| ProposalRecordV2 | Canonical encoding, template, grammar/menu/constraint hashes, circuit size, mixture parameters or reproducible reference, sampling seed and likelihood |
| ObjectiveV2 | Frozen unary/pair coefficients, semantic scale, explicit profile costs, integer quantisation, objective hash and pair set |
| VerificationReportV2 | Assignment/theory/policy hashes; obligation-by-obligation verdicts; input/query support; backend versions; explanations; resources |
| RepairResultV2 | Selected replacements; alignment; ontology patch; verification scope; logical/search/coverage status; incumbent value, bound/gap; pending assignments and failure events |

Candidate identity excludes learned utility, but includes activation and restrictions that affect feasibility. Deduplication by canonical emitted axioms alone is insufficient when activations differ. An axiom identity maps to all emitting candidates and to any fixed occurrence.

Ontology patches identify the exact original axiom occurrence/import and replacement. They are applied in the repair view, not written into upstream ontology files. Shared imports and duplicate axioms are tracked: editing one occurrence does not remove another occurrence. A request to suppress a semantic axiom across imports must explicitly enumerate its affected occurrences.

Property correspondences remain in the logical input. Class-expression templates apply only to class mappings. The initial non-class repair menu is keep/delete where eligible; unsupported richer property/individual revisions are identified, not coerced or silently dropped. Legal OWL punning uses (IRI, entity kind), not IRI alone.

## Result status is factored

- logical_status: VERIFIED_FEASIBLE, VERIFIED_INFEASIBLE, UNKNOWN, INVALID_INPUT, ERROR.
- verification_scope: full_owl, complete_supported_fragment, or partial_detection, plus exact ontology/query support metadata.
- search_status: OPTIMAL_IN_POOL, INCUMBENT_WITH_GAP, NO_FEASIBLE_IN_POOL, UNRESOLVED, ERROR.
- candidate_coverage: sampled, bounded_enumerated, or other precisely declared search universe.
- model_status: checkpoint/version, calibrated uncertainty if actually established, missing evidence and applicable distribution shift.

VERIFIED_INFEASIBLE describes a checked assignment or fixed-policy impossibility, not every candidate. NO_FEASIBLE_IN_POOL requires an exhausted exact master with only sound exclusions and no unresolved pending assignment. A partial detector cannot authorise VERIFIED_FEASIBLE for the full policy. A finite-pool bound does not upgrade logical scope.

With no verified incumbent, assignment/output alignment is absent and LB/gap are null. Diagnostic candidates can be retained but are non-authorising. With an incumbent, LB is its frozen utility and the bound includes pending alternatives. Zero gap supports an optimum only with the requisite verification and objective evidence.

## Reasoning interface

The existing hierarchy-only reasoner interface is not a repair verifier. Add a narrow adapter with inspect_support, diagnose_baselines, check_assignment and optional explain operations. Each returns explicit supported constructs/queries and result completeness.

Use the shared pyowl-core representation and existing optional backend architecture. Reuse qualified pyELK/pyHermiT or other compatible adapters; do not invent capability or require a second OWL model/path reparse. A qualified LogMap-style sound incomplete detector is an optional fast path. HermiT/ELK names in the methodology describe reasoning options, not a guarantee about installed wrappers.

## Evidence and replay

Persist input/import identities, eligibility, policy/exception evidence, canonical inventory, graph/proposal provenance, model/profile versions, frozen integer objective, accepted cuts, pending assignments and complete incumbent reports. Soft explanations are never merged into asserted axioms.

Safety replay reconstructs T_R and checks its full recorded policy without the neural model or optimiser. Optimality replay additionally validates exclusions, pending alternatives, and the exact master bound. Same-adapter replay establishes reproducibility, not independent verification. Artifact hashes establish integrity, not logical truth.

The parent process holds the last completed verified incumbent and valid bound. No unbudgeted verification begins after a deadline. Caches include ontology patches and all semantic dependencies; editing an axiom invalidates inferences whose support used it.

## Repository integration

Use lazy optional imports under exact.repair. Keep product repair disabled by default. Reuse snapshots, workers, canonical serialization, storage and configuration infrastructure. Prefer a small number of functional modules over generic solver/model/plugin frameworks. [06](06-first-experiment-implementation.md) gives the proposed layout.

Version-1 readers must reject the new object and status schema rather than interpret ontology edits as mappings. Explicit legacy conversion may reconstruct a restricted v2 problem, but cannot invent exception proofs, circuit likelihoods, complete labels, or verification coverage.
