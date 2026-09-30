# XR-2.1 repair actions and research implementation

Revision: **XR-2.1 — 30 September 2026**. Requirements describe the target research implementation. The inspected runtime baseline is `b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04`; writing this specification does not implement or qualify that target. [12](12-implementation-migration.md) tracks implementation gates, and [13](13-semantic-fidelity-supervision.md) defines semantic supervision. Historical executable protocol JSON remains unchanged.

**CG-A001 — Scope.** This file replaces the XR-P1 MLP-only/Conference-only first-delivery assumptions. All action families below belong to the research language. Smaller menus are named ablations or declared expansion stages, not unannounced delivery restrictions. The generated-only pilot is evidence about its captured finite cases and incomplete campaign, not about completed Conference/Bio-ML transfer.

## 1. Complete replacement semantics

**CG-A002 — Replacement unit.** A candidate contains all axioms that remain for its object. The original object is removed before its candidate is emitted. Keep and delete are explicit. EquivalentClass(S,T) is normalised to its two subsumptions.

**CG-A003 — Semantic classification.** Relative to immutable background B, Γ weakens α when B ∪ {α} entails every member of Γ. Strict weakening additionally requires B ∪ Γ not to entail α. For an original bundle Δ, weakening means B ∪ Δ entails every member of Γ; strict weakening additionally means that at least one member of Δ is not entailed by B ∪ Γ. Any generalisation premise used in that claim must remain fixed or be represented as a candidate dependence. An unknown entailment check supports neither weakening nor strictness. Record intrinsic template guarantees separately from qualified background-dependent proofs. Do not derive a useful weakening claim from an inconsistent background by vacuous entailment. Feasibility and preservation of intended meaning remain distinct requirements.

### Mapping actions

| Stable action tag | Emitted replacement/example | Interpretation |
|---|---|---|
| keep | original axiom set | No edit |
| delete | ∅ | Delete the object if eligible |
| retain_subsumption | S ⊑ T or T ⊑ S from S ≡ T | Remove the other direction |
| replace_endpoint | S ≡ T' or a declared directional replacement | Revision; not generally weakening |
| specialise_subclass | S ⊓ E ⊑ T; optionally retain T ⊑ S | Weaken the selected inclusion by specialising its subclass expression |
| add_necessary_condition | T ⊑ E, with all other retained axioms stated | Add a necessary condition to the replacement |
| complex_equivalence | {S ⊓ E ⊑ T, T ⊑ S, T ⊑ E} | Exactly T ≡ S ⊓ E |
| composite | Explicit compatible combination of the above | Classified by actual emitted axioms |

Mirror source and target templates explicitly. Record subclass/superclass roles rather than relying on informal side names. S ⊓ E ⊑ T and T ⊓ F ⊑ S do not entail S ⊓ E ≡ T ⊓ F in general.

Specialisation and adding a necessary condition are independent. The bundle {S ⊓ E ⊑ T, T ⊑ S} is useful alone. {T ⊑ S,T ⊑ E} gives necessary conditions without the sufficient inclusion. Adding T ⊑ E strengthens the first bundle, but the complete complex equivalence may be incomparable with original S ≡ T.

Weakening-only controls respect original relation direction. Reversal/strengthening of a directional input is never silently labelled weakening; any such revision requires an explicitly enabled template and evidence. Initial settings do not automatically offer reverse simple subsumptions for directional originals.

### Ontology-axiom actions

| Stable action tag | Replacement | Condition and consequence |
|---|---|---|
| remove_disjointness | Remove selected A ⊓ B ⊑ ⊥ | Preserves other pairwise assertions from an n-ary disjointness axiom |
| remove_superclass_conjunct | A ⊑ B ⊓ C → A ⊑ B | Weakening; loses C commitment |
| specialise_ontology_subclass | A ⊑ B → A ⊓ E ⊑ B | Weakening; activates Sat(A ⊓ E) |
| generalise_superclass | A ⊑ B → A ⊑ D | Weakening if fixed B_background entails B ⊑ D |
| generalise_domain | Domain(r,C) → Domain(r,D) | Weakening if C ⊑ D is fixed/proved |
| generalise_range | Range(r,C) → Range(r,D) | Same premise; changes object typing |
| generalise_existential_filler | A ⊑ ∃r.C → A ⊑ ∃r.D | Same premise; loses the narrower filler requirement |

Keep and eligible delete also apply to ontology objects. A component/import is provenance, not permission to delete an entire import. Select exact axiom occurrences; preserve duplicate fixed copies. Human-authored axioms may be eligible with a greater default cost and explicit evidence. Returned patches describe the repair view, not automatic upstream mutation.

Arbitrary negation/cardinality/role-chain revision is outside the initial generator grammar. Such constructs stay in the verification input. Non-class mappings have typed keep/delete controls; richer property/individual repair requires additional templates and tests.

## 2. Required worked fixtures

**CG-A004 — Fixture preservation.** Keep these fixtures as exact semantic conformance cases, including each complete emitted bundle and its activated expressions. They are not sufficient evidence of large-scale repair quality.

**Overlapping roles:** AuthorReviewer ⊑ Author_s and Reviewer_s; Author_t disjoint Reviewer_t; equivalences on both roles. Test deletion, retaining Author_t ⊑ Author_s, removing target disjointness, and removing a source superclass conjunct. Verify the entire monitored signature.

**Accepted papers:** Rejected_s ⊑ Paper_s; Accepted_t and Rejected_t subclasses of Paper_t and disjoint. Paper_s ≡ Accepted_t and Rejected_s ≡ Rejected_t make Rejected_s unsatisfiable. Endpoint revision to Paper_t resolves it.

Add E = ∃hasDecision.Acceptance, AcceptedSubmission_s ⊑ Paper_s ⊓ E, and Rejected_s ⊓ E ⊑ ⊥. Test the specialised bundle, independently adding Accepted_t ⊑ E, and the complex equivalence. Check Sat(Paper_s ⊓ E). Add Invited_t ⊑ Accepted_t and Invited_t ⊓ E ⊑ ⊥ to demonstrate that the extra necessary condition can be wrong.

**Participant inclusion:** The editable ontology axiom is Participant_s ⊑ Speaker_s. Fixed source axioms are Listener_s ⊑ Participant_s, Presenting_s ⊑ Participant_s, Listener_s ⊓ Presenting_s ⊑ ⊥ and Speaker_s ⊑ Person_s. Mappings are Listener_s ≡ Listener_t and Speaker_s ≡ Speaker_t; the target asserts Listener_t ⊓ Speaker_t ⊑ ⊥. The original makes both Listener classes unsatisfiable. Replacing the editable axiom by Participant_s ⊓ Presenting_s ⊑ Speaker_s preserves the speaker commitment for presenting participants and activates Sat(Participant_s ⊓ Presenting_s). Replacing it by Participant_s ⊑ Person_s instead preserves a broader type. Both remove the inherited speaker commitment from all listeners.

**Writer typing:** Source axioms are ImportedReview ⊑ ∃writtenBy.Software_s, Range(writtenBy,Person_s), Person_s ⊑ Agent_s and Software_s ⊑ Agent_s. Mappings are Person_s ≡ Person_t and Software_s ≡ Software_t; the target asserts Person_t ⊓ Software_t ⊑ ⊥. The existential requires a software writer, while the range makes every writer a person, so ImportedReview becomes unsatisfiable. Compare changing the range to Agent_s with replacing the existential by ImportedReview ⊑ ∃writtenBy.Agent_s. The former retains the software-writer requirement; the latter loses it and permits a person witness.

**Domain counterpart:** Replace the writer fixture's first two source axioms by SoftwareReviewer ⊑ Software_s ⊓ ∃writes.Review and Domain(writes,Person_s); keep the hierarchy, mappings and target disjointness. SoftwareReviewer is forced into disjoint person/software types. Changing the domain to Agent_s removes the person commitment from the subject.

For the overlap fixture, the original alignment contains Author_s ≡ Author_t and Reviewer_s ≡ Reviewer_t. Make AuthorReviewer ⊑ Author_s ⊓ Reviewer_s one editable conjunctive axiom when testing conjunct removal; keep the target disjointness as a separate eligible object. This fixes which original occurrence is replaced.

Treat each fixture as a standalone asserted theory containing exactly its stated axioms, all mentioned named classes monitored, no ABox facts and no source exceptions. Verify that the source ontologies are coherent individually. Unless the fixture explicitly combines edits, replace only the tested object and keep the other objects. Check all named classes, not only the displayed witness.

Required expected outcomes:

- Original overlap, accepted-paper, participant, range and domain inputs violate coherence.
- Each stated weakening/deletion alternative can restore coherence with the stated other choices.
- The no-invited-exception paper case admits both the specialised bundle and complex equivalence.
- With Invited_t, adding Accepted_t ⊑ E makes Invited_t unsatisfiable even though the specialised bundle remains feasible.
- Keeping only Accepted_t ⊑ Paper_s and Accepted_t ⊑ E shows the necessary-condition action without the sufficient specialised inclusion.
- Endpoint substitution, adding a necessary condition and complex equivalence are not generally weakenings of the original mapping.

A missing decision record is not a negative OWL assertion. A coherent outcome is not automatically the intended repair.

## 3. Candidate construction and costs

**CG-A005 — Shared construction contract.** Preparation, supervised training, development, inference and replay must share a captured construction contract: original bundle/occurrence, relation and kinds, eligible actions, typed observed menus, endpoint alternatives, grammar bounds, immutable background, qualified proof constraints, canonicalization, cost-profile identity and candidate budget. A phase-specific limit is explicit metadata. No phase may silently depend on teacher-injected candidates that inference cannot construct. This contract and its probability semantics are detailed in [10](10-constrained-generation.md), CG-005–CG-025 and CG-034–CG-037.

**CG-A006 — Deterministic elementary candidates.** Build applicable keep/delete/retained-direction states and all declared enabled endpoint alternatives independently of draws and circuit compilation. Their presence does not force their selection or imply they are safe. Locked/ineligible objects retain only permitted choices. A directional input does not acquire a reverse subsumption through a convenience fallback. An inapplicable action is recorded as such; absence of a retrieved endpoint is a coverage result. Materialize newly retrieved endpoint alternatives as actual complete replacement candidates on every construction path, with side/type validation, original relation orientation, provenance and costs. Merely placing their symbols in the graph or expression menu is insufficient.

**CG-A007 — Rich proposals.** Retrieve finite class/property menus from observed definitions, typed matcher alternatives and explanation neighborhoods; retain evidence origin and truncation. Generate new intersections/existentials using [10](10-constrained-generation.md), rather than limiting the study to copying existing restrictions. Preserve independent specialisation, necessary-condition and complex-equivalence templates, explicit retained directions, and both source/target mirrors. Apply bounds to canonical generated arguments and attached active expressions. Endpoint substitution is a finite typed action, not invented vocabulary. Non-class mappings keep their explicitly supported typed controls.

**CG-A008 — Semantic filtering scope.** Reject malformed or ill-typed syntax. Permanently reject a context-dependent action only when the encoded exclusion has qualified immutable support and is relevant to the policy. An expression proved empty relative to immutable B may be excluded when its use activates a satisfiability obligation. Do not permanently filter using an editable range, disjointness or hierarchy premise; removing that premise may make the action admissible. Such dependencies belong in conditional presence/activation cuts in the global loop. A missing assertion, unsupported reasoner fragment, failure to derive a contradiction or timeout is not a positive semantic certificate. See CG-012–CG-017 for the exact guarantee and the `hasDecision` range/disjointness example.

**CG-A009 — Pool admission and budget.** Normalize complete bundles and active expressions, deduplicate identical content, merge provenance and validate metadata before selection. Freeze candidates after deterministic controls, sampled candidates and budget selection have been reconciled. Retain every applicable mandatory elementary entry and an available representative per enabled action family before global rank filling. If the cap cannot fit mandatory entries, return `invalid_budget`; never silently remove a family. An empty retrieved/proved-admissible family has an explicit reason. No representative is fabricated to make a coverage table look complete. Generation failures may leave a usable reduced pool, but that is an identified partial-generation result, not completion of the intended language.

**CG-A010 — Freeze boundaries.** Recompute all utility factors on the final inventory and retain its content hash before the master starts. Proposal probabilities are not interchangeable with semantic utility. A change in effective coverage, expansion or changed evidence creates a new epoch and inventory/solve identity with a newly frozen objective; old certificates and upper bounds retain their original scope. Preserve a previously verified incumbent only with its original theory/proof identity and revalidate as required by the changed input. A candidate removed by a pool cap is a coverage omission, not a logical impossibility.

**CG-A011 — Cost contract.** Cost features include deletion, change to original relation, expression size, endpoint change, ontology edit, and human-authored ontology edit. Keep has zero edit cost. Count each explicit edit penalty once, separately from predicted semantic benefit. Record the exact profile and any semantic-priority conditioning as separate inputs. Reusing one axiom in several candidates is not a logical incompatibility. Simple additive costs count decisions; a distinct-content penalty needs an explicit encoding and a named objective variant. Duplicate fixed copies affect logical presence even when an editable occurrence is removed. Human-authored provenance is an observed attribute, not automatic evidence that an axiom is correct.

## 4. Runtime baseline and target implementation boundaries

**CG-A012 — Baseline facts versus requirements.** At the inspected commit, candidate/template construction, SDD WMC, neural inventory scoring, bounded generation workers and selected pilot results already exist. This specification does not describe all of them as unimplemented. The following target gaps require separate evidence before being closed:

| Inspected baseline observation | XR-2.1 target / acceptance evidence |
|---|---|
| Grammar compiler uses one right-linear vtree, disables automatic collection/minimization, and limits allocated manager nodes before final collection | Audited native reference ownership; controlled collection; allocation/live/dead/reachable and element counts; measured vtree alternatives; unchanged accepted support/probabilities |
| Generation has local caches within spawned workers | Verified immutable exact-artifact reuse across workers; optional separately qualified alpha-renamed schema reuse |
| Current grammar combines template alternatives and union slot vocabulary | Small applicable per-template/family SDDs with CG-019 branch-mass correction; no silent distribution change |
| Retrieved endpoint alternatives exist in retrieval records, while `mapping_grammar` does not pass newly retrieved alternatives to its elementary `mapping_candidates` call | Shared explicit endpoint materialization through preparation, training and inference; final-pool fixture assertions |
| Missing-candidate corpus control removes one candidate before subsequent generation can recreate it | Evaluator-only final-inventory omission and verified absence, separate from a missing-symbol experiment |
| Proposal compilation failure can skip proposal loss while value training continues; checkpoint criterion prioritizes cached decoded coverage/regret | Explicit actual proposal-supervision coverage and generated-pool eligibility/quality selection; cached inventory metrics kept separate |

The pilot's 79/90 development generation failures motivate these gaps; they do not prove the repaired design succeeds. Use the exact archived case/language/budget evidence in [12](12-implementation-migration.md), without treating historical losses as current implementation measurements.

**CG-A013 — Responsibility boundaries.** Retain existing modules where practical; names below describe responsibilities, not an instruction to reorganize files solely for aesthetics:

| Existing area | Target responsibility |
|---|---|
| `exact/repair/records.py`, `api.py` | Validated shared input, immutable/editable occurrence partition, complete policy and captured evidence |
| `candidates.py`, `grammar.py` | Complete action bundles, canonical expressions, typed finite menus, explicit template families and independent syntax acceptance |
| `retrieval.py` | Bounded observed menus, typed endpoint alternatives, explanations/provenance and omission accounting |
| `compilation.py`, `circuit.py` | Qualified SDD construction/transport, reference ownership, immutable cache, exact branch/component normalization and sampling |
| `pipeline.py` | Shared construction contract, family/round statuses, deterministic controls, final inventory/objective freeze |
| `model.py`, `learning.py` | Observable graph encoding and proposal/value/interaction supervision, distinct semantic/risk/cost roles from [13](13-semantic-fidelity-supervision.md) |
| `kernel.py`, `maxsat.py` | Exact selection over the declared inventory, justified cuts, pending unknowns and valid bounds |
| `owl.py` and qualified detector adapters | Sound proof-supported rejection and complete qualified acceptance with explicit fragment/query boundaries |
| `tools/repair/` | Frozen corpus/teacher provenance, failure-inclusive training/evaluation, matched comparisons and replay |

Reuse existing shared-core, solver and graph infrastructure. Do not replace a mature SAT solver, graph framework or knowledge compiler with an unqualified local reimplementation. A bounded sound detector may implement or adapt a specified known inference fragment with replayable supports and soundness qualification; it is not a replacement for a complete OWL verifier or a claim to implement a new complete OWL calculus. Follow [12](12-implementation-migration.md) for staged changes and the repository's shared-core dependency constraints.

## 5. Required integration fixtures

**CG-A014 — Contextual semantic boundary.** Add the exact immutable-range example from [10](10-constrained-generation.md), CG-016. Test the following independently:

- Grammar-valid `EXISTS hasDecision.Rejected` is provably empty when the range and Accepted/Rejected disjointness are immutable.
- Its use as an activated subclass expression is rejected under the non-vacuity policy, while a merely vacuous inclusion is not mislabeled an OWL inconsistency by itself.
- Making either premise editable prevents an unconditional local ban; replacing/removing the relevant support can make that activated expression satisfiable.
- An immutable duplicate support axiom keeps the proof applicable even after its editable copy is removed.
- An unsupported or timed-out detector supplies no fabricated safe/unsafe conclusion; final global verification remains mandatory.

**CG-A015 — Endpoint and language coverage.** Provide a case whose observed matcher alternatives retrieve a useful typed endpoint absent from the initial candidate list. Verify its identical complete bundle, orientation and costs in preparation, training, generated development and inference. Add wrong-side/wrong-kind alternatives, locked objects, `=`/`<`/`>` relations, and a cap below mandatory controls. Test newly composed restrictions absent from asserted syntax, independent necessary-condition bundles, every ontology-action premise and source/target mirror. An endpoint outside the captured observed signature is reported as an omission, not silently introduced.

**CG-A016 — Omission and aliases.** Remove a designated candidate after the entire generator and duplicate merger, ensure no elementary, representative or sampled path recreates its exact bundle/activation, and then run selection. Keep the intervention out of model inputs. Test multiple template encodings of that same candidate. Separately test removing a required symbol/constructor from the captured language and verify actual unreachability. Report whether another syntactically different repair remains semantically equivalent; a single omitted candidate is not automatically a no-solution fixture.

**CG-A017 — Failure propagation.** Inject family-level timeout, allocated-node cap, RSS cap, empty support, zero probability, numeric failure, corrupt cache, invalid binding and cancellation. Check that deterministic controls remain auditable, partial families stay in denominators, and no partially frozen objective is exposed. A reduced-pool solve reports its own inventory scope. If a mandatory candidate cannot be constructed or the cap is invalid, return the corresponding construction error rather than pretending to have a complete fallback.

**CG-A018 — Joint selection.** Include two candidates individually passing every local contextual filter whose joint theory is incoherent. Check complete verification, sound presence/activation feedback and exact-pool bounds. Also include a verified feasible but semantically undesirable repair: coherence is not intended meaning. The no-invited-exception/Invited_t paper contrast must preserve the difference between a specialised inclusion, independent necessary conditions and complete complex equivalence.

## 6. Delivery gates and acceptance

**CG-A019 — Ordered gates.** Complete and record the following before treating the target as validated:

1. Validate records, policy and complete action bundles; qualify deterministic endpoint/omission controls and the exact worked fixtures.
2. Establish independent tiny-language syntax/probability oracles and qualified contextual proof fixtures before optimizing circuit construction.
3. Audit compiler ownership/resource supervision and compare vtrees; then split families while preserving support, fixed-bit mass, mixture posteriors and duplicate likelihoods.
4. Add exact immutable cross-worker caching; introduce schema reuse only after explicit binding/isomorphism qualification.
5. Share construction across training/development/inference and freeze generated pools for verified supervision; expose skipped proposal losses and generation eligibility.
6. Re-run exposed pilot cases as engineering regression only; freeze revised choices before new held-out generated structures and real-data transfer.
7. Compare circuit/semantic-decoder arms under matched language and semantic conditions, including all compilation, proof/lookahead, verification and failed-attempt cost. Retain or replace circuits on measured evidence.

These are dependencies, not permission to omit the full research language. No gate authorizes editing a historical protocol or changing test labels to match a new implementation. New experiments require new versioned captures under [02](02-experimental-protocol.md) and [08](08-experiment-matrix-and-handoff.md); this specification creates no executable experiment configuration.

**CG-A020 — Acceptance record.** Each implemented requirement must link its code revision, configuration/input hashes, qualified backend/library versions, conformance result and any remaining limitation. Check exact emitted OWL axioms for every template and side; independent necessary conditions; ontology occurrence patches and duplicate fixed copies; non-vacuity after joint selection; no stale originals/closure; signed pair encoding; sound presence/activation cuts; unknown scheduling and valid global bounds; circuit distribution/canonicalization; endpoint and omission invariants; no hidden labels; typed semantic teacher rewards; and scoped replay paths. Actual backend capability checks remain necessary.

**CG-A021 — Honest outcome reporting.** Report separately construction/generation completion, circuit/contextual guarantee scope, useful coverage, verified feasibility, semantic benefit, edit cost, optimization status and total resource cost. The target may improve scoring without improving generation, or generate admissible candidates without finding the intended repair. Such distinctions remain visible in model selection and final tables. Production-readiness, Conference/Bio-ML benefit and superiority over semantic decoders require their own completed evidence.
