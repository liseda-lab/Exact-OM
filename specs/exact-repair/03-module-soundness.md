# XR-2.1 reasoning, sound detection and complete acceptance

**Revision:** XR-2.1, 30 September 2026. **Status:** normative redesign requirements, not a claim of implemented or benchmarked capability. Audited runtime: b4c1ed0d5e12c45974bdcb4d230fb2ab6c6deb04. RE-* identifiers belong to this document; [04](04-minimal-exact-kernel.md) owns selection, cut encoding and bounds. Planned records use the v3 migration in [01](01-architecture-and-contracts.md); current runtime/archive records remain v2. Historical protocol JSON and the finite reference implementation are unchanged.

## 1. Responsibilities and guarantees

### RE-01 — Separate cheap rejection from complete acceptance

| Component | Admissible conclusion | Insufficient conclusion |
|---|---|---|
| Sound restriction-aware Horn detector | Supported inconsistency, unsatisfiable monitored/active expression, or prohibited entailment with sufficient asserted support | No detections cannot establish full-policy feasibility |
| Complete supported-fragment backend | Decided obligations on the actual supported ontology and query scope | Success after silently dropping unsupported axioms |
| Complete expressive backend | Decided obligations within qualified OWL/query semantics | Guaranteed completion on every ontology |
| Learned interaction/risk model | Proposal values, verification ordering, estimated uncertainty/cost | Logical cuts, exceptions or acceptance |
| Exact MaxSAT master | Optimum or qualified bound for the frozen Boolean problem | Semantic validity of an unchecked plan |

The detector reduces expensive checks by excluding assignments sharing proved conflicts. Routing, caches and incremental reasoning reduce an individual check's cost. Learned interactions improve proposal quality and verification order. Measure these effects separately.

An empty detector result means only that its rule set and budget found no violation. The full selected theory is reconstructed from fixed assertions and complete selected bundles. There is no mandatory OWL query per sampled expression; expression obligations apply when their explicit activation predicates are true.

## 2. Asserted inputs and complete policy

### RE-02 — Preserve semantics, typed identities and occurrences

Use the shared Java-free pyowl-core representation and existing optional backend architecture. Resolve imports before complete acceptance. Preserve document/import identities, canonical axioms, typed entities, anonymous identities and occurrence provenance. Legal punning uses (IRI, entity kind).

Each original occurrence belongs to the fixed remainder or one explicitly editable object. Editing one occurrence does not remove another occurrence of that axiom. Soft explanations and inferred edges never become fixed assertions.

A detector may use a resolved asserted subset while imports are incomplete if every premise has trustworthy identity. Its result has partial-detection scope. Missing imports prohibit full acceptance and negative-entailment conclusions about the complete theory.

### RE-03 — Freeze the full public monitored signature

Let C_orig contain public classes in the resolved original source, target and mapping axioms. C_inventory contains public classes introduced anywhere in finite replacement bundles, explicit endpoint alternatives and candidate expressions. C_policy contains public classes in additional policy queries:

~~~text
C_mon = (C_orig ∪ C_inventory ∪ C_policy ∪ caller_added_classes) minus {owl:Nothing}
~~~

Classes remain monitored even when their last asserted occurrence disappears. Use qualified fresh-entity semantics or semantically inert typed declarations. Private probes are declared separately and checked only through their active obligations; public candidate vocabulary cannot be disguised as instrumentation.

The same policy validator applies to preparation, direct construction, deserialization, checkpoint restoration and external inventory ingestion. Recompute expected coverage; validate query types, occurrence identities, activations and exception evidence. Reject silently narrower policies. An explicitly named restricted-coherence research policy needs distinct scope and cannot claim the default guarantee.

This closes the audited gap at api.py:314–341 and records.py:245–250. A replacement introducing fresh U ⊑ ⊥ must not pass default coherence because U was absent originally. Any later vocabulary/menu expansion starts a new epoch and reconstructs the policy.

### RE-04 — Freeze exact obligations and their activation

The policy requires consistency; satisfiability of all monitored non-exempt classes; every required positive entailment; absence of prohibited entailments; all active-expression satisfiability obligations; and declared edit/structural restrictions.

For S ⊓ E ⊑ T, check Sat(S ⊓ E), not Sat(E). Mapping, ontology, equivalent and composite replacements must capture every obligation implied by their full templates. Activation is an explicit Boolean predicate over selected candidates, not an action-name convention.

Unknown, unsupported, unavailable, timeout, query failure and resource exhaustion have distinct causes. A sound completed failure rejects one assignment. Acceptance requires every applicable obligation to pass completely on the bound theory/policy. Failure to prove required entailment is not proof of non-entailment.

## 3. Shared diagnosis and immutable source exceptions

### RE-05 — Capture four baselines once per case

Capture O_s, O_t, O_s ∪ O_t and T_0 = O_s ∪ O_t ∪ Ax(M_0) separately. Each report records theory, query scope, support/completeness, witnesses, timing and unresolved obligations. Source/target diagnosis normally uses its side-local public signature; combined baselines use their declared combined scope. Do not present different scopes as identical.

Useful detection/search may begin before baseline classification completes. Unknown remains unknown. Absence of source findings does not attribute a later defect to mappings; union-only contradictions are not source-only defects.

Matched arms consume the same captured baseline evidence. Charge common preparation once and report separately, or allocate identical declared cost to every arm. Additional arm-specific reasoning is charged to that arm. Backend/budget differences are explicit experimental factors.

The audited exact path performs three source/target/union checks inside selection at kernel.py:269–294, while greedy starts with alignment verification at study.py:324–368; study.py:461–464 selects these asymmetric paths. Remove this confound before efficiency comparisons.

### RE-06 — Acquire exceptions only from proved original source defects

An optional exception names a monitored class, side, original complete theory, exact query, and completed source-local unsatisfiability evidence. A premise hash alone is insufficient. For XR-2.1 require the original side to be established consistent before admitting an incoherence exception: an inconsistent source entails every class's unsatisfiability and must not license indiscriminate exemptions. Record source inconsistency separately; the repair can still edit it.

Freeze exception acquisition before optimization. Unknown source evidence, union-only defects and learned predictions cannot add exceptions. Never expand them after inspecting a selected plan. Strict-coherence mode has none.

Cache immutable exception proofs under original theory/query, normalization, backend/proof-trust and policy identities. Do not reclassify unchanged sources for every plan. Safety replay validates or re-establishes the evidence within its own budget. A repaired source view never rewrites original source provenance. A different exception regime must be explicitly specified and evaluated as a different policy.

### RE-07 — State original-input preservation separately from optimality

Default product behavior returns an already completely verified original input unchanged. That does not prove it maximizes arbitrary learned utility. Report actual utility/bound and bypassed optimization. Studies may disable preservation explicitly for matched objective comparisons.

## 4. Restriction-aware detector and proof supports

### RE-08 — Publish normalization and a sound completion calculus

The initial detector is sound and incomplete. Its minimum useful scope includes named subclass propagation, conjunction, disjointness/bottom, object-property domain/range, existential restrictions, and qualified universal/existential interaction. Role inclusions and further constructors require explicit sound rules and qualification. A named-class transitive closure alone is insufficient.

Normalize supported axioms into typed rules with exact provenance back to assertions. Fresh symbols are private and have a conservative-extension or consequence-preservation argument for every exported conclusion. Record any one-way reduction and its allowed inference direction. Unsupported axioms stay in the full theory and are listed as detector omissions; silently weakening them is forbidden.

Required illustrative sound behavior includes:

~~~text
A ⊑ B, B ⊑ C                         entails A ⊑ C
A ⊑ B, A ⊑ C, B ⊓ C ⊑ D             entails A ⊑ D
A ⊑ ∃r.B, B ⊑ C                     entails A ⊑ ∃r.C
A ⊑ ∃r.B, ∃r.B ⊑ C                  entails A ⊑ C
A ⊑ ∃r.B, A ⊑ ∀r.C                  entails A ⊑ ∃r.(B ⊓ C)
A ⊑ ∃r.D, D ⊑ ⊥                     entails A ⊑ ⊥
Domain(r,C)                          is equivalent to ∃r.Thing ⊑ C
Range(r,C)                           is equivalent to Thing ⊑ ∀r.C
~~~

These are soundness obligations, not a complete algorithm for arbitrary OWL. Keep antecedent contexts and existential witness obligations distinct. Separate existentials need not share witnesses; universals constrain each applicable successor. Never invent witness equality.

Thus A ⊑ ∃r.B, A ⊑ ∀r.C, B ⊓ C ⊑ ⊥ makes A unsatisfiable. In contrast, replacing the universal by A ⊑ ∃r.C can admit separate disjoint witnesses. Both are mandatory tests.

Cardinalities, inverse/complex roles, nominals, equality, ABox facts, datatypes and further Boolean constructors require separately qualified rules. Omission means incomplete detection. Rules over existing individuals alone do not establish existential class satisfiability.

### RE-09 — Every exported violation has sufficient asserted support

Track dependencies through normalization and completion. An exported failure identifies exact query/polarity, supported semantics, activation and sufficient asserted axiom set Γ. Shared proof DAGs are allowed. Support limits never remove unproved premises: retain a larger valid support, a valid whole-plan failure, or no exportable proof.

Monotone failures include inconsistency, C ⊑ ⊥ for a monitored class, E ⊑ ⊥ for an active expression, and prohibited entailment. The support must entail that exact failure. Ontology edits and mappings participate alike.

An explanation is sufficient support; a justification is subset-minimal support. Minimality is optional and outside the acceptance critical path. Bounded shrinking retains its last proved support when interrupted; unknown shrink results cannot remove axioms.

Record the trusted base: a checked derivation in qualified rules, or trusted output of a qualified sound backend. Identity hashes are not proofs. False-positive proofs are defects; declared misses are coverage limitations.

### RE-10 — Bind proof and event records independently of solver state

Planned ProofSupportV3 and VerificationEventV3 contracts include:

| Group | Required content |
|---|---|
| Binding | Epoch/input, theory, policy, query and activation identities; event sequence/ID |
| Semantics | Obligation kind, query AST, expected/observed truth and completeness |
| Support | Asserted support identities, normalization/probe dictionary, derivation or backend evidence |
| Applicability | Immutable dependencies, activation predicate and relevant exception identity |
| Qualification | Backend/rule/package versions, capabilities, imports and query support |
| Resources | Time/query counts, support limits, cache status and completion/failure cause |

Before recording a cut, the parent validates support presence, query membership, activation, identity and evidence qualification. Across epochs, a reusable proof must be revalidated and rebound to current emitters/activations.

### RE-11 — Export global presence cuts and respect nonmonotonic failure

Send Γ and activation to [04](04-minimal-exact-kernel.md), which accounts for all editable/fixed emitters. The producer observed during one diagnosis is not the entire semantic presence condition.

An unconditional fixed-only failure proves frozen-policy impossibility. A conditional fixed-only failure prohibits only its activation. Action tags and candidate indices do not establish applicability.

Definite failure of a required positive entailment normally gives a whole-assignment exclusion. Added axioms can restore the consequence. Stronger generalization needs a separate proof and cannot use ordinary monotone presence-cut reasoning.

### RE-12 — Contextual circuit bans require immutable proofs

A local circuit may ban a candidate only if its proof uses immutable epoch context plus that represented candidate/activation, and establishes failure for every removed global completion. Syntax and lock constraints remain separate.

Dependencies on editable ontology axioms, other mappings, replaceable endpoints or conditional choices belong in global presence/activation cuts. Never freeze them as universal local bans. Learned risk may affect logits/ranking but cannot hard-prune a semantically feasible candidate.

## 5. Complete verification and streaming

### RE-13 — Qualify whole-input/query support before cost-based routing

Qualification inspects selected assertions, imports, generated axioms, datatypes and every query family. Constructor lists alone do not establish profile requirements such as role simplicity or regularity.

The route is:

1. Reconstruct and validate the selected theory and active obligations.
2. Reuse applicable sound supports and run the bounded detector.
3. On no proved failure, enumerate installed complete routes for the entire input and remaining queries.
4. Among qualified routes, choose by a recorded measured-cost policy and budget.
5. Execute and aggregate completed results bound to the same theory/policy.
6. Accept only on full coverage; otherwise export failure or explicit unresolved obligations.

Prefer eligible native ELK for completely supported EL input/query combinations; use qualified expressive pyHermiT/native support otherwise. Package names, native flags, EL candidate grammar and successful construction do not prove completeness. Verify the actual public shared-snapshot APIs and complete-result semantics. No Java fallback, second OWL model or path reparse is required.

Mixed-backend checking is valid only when each query has a complete sound route over the same full theory, or a separately proved preserving module. An EL backend on filtered expressive input cannot authorize full-policy acceptance.

Unavailable backends, unsupported constructs, failed profile gates, incomplete imports and exhausted budgets leave unresolved obligations. Unsupported is not infeasible; another replacement may be supported.

### RE-14 — Private probes preserve expression semantics

Use qualified direct expression queries, or fresh definitions P_E ≡ E with a proved conservative-extension reduction, collision-free identities and retained dictionary. Probes are instrumentation, not public output or editable original assertions. Translate their proof supports back to original expressions and asserted premises.

All expression results concern the jointly selected theory. Prior satisfiability of E does not establish current Sat(S ⊓ E).

### RE-15 — Retain definitive failures across later interruption

Workers stream bounded, checked per-obligation events immediately on completion. The parent validates and durably records events before acknowledgement. A definite failure may stop rejection work; further diagnostics have a separate bounded allowance.

Later query, explanation, transport or cleanup timeouts cannot erase an acknowledged proof. At minimum it rejects that exact assignment. If monotone support was validated, its cut survives. Partial transport frames and unfinished explanations supply no new evidence.

Acceptance needs a final VerificationReportV3 coverage record. The parent recomputes expected obligations and verifies all are present, complete and passing. Missing queries cannot be hidden by shortened lists. Contradictory evidence for the same theory/query quarantines the backend path until resolved.

The audited owl.py:338–365 continues after failures, and kernel.py:243–247 replaces a timed-out whole-worker result with unknown. Streaming fixes that loss without treating partial positive results as acceptance.

## 6. Caches, incrementality and modules

### RE-16 — Cache semantic evidence under explicit dependencies

Default keys include selected theory, policy/activation, source-exception evidence, exact query, backend/rule/normalization versions, import completeness and support manifest. Smaller keys require a proved dependency contract. Similar graph neighborhoods are insufficient.

Distinguish:

- Immutable source/exception evidence.
- Exact theory/query results, rebound to new assignments only after checking policy and activation.
- Monotone violations reusable when all support axioms and activations remain.
- Satisfiability and non-entailment results, which cannot transfer freely to supersets.
- Required-entailment proofs reusable under surviving support, without proving other obligations.
- Unknown/resource results, which guide bounded retries but never establish truth.

Give operational failure entries backend-specific expiry/retry rules; one short timeout cannot permanently blacklist a plan under a later admissible budget.

### RE-17 — Qualify removal and replacement before incremental acceptance

Use supported delta/session APIs with reference counts for duplicate emitters. An axiom leaves the theory only when its final editable emitter disappears and no fixed occurrence remains. Invalidate dependent inferences and active probes; never promote closure facts to fixed assertions.

After timeout, cancellation or ambiguous update, discard or restore from a checked snapshot. Initially require a fresh reconstructed complete check before incumbent promotion. Remove that duplicate check only after differential qualification establishes incremental correctness for additions, deletions, replacements, imports, duplicates and all enabled constructs. Record the authorizing mode.

### RE-18 — A neural neighborhood is not a logical module

A subset can prove a supported monotone violation; its lack of a violation cannot normally establish full-theory feasibility. Required-entailment failure in a subset is also inconclusive.

~~~text
B = { C ⊑ ∃r.A, ∃r.B ⊑ ⊥ }
selected replacement adds A ⊑ B
an extraction restricted to mapped endpoints {A,B} can miss unsatisfiable C
~~~

Acceptance through a module requires preservation of every enabled query for every replacement in its reuse envelope. Include original classes, candidate public symbols/expressions, policy signatures, properties/individuals and symbols affected by ontology patches. Extension-only theorems do not automatically cover deletion/replacement.

Name the extractor, theorem, preconditions and rebuild conditions, and test against full checks. No generic claim that every locality module contains every relevant justification is assumed. Until qualified, modules accelerate rejection only.

## 7. Workflow, metrics and tests

### RE-19 — Reference orchestration

~~~text
prepare_reasoning(case, inventory, policy):
    validate imports, assertions, occurrences and public vocabulary
    capture/share four baseline identities and bounded evidence
    acquire and freeze RE-06 source exceptions
    freeze complete signature, queries and activation predicates
    build immutable provenance indexes and capability manifests

check_plan(epoch, assignment, budget):
    reconstruct theory and expected obligations
    validate reusable exact reports and applicable proof supports
    run detector; emit validated failures as they complete
    if any applicable failure: reject with retained evidence
    qualify complete routes for unresolved obligations over full theory
    if no complete route: return unknown with unresolved obligations
    execute queries; stream completed events
    on definite failure: reject; optionally extract bounded support
    on interruption: retain definitive failures, otherwise return unknown
    require complete passing coverage of every expected obligation
    require fresh reconstruction unless incrementality is qualified
    return verified feasible with final coverage evidence
~~~

Rejection describes one assignment, not the whole pool. An unknown baseline permits useful detection/search; unresolved acceptance or exception obligations prevent promotion.

### RE-20 — Report work at the correct granularity

Separate common baseline cost from each arm's detector, support extraction, master, backend startup/compilation, complete checking, transport, replay and cleanup. Count plans, detector passes, individual OWL queries, cache hits, restarts, unknowns by cause, accepted supports and eliminated assignments. One whole-theory pass is not one OWL query.

Keep all scheduled cases in coverage denominators. Report first incumbent time, final bounds, query scope and semantic quality separately. Fewer checks does not establish cheaper checks. Supervision and serialization follow K-12 in [04](04-minimal-exact-kernel.md).

### RE-21 — Required qualification and regression matrix

These are implementation acceptance requirements, not claims that current tests pass.

| ID | Fixture/intervention | Required observation |
|---|---|---|
| RE-T01 | Fresh U appears only in a replacement with U ⊑ ⊥ | Default policy monitors U and rejects selected incoherence |
| RE-T02 | Direct load lacks original/candidate classes | Reject or explicitly migrate before search |
| RE-T03 | Delete a class's last assertion | Keep frozen monitoring under qualified fresh-entity semantics |
| RE-T04 | Impossible private expression unselected | No inactive-expression rejection |
| RE-T05 | Sat(E), but not Sat(S ⊓ E) | Reject selected specialisation using whole antecedent |
| RE-T06 | Universal/existential contradiction in RE-08 | Qualified restriction rule finds supported failure |
| RE-T07 | Two separate disjoint existential fillers | No invented shared-witness contradiction |
| RE-T08 | Detector-unsupported cardinality/ABox/datatype | Preserve full input, record omission, forbid detector acceptance |
| RE-T09 | Prohibited entailment using editable premises | Export sufficient asserted support |
| RE-T10 | Absent required entailment restored by another plan | No unsound monotone cut |
| RE-T11 | Duplicate editable emitters plus fixed import | One deletion does not erase semantic presence |
| RE-T12 | Truncated proof support | No unproved cut; retain valid larger support or exact rejection |
| RE-T13 | Source unknown, union-only failure, inconsistent source | No automatic incoherence exceptions |
| RE-T14 | Consistent source with proved unsatisfiable class | Freeze proof and reuse across plans |
| RE-T15 | Completed violation followed by stall/cleanup timeout | Acknowledged proof survives |
| RE-T16 | Missing or contradictory streamed queries | Coverage fails or evidence path is quarantined |
| RE-T17 | Expressive input with EL candidate syntax | Actual full-input/query qualification controls routing |
| RE-T18 | Overlapping complete backends | Differential agreement; disagreement blocks qualification |
| RE-T19 | Removal invalidates inference; interrupted session update | Rebuild/rollback; no stale acceptance |
| RE-T20 | Out-of-signature C module counterexample | Unproved module cannot authorize |
| RE-T21 | Same theory, different assignments/activations | Rebind identity and validate active coverage |
| RE-T22 | Local circuit ban uses editable premise | Withhold ban; emit conditional global cut |
| RE-T23 | Exact and greedy share a case | Same baseline evidence/accounting treatment |
| RE-T24 | Support shrinking times out | Retain last valid sufficient support |

Also qualify property/individual mappings, punning, anonymous identities, imported documents, ontology edits, nonempty first candidates and consistent-but-incoherent theories. Logical correctness, operational bounds and comparative quality remain separate claims.

### RE-22 — Migration boundary

The audited runtime reconstructs replacement theories, distinguishes complete results from unknown, checks active expressions, and contains axiom-presence encoding. Its main loop uses whole-plan cuts, constructs fresh HermiT adapters and returns one reasoning envelope. The integrated detector, shared baselines, stronger validation, proof streaming and qualified reuse specified here are implementation work still to be completed. Historical synthetic results and unchanged protocol artifacts do not validate XR-2.1.
