# XR-2 repair actions and research implementation

This file replaces the XR-P1 MLP-only/Conference-only first-delivery assumptions. All action families below belong to the research language. Smaller menus are ablations, not unannounced delivery restrictions.

## 1. Complete replacement semantics

A candidate contains all axioms that remain for its object. The original object is removed before its candidate is emitted. Keep and delete are explicit. EquivalentClass(S,T) is normalised to its two subsumptions.

Relative to fixed B, Γ weakens α when B ∪ {α} entails every member of Γ. Strict weakening additionally requires B ∪ Γ not to entail α. Any generalisation premise used in that claim must remain fixed or be represented as a candidate dependence.

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

Elementary alternatives are supplied deterministically. Retrieve finite class/property menus from observed definitions, matcher alternatives and explanation neighbourhoods. Generate new intersections/existentials using [10](10-constrained-generation.md); do not restrict the study to copying existing restrictions.

Reject malformed/ill-typed syntax. Do not permanently filter using editable axioms without dependencies. Baseline-impossible or redundant expressions can become meaningful after another edit. Unsupported verification is a scope/unknown issue, not evidence that an expression is logically false.

Freeze candidates after deduplication and budget selection. Retain keep/delete/directional controls and one available representative per enabled action family before global rank filling. If the cap cannot fit mandatory entries, report an invalid budget; never silently remove a family. An empty retrieved family remains a measured retrieval failure.

Cost features include deletion, change to original relation, expression size, endpoint change, ontology edit, and human-authored ontology edit. Keep has zero edit cost. Reusing one axiom in several candidates is not a logical incompatibility. Simple additive costs count decisions; any distinct-content penalty needs its own encoding.

## 4. Small implementation boundary

Suggested files, split only when useful:
~~~text
exact/repair/kernel.py      records, policy, orchestration, bounds, replay
exact/repair/maxsat.py      one qualified weighted MaxSAT adapter
exact/repair/owl.py         shared-snapshot diagnosis/verification adapters
exact/repair/candidates.py  grammar, templates, canonicalisation, finite menus
exact/repair/model.py       standard HGT/R-GCN layers, readouts, candidate/value heads
exact/repair/circuit.py     existing compiler/WMC integration, mixture conditioning
tools/repair/              corpus, symbolic teacher, training and study runners
~~~
Reuse existing infrastructure. Do not implement a new OWL calculus, SAT solver, graph framework or knowledge compiler. Graph and circuit dependencies are justified by the research comparisons, not by generic extensibility.

Sequence: records/compiler and finite conformance → reasoning/master loop → all actions and symbolic teacher → graph/value model → constrained proposals → matched comparisons and real-data transfer. The resulting prototype covers the full methodology; internal milestones are not substitutes for it.

## 5. Acceptance

Check exact emitted OWL axioms for every template and side; independent necessary conditions; ontology occurrence patches; non-vacuity after joint selection; no stale originals/closure; signed pair encoding; sound presence/activation cuts; unknown scheduling and global bounds; circuit distribution/canonicalisation; no hidden labels; typed teacher rewards; and all scoped replay paths. Actual backend capability checks remain necessary.
