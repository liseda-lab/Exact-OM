# XR-WP3 — Guarded, qualified, and replacement class states

**Status:** proposed<br>
**Depends on:** XR-WP2 accepted for the target policy/profile; XR-E01 pilot eligibility gate<br>
**Unlocks:** G3, XR-E02, XR-E05, full-action XR-E03<br>
**Research boundary:** implements candidates needed for RQ2, RQ3, and RQ6. It cannot claim that
they preserve useful semantics until the named experiments pass.

## Objective

Extend the class compiler with bounded, auditable semantic revisions while preserving the finite
search space, profile completeness, state-to-axiom provenance, all-off fallback, and global final
verification.

## State families

### Guarded directions

Support both mirrors:

- guarded forward: `s AND Gs SubClassOf t`;
- guarded backward: `t AND Gt SubClassOf s`;
- mixed and two-way configurations formed from independently selected forward/backward modes.

A guard belongs to the same ontology side as its guarded antecedent. The initial grammar is:

```text
G := named class | intersection(G, G) | exists(object_property, G)
```

Generation is bounded by syntax depth, constructor count, structural radius, per-direction count,
and total state count. Canonicalization flattens/sorts intersections, removes duplicates, and uses
stable shared-core expression serialization.

### Qualification components

Represent target-qualified and source-qualified equivalence as decomposed components, never opaque
magic actions. For example, target-qualified equivalence comprises forward, guarded backward, and
the separate qualification `s SubClassOf Gt`. The optimizer may select valid strict subsets.

Qualifications enter a state only when their corresponding guard/evidence is present, profile
compatible, and independently provenance-bearing. Bundled versus independent selection is an
explicit XR-E02 ablation.

Independence is realized within the per-object state budget. When the enumerated cross-product of
components would exceed that budget, the solver adapter may use the factorized encoding permitted
by `01-architecture-and-contracts.md` §Master solver instead of silently dropping composite
states; the chosen realization is recorded in the inventory manifest.

### Endpoint replacement

Allow a bounded source or target alternative and apply any otherwise valid relation configuration.
Replacement is not classified as a logical weakening; the whole assignment always returns to the
oracle. Candidate sources are Exact-OM top-k alternatives, local parent/child/sibling entities,
trusted identifier/lexical links, relation-compatible neighbors, and expert-provided alternatives.

Every replacement records original and selected endpoints, retrieval rank/score, generator,
ontology-side validation, and reason for inclusion. Unsupported endpoint movement receives a
configurable utility penalty and cannot be hidden in a generic “changed” flag.

Replacement can make two distinct revision objects select states with semantically identical
generated axiom sets. Inventory generation records such cross-object collisions, and the engine
applies the constraint or utility correction required by `01-architecture-and-contracts.md`
§Master solver so identical semantic content is never counted twice.

## Guard admissibility

For a guarded forward antecedent `s AND Gs` (and symmetrically backward):

1. reject at generation if the immutable core already entails bottom;
2. reject or penalize if the core entails `s SubClassOf Gs` (redundant guard; config must say which);
3. require final assignment not to entail the guarded antecedent is bottom;
4. validate profile support for every constructor and referenced entity;
5. reject tautological/self/duplicate generated modules;
6. record all prefilter queries/results and final non-vacuity explanation data.

Non-vacuity is a conditional policy: the selected guard state literal participates in its conflict
even if bottom is entailed entirely by immutable axioms. XR-WP3 adds tests proving those conflicts
exclude the active guard without wrongly excluding assignments where that state is off.

Non-vacuity is necessary but not semantic usefulness. Recovery metrics separately require query or
reference/expert support.

## Candidate pipeline

Generation occurs in ablatable layers:

1. base states from XR-WP1;
2. named/asserted guard candidates from local definitions and subclass restrictions;
3. subtype differentia and structurally distinctive expressions;
4. relation-evidence existential guards;
5. independent qualification components;
6. target then source replacements;
7. bounded composite configurations.

Each layer emits a pre-budget ranked pool. A diversity-aware reducer retains at least one legal
state from each enabled family before filling remaining slots by hand-coded/proposal score. It
never removes off. Record prefilter and budget attrition so XR-E05 can separate proposal failure
from pruning failure.

An LLM integration, if added later by XR-WP4, may rank or instantiate only symbols and constructors
in the closed inventory. Parser failure, out-of-inventory symbols, or profile mismatch rejects the
proposal before state construction.

## Configuration

Add strict fields for:

- enabled state families;
- guard depth, constructor count, structural radius, pool and retained counts per direction;
- allowed object properties/namespaces and expression sources;
- qualification count and bundled/independent mode;
- replacement counts per side/source and maximum endpoint distance;
- total states per revision object and per-family minimum diversity;
- redundant-guard behavior and complexity penalties.

Candidate generation freezes its complete ordered inventory before an exact run. Adaptive
expansion is a separate anytime mode in XR-WP5 and cannot share exact optimality claims with a
post-hoc expanded search space.

## Tests

### Compilation examples

Assert exact canonical modules for guarded forward/backward, mixed, two-way, target-qualified,
source-qualified, and replacement configurations. Verify semantic mirror orientation explicitly;
source-side guard tests must fail if accidentally compiled on the target side.

### Logical/property tests

- plain forward entails its guarded-forward module; plain backward entails its mirror;
- off remains least informative and always available;
- target/source qualified decompositions reconstruct the expected three axioms;
- independent component subsets compile exactly as selected;
- baseline-vacuous candidates are removed;
- globally vacuous selected guards are rejected with a valid conditional conflict;
- replacement never bypasses global verification;
- enumeration shows every new no-good preserves all safe assignments on small cases.

### Budget/determinism tests

- identical snapshots/evidence/configs yield byte-identical ordered inventories;
- each enabled family receives its reserved diversity slot when legal;
- changing a bound changes the inventory/config hash;
- duplicate normalized guards from different sources merge provenance;
- cross-object collision fixtures show the objective does not double-count identical semantic
  content and the recorded constraint/correction replays;
- candidate counts and attrition reasons reconcile exactly.

### Profile tests

EL-compatible grammar remains on the declared EL path. States leaving it are rejected before the
master or trigger a whole-run switch to a declared complete DL adapter. Mixing incomplete profile
checks in one certificate is prohibited.

## Acceptance criteria

XR-WP3 is done when:

1. all rich compilation and mirror-orientation tests pass;
2. conditional non-vacuity conflicts pass exhaustive safe-assignment checks;
3. candidate inventories are deterministic, finite, diverse, and provenance complete;
4. every accepted guard/replacement reconstructs from the certificate;
5. unsupported/profile-changing states follow explicit rejection/switch semantics;
6. base-only configuration reproduces XR-WP2 results byte for byte except new schema-compatible
   metadata;
7. XR-E02 can enable every required family ablation without code changes;
8. XR-E05 receives pre/post-filter candidate traces and oracle-ceiling inputs.

## Experiment handoff

- XR-E02 decides whether richer states improve preservation and which primitives matter (RQ2/RQ3,
  H2/H5).
- XR-E05 evaluates proposal/budget recall and recovery bottlenecks (RQ6/H6).
- XR-E03 may compare joint/sequential optimization over the identical frozen rich state inventory.

No candidate family is promoted merely because it can produce safe outputs.
