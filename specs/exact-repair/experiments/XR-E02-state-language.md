# XR-E02 — Semantic preservation and state-family ablations

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP3; XR-E01 pilot with no false-safe outputs<br>
**Questions/hypotheses:** RQ2/RQ3, H2/H5<br>
**Primary outcome:** paired semantic-preservation gain at equal eligibility

## Authorized claims

1. Whether the frozen full class-state language preserves more correct/useful semantic content than
   exact deletion-only repair at the same policy eligibility.
2. Which preregistered state families contribute on named corruption/logical mechanisms.

It cannot claim candidate generation is generally sufficient (XR-E05), learning superiority
(XR-E04), or scalability (XR-E07).

## Variants

Use identical core, provisional alignment, policy, hand-coded external-aligned utility, solver, and
budgets. Vary only the available state inventory:

1. deletion only: provisional semantic state or off;
2. base directions: off/equivalence/forward/backward;
3. add target-side guarded backward only;
4. add source-side guarded forward only;
5. both guarded directions;
6. qualifications bundled with their qualified equivalence;
7. qualifications independently selectable;
8. add endpoint replacements;
9. full frozen XR-1 language;
10. candidate-oracle inventory for a clearly labelled ceiling on tractable cases.

Also run comparable established repair systems and no-repair descriptively under the common rules.
All XR variants use exact solving on the primary tractable corpus; any anytime large-corpus appendix
is separate.

## Corpus

- mechanism-aware corruptions stratified by over-general equivalence, reversed direction, missing
  guard/qualification, endpoint over-merge, correlated hub, and mixed conflict;
- public relation-aware tasks for mapping quality;
- Complex tasks/curated cases for expression recovery;
- captured matcher outputs with discovered repairable conflicts;
- logical guard/vacuity cases.

Freeze family quotas and total budgets so adding a state family does not silently increase every
other family's candidate count. Report both equal-total-budget and additive-language sensitivity
analyses if G4 selects them.

## Outcomes

Co-primary selection is fixed at G4:

- weighted query-basis preservation/precision relative to clean/reference integration;
- relation-aware mapping F1.

Secondary:

- confirmed correct information deleted/retained;
- external oracle utility/regret;
- recovery gain over best unguarded state;
- guard non-vacuity/non-redundancy and expression complexity;
- exact/bidirectional-subsumption/query similarity for complex references;
- eligibility, fallback, and unknown rates;
- per-mechanism action selection and semantic loss.

Any variant with a false-safe output is ineligible rather than assigned a quality score that can
win the comparison.

## Contrasts

- H2: full language versus exact deletion-only, pair-level primary outcome.
- H5: both-guard variant versus target-only in the preregistered unconditional source-to-target
  mechanism stratum.
- RQ3 family effects: adjacent variants above, corrected as one planned comparison family.
- Qualification decomposition: independent versus bundled.
- Replacement effect: full versus no replacement.

Report interaction with mechanism/domain without turning post-hoc strata into confirmatory claims.

## Decision rule

H2 requires zero eligibility degradation and a positive paired effect exceeding the G4 minimum
important difference on the selected primary outcome. H5 uses its corrected stratum-specific
threshold. If rich states help only synthetic cases, report that boundary explicitly; it does not
support a broad benchmark claim. The report must compare effect direction and magnitude between
synthetic corruptions and captured matcher errors; discordance is reported explicitly and bounds
any cross-setting conclusion.

## Artifacts

- identical-input variant manifest and inventory hashes;
- per-pair/per-component semantic outcomes;
- selected-state family and mechanism tables;
- guard/replacement provenance and non-vacuity replay;
- certificates and eligibility flow;
- corrected comparison report.

## Non-claims

- No unrestricted optimal repair claim beyond the frozen inventory/utility.
- A safe non-vacuous guard is not automatically a correct complex mapping.
- Family frequency is not evidence of causal value; only preregistered contrasts support it.
