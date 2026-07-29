# E11 — Property-Equivalence Matching

**Motivation** (audit obs. 11; WP-F capability gap): object and data properties now flow
through the matcher, but plumbing success is not evidence that the class-oriented evidence
model produces good property alignments. Property identity is often expressed through
domain/range, super-property structure, inverse/characteristic axioms, and usage patterns rather
than labels alone. Results aggregated with classes would hide poor property performance because
property references are usually much smaller.

## Research questions

- **RQ11.1**: At fixed candidate-pool size, does the kind-aware pipeline retrieve and align
  object and data properties better than a lexical-only property baseline?
- **RQ11.2**: Which property evidence contributes independently: annotations, domain/range,
  `subPropertyOf`, inverse/characteristic axioms, or local usage edges?
- **RQ11.3**: Are gains consistent for object and data properties, and for exact-label versus
  lexically-dissimilar matches?
- **RQ11.4**: Do class-derived scoring/selector constants remain calibrated for properties, or
  is a property-specific profile required?

## Hypotheses

The full property bundle improves property-macro F1 by at least 2 points over lexical-only on
at least one eligible reporting track, with domain/range and `subPropertyOf` providing the
largest gains on lexically-dissimilar pairs. A class-default selector is expected to
over-abstain on the smaller property pools; target-label-free recalibration from E15 should
recover some of that loss without requiring property gold labels.

## Experimental change

No new entity-kind plumbing belongs here; WP-F is the fixed foundation. Add config-gated
property evidence groups so they can be ablated without changing the class or instance bundle:

- `labels_annotations`: property labels, aliases, and literal annotations;
- `signature`: anchored domain/range classes, with domain and range kept distinct;
- `hierarchy`: direct/transitive `subPropertyOf` evidence;
- `characteristics`: inverse-of, equivalent-property, functional/symmetric/transitive flags;
- `usage`: predicate-neighborhood summaries from assertion/projection edges.

All candidates remain within kind. Object and data properties use separate retrieval indexes
and results slices. Annotation properties remain out of scope until WP-F exposes them as a
supported matching kind.

Implementation boundary: experiment flags live under `matching.channels.property.*` and are
consumed by `exact/impl/datasets/pair_adaptive_context.py` and
`exact/impl/models/pair_adaptive_channels.py`; property-aware retrieval stays in the existing
dataset/candidate-generation layer. Selector changes are limited to kind-specific feature
normalization and remain flag-gated.

## Arms

Stage 1 holds retrieval and selection fixed: lexical-only / lexical+annotations / full bundle,
then leave-one-evidence-group-out ablations of the full bundle. Stage 2 runs the surviving
scorer with {class-default selector, property-specific supervised selector where train refs
exist, strongest target-label-free selector from E15}. The property-specific supervised arm uses
E18's ranking head and E19's fusion weights fitted on the property slice rather than a
separate property-only design; property results are a supervision-regime slice of those
experiments, and RQ11.4's calibration question is the per-kind scope arm in E19. A candidate-recall diagnostic compares
label-only retrieval with label+domain/range retrieval at the same mean pool size; it is kept
separate from the scorer ablation.

## Validation

Use the E00 inventory to admit only tasks with kind-resolved property references: OAEI-KG is
the primary reporting family; eligible Conference or other pinned pairs may be added when the
inventory confirms property gold. Report object-property and data-property results separately,
then their unweighted macro—never pooled with classes. Three seeds.

Primary metric: macro F1 over eligible task×property-kind cells. Secondary: candidate
recall@k, MRR/Hits@1, P/R, abstention rate, ECE, evidence coverage, and F1 slices for exact-label
versus non-exact matches. Report full-pipeline results and an oracle-candidate result so a small
property vocabulary does not conceal retrieval misses.

## Promotion

Standard criteria apply within each property kind. A property-specific default promotes only
if it beats lexical-only with a 95% paired-bootstrap CI excluding zero on the property macro
and no eligible property task regresses by more than 1 F1 point. A data-property result cannot
be inferred from an object-property win (or vice versa). If no track has enough data-property
gold for a powered test, RQ11.3 is recorded as inconclusive.

**Pre-registered criterion-2 override**: the per-property-task bound is 1 F1 point rather than
the plan default of 0.5 because the eligible property references are small; the property macro
CI and per-kind decision remain mandatory. The results note also reports whether the 0.5-point
default gate would have passed.

**Effort**: M. **Risks**: very small reference sets; domain/range anchors may import class
matching errors; usage edges can reward hubs. Report anchor coverage and degree-stratified
results, and include gold-class-anchor diagnostics only as a non-deployable oracle.
