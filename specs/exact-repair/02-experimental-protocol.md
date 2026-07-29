# XR common experimental protocol

**Status:** proposed; pilot values freeze at G4<br>
**Applies to:** XR-E00–XR-E09

Every Exact-Repair study inherits this protocol. Experiment files may narrow it, but a deviation
must be preregistered before the confirmatory run or labelled exploratory.

## Separation from implementation acceptance

Implementation tests answer “does this component satisfy its contract on controlled fixtures?”
Experiments answer a named research question on frozen evaluation units. Test fixtures cannot be
reported as benchmark performance, and benchmark observations cannot waive a failed safety
contract.

All experiments operate on captured provisional alignments. A matcher is run once per frozen
input/config/seed, then every repair baseline consumes the identical captured mapping/evidence
artifact. This prevents repair comparisons from being confounded by rematching.

## Benchmark families

XR-E00 freezes exact releases, licensing, checksums, task subsets, and acquisition instructions for:

1. OAEI Bio-ML tasks with relation-aware references where available;
2. OAEI Conference and Anatomy;
3. OAEI Complex for expression/guard studies;
4. DISO-OAEI as a held-out cross-domain family when distributable;
5. outputs from several matcher systems, including Exact-OM, where licenses allow;
6. a mechanism-aware corruption benchmark;
7. a hand-authored logical mechanism suite.

The proposal's dataset names are candidates, not permission to use mutable “latest” data. No
confirmatory run uses an unpinned URL or release. Missing/non-comparable systems and datasets are
reported, not silently replaced.

### Mechanism-aware corruptions

Public OAEI families contribute few independent ontology pairs (Anatomy is one pair; Bio-ML a
handful), so pair-level tests on them alone are underpowered for several hypotheses. The
mechanism-aware corruption benchmark is therefore the primary confirmatory bed for H2, H3, and H5,
sized during the XR-E00 pilot to the preregistered power target; public-pair results are the
external-validity check and are reported with effect sizes and intervals regardless of
significance.

Each record contains immutable core hashes, clean/reference alignment, corrupt alignment,
corruption trace, intended mechanism, policy, seed, and all exact optima where enumerable.
Operators include parent/child/sibling/homonym replacement, subsumption strengthening/reversal,
guard/qualification removal, one-to-many hubs, correlated lexical errors, trusted/mutable
interactions, and high-confidence miscalibration.

Retain an instance only if verification shows the intended violation or a preregistered meaningful
safe-choice ambiguity. Split by clean ontology pair and corruption seed; variants of one clean
component cannot cross train/test boundaries.

### Logical mechanism suite

Include minimal, irrelevant-axiom, multiple-explanation, and trusted-mapping variants for:

- disjoint class paths and bottom;
- existential/universal clashes;
- qualified/unqualified cardinality and functionality;
- role disjointness, chains, asymmetry, and irreflexivity;
- nominals, equality/inequality, datatypes, negative assertions, and ABox inconsistency;
- baseline unsatisfiable classes and unsafe immutable cores;
- finite forbidden native subsumptions;
- vacuous/redundant guards;
- multiple equal optima;
- timeouts, unsupported constructs, and missing explanations.

Cases outside an adapter's declared complete profile test fail-closed status, not safe repair
quality.

## Experimental units and splits

The primary independent unit for broad quality claims is the ontology pair or matcher-output pair.
Conflict components, mappings, states, candidates, and solver iterations are nested observations.
Analyses use clustered/hierarchical treatment when nested observations contribute inference.

Split hierarchy:

1. ontology pair for the primary held-out split;
2. domain for transfer;
3. matcher identity for score-regime transfer;
4. corruption mechanism and seed;
5. conflict component for expert/preference labels.

No test entity text, embedding, conflict, exact label, or pair-specific calibration statistic may
enter model selection unless the experiment explicitly studies transductive adaptation. Such a
study is separate from offline generalization.

## Common baselines

Where semantically applicable, use:

1. no repair;
2. delete every mapping in a discovered explanation;
3. exact minimum-cardinality deletion;
4. confidence-weighted deletion;
5. direction-only exact repair;
6. sequential diagnosis then recovery with the same candidate actions;
7. Exact-Repair with hand-coded utility;
8. pointwise, pairwise, and decision-focused utilities;
9. executable established repair systems such as LogMap (including its conservativity-repair
   variant), AML, or ALCOMO when input/output semantics and policy checks are comparable;
10. reference-derived oracle utility on tractable instances.

Every returned baseline alignment receives the same final policy verification. Unsupported state
languages are identified as limitations; missing outputs, timeouts, unknowns, and fallbacks remain
visible. An existing system is excluded only by a frozen, documented compatibility rule.

## Outcomes and metrics

### Eligibility and safety

- consistency;
- newly unsatisfiable monitored classes;
- newly true forbidden entailments;
- active vacuous guards;
- false-safe count/rate under replay and independent verification;
- oracle unknown, timeout, fallback, and unsafe-core rates;
- percentage with replayable exact safety certificates.

The safety target is zero false-safe outputs for supported exact mode. Mean violation reduction is
not a substitute.

### Mapping and semantic quality

- precision/recall/F1 by relation type with relation-aware partial credit;
- confirmed-correct deletions and confirmed-incorrect retentions;
- source/target mapping coverage;
- weighted preservation and precision over a frozen query basis;
- competency-question agreement and hierarchy preservation;
- semantic loss per repaired conflict;
- external reference-derived utility and regret.

The frozen query basis must be derived independently of every signal used by state utility or
candidate ranking; XR-E00 records the derivation procedure and its independence argument.
Newly-unsatisfiable-class counts and conservativity-violation counts are reported as secondary
outcomes for comparability with published repair systems.

Reference absence is unknown unless a curated negative is available. Positive-unlabeled and
set-valued scoring rules must be frozen in XR-E00.

### Guards, replacements, and candidates

- top-k recall and mean reciprocal rank;
- non-vacuity and non-redundancy;
- logical equivalence or bidirectional subsumption to a reference expression;
- query similarity, tree edit distance, expression depth/constructor count;
- accepted recovery gain over the best unguarded state;
- candidate-oracle upper bound and failure attribution.

### Optimization and efficiency

- objective, best bound, absolute/relative gap, and regret;
- time to first verified incumbent and time to zero gap;
- master/reasoner/explanation time, calls, conflicts, conflict sizes, and shrink benefit;
- loading/state-generation/model time, peak RSS, and certificate size;
- scaling against axioms, mappings, revision objects, states per object, component topology, and
  OWL profile.

## Fair comparison rules

- Candidate inventory, policy, external evaluation utility, hardware allocation, and wall/reasoner
  budgets are identical for methods in a causal contrast unless the contrast is explicitly about
  one of them.
- Solver/model internal objectives may differ; evaluation utility is external and frozen.
- A richer-state method is compared to deletion at equal eligibility, not by averaging unsafe
  outputs into its score.
- Training, calibration, threshold/state-budget selection, and pilot inspection use no
  confirmatory test labels.
- Exact and anytime results are separate. A timeout with zero gap may be exact; a returned
  incumbent without a valid bound is safe but has no quantified optimality claim.
- Warm caches are used only in a separately labelled deployment scenario. Cold-cache runs are the
  primary reproducible performance comparison.

## Statistical plan

- Report every ontology-pair result and paired effect, not only pooled mapping counts.
- Use paired bootstrap confidence intervals for pair-level effects and a paired nonparametric test
  when the frozen number of pairs supports it.
- Report effect sizes and intervals even when a threshold is not crossed.
- Use multiple training seeds; distinguish data-split, corruption, and model randomness.
- Use clustered bootstrap, mixed effects, or another preregistered hierarchical method for conflict
  components nested within pairs.
- Correct the preregistered action-family and robustness families for multiple comparisons.
- Do not impute failed/unknown runs as safe or as zero-quality; report them as competing outcomes
  and provide a clearly labelled sensitivity analysis if needed.

Numerical superiority/margin thresholds are fixed after the XR-E00 pilot and before confirmatory
test evaluation. The zero false-safe gate is fixed now and is not relaxed by the pilot.

## Reproducibility record

Every run records:

- git commit and dirty-tree patch hash;
- Python, Exact-OM, `pyowl-core`, reasoner, solver, model, and evaluator versions;
- hardware/OS, worker counts, resource limits, and cold/warm cache state;
- dataset/ontology/reference/import hashes and licenses;
- captured provisional alignment/evidence hash;
- policy, compiler, candidate inventory, utility, query basis, and config hashes;
- all seeds and split-manifest hash;
- certificate, replay report, logs, metrics, and failure events.

Aggregate reports are regenerated from per-run immutable records. Hand-edited result tables are not
authoritative.

## Pilot and confirmatory phases

The pilot may choose state budgets, numerical go/no-go margins, time budgets, query-basis weights,
and feasible benchmark subsets. It must not use confirmatory test outcomes. At G4, XR-E00 writes a
frozen protocol manifest containing:

- primary/secondary outcomes and direction of benefit;
- dataset/split/exclusion manifests;
- baseline versions/configs;
- sample-size or attainable-power rationale;
- statistical models and multiplicity families;
- hardware/budget matrix;
- decision thresholds;
- analysis-script hashes.

Any later change is a versioned amendment with timestamp and rationale. Analyses affected by the
amendment are exploratory unless rerun on a new untouched evaluation split.

## Reporting template

Each XR experiment report contains:

1. question/hypothesis and authorized claim;
2. deviations/amendments;
3. eligibility flow table for every method;
4. primary effect with interval and pair-level plot/table;
5. secondary and mechanism-stratified outcomes;
6. unknowns, failures, fallbacks, and excluded/non-comparable systems;
7. runtime/resource accounting;
8. artifact manifest and certificate replay summary;
9. conclusion stated within the experiment's non-claim boundary.
