# XR research contract

**Status:** proposed; freeze at G0<br>
**Applies to:** every file in `specs/exact-repair/`

This document bounds what Exact-Repair is intended to establish. It is a claim contract, not an
implementation design. Any experiment that changes a primary outcome, exclusion, dataset split,
or hypothesis after G4 creates a new suite revision or is labelled exploratory.

Where restatements differ, [README.md §Cross-cutting invariants](README.md#cross-cutting-invariants)
is normative.

## Construct definitions

- **Revision object:** one provisional mapping plus type, endpoints, provisional relation,
  evidence, governance state, and a finite candidate-state inventory.
- **Assignment:** exactly one selected state for every revision object.
- **Immutable core:** source ontology, target ontology, and trusted integration axioms that the run
  may not revise.
- **Safe:** independently defined by the versioned `SafetyPolicy`; it never means merely “fewer
  violations.”
- **Eligible output:** a result whose status authorizes use of its alignment and whose certificate
  passes final verification for every enabled policy family.
- **Optimal:** maximum recorded integer utility among safe assignments in the frozen finite state
  space, proven by the selected exact backend with zero gap.
- **Semantic preservation:** relation-aware mapping quality and/or preservation of a predeclared
  weighted query basis. Row retention alone is not semantic preservation.
- **Recovery:** useful, non-vacuous semantic content selected beyond the best eligible unguarded or
  deletion-only state. A safe state with an impossible antecedent receives no recovery credit.
- **Generalization:** performance on an ontology pair, domain, or matcher output excluded from all
  fitting, calibration, candidate tuning, and threshold selection.

## Claim hierarchy

Results are interpreted in this order:

1. eligibility and false-safe rate;
2. semantic quality among eligible outputs;
3. candidate-relative optimality or regret;
4. resource efficiency;
5. robustness and reviewability.

A method failing level 1 cannot compensate with a better mean score at later levels. Unknown
outcomes are reported as unknown or fallback, never averaged into safe outcomes.

## Research questions

### RQ1 — safety and implementation fidelity

Does the supported-profile implementation return only assignments satisfying the declared policy
and preserve the proposal's formal invariants?

- **Primary outcome:** false-safe count and rate under final certificate replay and, where
  available, an independent complete verifier.
- **Support required:** XR-E01, including exhaustive finite instances, the logical mechanism
  suite, certificate audit, and fault injection.
- **Bound:** the claim applies only to profiles and policy query families for which the adapter
  reports complete support. It is not a proof about unrestricted OWL or unknown reasoner behavior.

### RQ2 — value of typed semantic states

At identical eligibility, does the full class-state language preserve more correct semantic
content than exact deletion-only repair?

- **Primary outcome:** pair-level change in weighted query preservation, with relation-aware
  mapping F1 as a co-primary outcome fixed at G4.
- **Support required:** XR-E02 on public and mechanism-aware repair benchmarks.
- **Bound:** this establishes value only for the frozen finite candidates; it does not claim that
  the returned repair is an unrestricted semantically optimal ontology repair.

### RQ3 — contribution of individual state primitives

Which directional, guarded, qualification, and endpoint-replacement primitives help which failure
mechanisms?

- **Primary outcome:** semantic-preservation gain of each preregistered action-family contrast,
  stratified by corruption/logical mechanism.
- **Support required:** XR-E02 action-family ablation with multiplicity correction.
- **Bound:** absence of an average gain does not prove a primitive is useless in every domain;
  post-hoc mechanism findings remain exploratory unless replicated.

### RQ4 — joint versus sequential repair

Does optimizing final mapping states jointly reduce final decision regret compared with selecting
a deletion diagnosis first and attempting recovery afterward?

- **Primary outcome:** regret against exhaustive or exact oracle-utility optima on tractable
  components.
- **Support required:** XR-E03 with identical action spaces, external utility, and budgets.
- **Bound:** the claim concerns the declared final utility. A deliberately lexicographic
  diagnosis-first objective is a different research target.

### RQ5 — learned decision utility

Does decision-focused state utility reduce held-out repair regret compared with provisional
confidence, hand-coded utility, and pointwise state prediction?

- **Primary outcome:** pair-level downstream regret under a frozen external evaluation utility.
- **Support required:** XR-E04 with ontology-pair splits and multiple training seeds.
- **Bound:** local classification accuracy is diagnostic. Safety is not attributed to learning.

### RQ6 — guard and replacement candidate coverage

Can bounded retrieval/ranking propose useful guards and replacements with enough recall to justify
their state-space and review cost?

- **Primary outcome:** top-k candidate recall and accepted recovery gain, reported separately.
- **Support required:** XR-E05, including a candidate-oracle ceiling and failure attribution.
- **Bound:** high recall does not establish semantic correctness; accepted candidates still
  require policy verification and semantic evaluation.

### RQ7 — transfer

How does semantic quality transfer across ontology pairs, domains, and matcher score regimes?

- **Primary outcome:** held-out pair/domain/matcher change in repair regret and semantic
  preservation relative to in-domain evaluation.
- **Support required:** XR-E06.
- **Bound:** “safety remains invariant under shift” is supported by RQ1/RQ9 eligibility evidence,
  not by an assumption that shifted models are accurate.

### RQ8 — scalability and exactness frontier

Which instance properties determine exact solvability, and how much utility is retained by a
verified anytime solution at fixed budgets?

- **Primary outcome:** probability/time to zero gap and verified-incumbent regret over time,
  modelled against conflict-component size and state count.
- **Support required:** XR-E07.
- **Bound:** a frontier is conditional on hardware, reasoner/solver versions, policy, candidate
  budgets, and ontology profiles. Raw ontology size alone is not the asserted predictor.

### RQ9 — robustness and fail-closed behavior

How does the system behave with incomplete references, pinned errors, baseline unsatisfiability,
unsupported constructs, and injected reasoner/solver failures?

- **Primary outcome:** false-safe rate and correct status/fallback behavior under each perturbation;
  quality degradation is secondary.
- **Support required:** XR-E08.
- **Bound:** robustness to tested failures does not authorize swallowing unclassified exceptions or
  treating arbitrary external failures as safe.

### RQ10 — explanation reviewability

Do state-level, certificate-derived explanations and counterfactual alternatives improve expert
repair review over raw scores or axiom justifications alone?

- **Primary outcome:** adjudicated decision accuracy and review time.
- **Support required:** XR-E09 with an approved human-participant protocol.
- **Bound:** perceived usefulness is secondary and does not demonstrate logical correctness or
  broad usability outside the sampled experts and tasks.

## Formal hypotheses

| ID | Hypothesis | Confirmatory experiment | Falsifying observation |
|---|---|---|---|
| H1 | Supported exact mode produces no false-safe assignments in the frozen evaluation corpus. | XR-E01 | Any replay-confirmed enabled-policy violation in a result labelled safe. |
| H2 | Full states improve semantic preservation over exact deletion-only repair at equal eligibility. | XR-E02 | No positive preregistered pair-level effect, or any eligibility degradation. |
| H3 | Joint optimization has lower final regret than sequential diagnosis/recovery. | XR-E03 | No positive paired regret effect under equal action space and budget. |
| H4 | Decision-focused utility lowers repair regret versus pointwise learning. | XR-E04 | No positive held-out regret effect under the frozen external utility. |
| H5 | Source-side guarded-forward states help corruptions caused by unconditional source-to-target paths. | XR-E02 | No corrected mechanism-stratum gain over the corresponding ablation. |
| H6 | Candidate recall is the dominant limiter of complex recovery after safety gating. | XR-E05 | Candidate-oracle headroom is small or oracle rejection/utility selection explains more missed recovery. |
| H7 | Conflict-component structure and state count predict exact runtime better than raw ontology size. | XR-E07 | The preregistered model comparison does not favor structural predictors. |
| H8 | Domain shift degrades semantic quality without increasing false-safe outputs. | XR-E06 + XR-E08 | Shifted safe outputs fail verification, or no semantic degradation is observed where the hypothesis predicts it. |

“Not rejected” is not written as “proved.” H1 is an empirical zero-failure claim over a frozen
corpus; the formal safety argument additionally depends on the proof obligations and adapter
capabilities in `01-architecture-and-contracts.md`.

## Theory-to-code proof obligations

The implementation must make these reviewable rather than merely cite the proposal:

1. every mutable object has an off state compiling to no axioms;
2. the immutable core is checked before optimization;
3. state compilation is deterministic and content-addressed;
4. every revision object has an exact one-hot solver constraint;
5. every learned score is outside the safety acceptance path;
6. every no-good excludes a sufficient unsafe combination and is replayable;
7. every safe status follows a complete `SAFE` oracle result for all enabled queries;
8. every unknown/timeout/unsupported result is non-authorizing;
9. exact optimality is claimed only for a frozen finite inventory and zero gap;
10. certificate replay reconstructs the final axioms and repeats the policy check;
11. module-restricted checks obey the lemmas in `03-module-soundness.md`; a module-level safe
    result verifies an incumbent only for policy families with a proven completeness lemma.

XR-WP1 and XR-WP2 implement these obligations. XR-E01 tests them; neither activity alone proves
that a utility model represents domain truth.

## Non-goals and prohibited claims

XR-1 does not support claims that it:

- repairs an unsafe immutable core;
- finds the unique objectively correct repair;
- provides full conservative extension unless that exact decision procedure is later specified;
- searches unbounded OWL expressions;
- supports class, property, and individual mappings through one untyped action language;
- treats reference absence as a confirmed negative;
- makes a safe mapping semantically correct;
- scales exactly to all ontology pairs;
- is the first system to use deletion, weakening, completion, explanations, hitting sets, or
  conservativity-violation repair;
- replaces domain-expert judgement.

Any publication or release note must distinguish established prior techniques from the proposed
integration: typed state selection, lazy reasoner-derived state conflicts, candidate-relative
global utility, learner-independent acceptance, and replayable exact/anytime statuses.

## Promotion rule

Research evidence may change an Exact-OM default only through a separate decision record that
names the frozen experiment artifacts, selected policy, candidate budget, utility checkpoint,
supported profiles, failure behavior, and migration path. Failed or mixed hypotheses remain in the
report. An implementation PR or a favorable exploratory plot is insufficient.
