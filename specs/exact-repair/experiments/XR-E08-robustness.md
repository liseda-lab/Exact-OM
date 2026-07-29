# XR-E08 — Imperfect references, governance, and fail-closed stress tests

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP2 and XR-WP5<br>
**Question/hypothesis:** RQ9/H8 (safety portion)<br>
**Primary outcomes:** false-safe count and status/fallback correctness under perturbation

## Authorized claim

How Exact-Repair's eligibility, semantic quality, and diagnostics respond to frozen perturbations in
references, trusted mappings, utilities, imports, profiles, and runtime dependencies. The primary
claim is fail-closed behavior, not invariance of quality.

## Perturbation families

### Supervision/reference

- mask increasing fractions of reference positives by group;
- convert missing reference entries to unknown versus naive negative (sensitivity only);
- inject bounded confirmed-label noise in training, never the safety oracle;
- reduce expert preference data and shift corruption mixtures.

### Governance/core

- pre-existing named unsatisfiable classes under relative policy;
- truly unsafe immutable core;
- correct and incorrect pinned/trusted mappings;
- coverage constraints conflicting with all-off;
- finite forbidden-entailment baseline changes.

### Candidate/utility

- perturb/calibrate/invert utility scores;
- high-confidence correct mapping versus several low-confidence incorrect mappings;
- candidate budget/guard depth increases;
- multiple equal optima and distant endpoint conflicts.

### Oracle/solver/system

- timeouts at each call/loop stage;
- unsupported axioms/profile mismatch;
- missing/invalid explanations;
- worker crash, stale/corrupt cache, solver infeasible/error/invalid bound;
- ontology import change/hash mismatch;
- certificate write interruption and replay mutation;
- alternative complete reasoner on overlap cases.

Each injection has an expected status/behavior oracle fixed before execution.

## Procedure

- Start from clean frozen cases with known accepted behavior.
- Apply one perturbation family at controlled severities, then selected interactions.
- Run deterministic and learned utility arms where relevant.
- Replay every safe result; audit that expected diagnostic/fallback events appear.
- Separate safety-path failures from semantic-quality degradation.
- For references/training noise, retrain only according to the frozen protocol and held-out splits.

## Outcomes

Primary:

- false-safe outputs;
- exact match between observed and expected status/authorizing-alignment behavior.

Secondary:

- fallback/unknown/unsafe-core rates;
- semantic regret/preservation degradation;
- over-deletion under reference masking and naive-negative sensitivity;
- pinned-state conflict diagnostic completeness;
- cache/certificate mutation detection;
- recovery time/cost after retriable failures.

## Decision rule

Any confirmed false-safe output fails the robustness and G2/G5a/G5b safety gates. A status mismatch or
missing mandatory diagnostic fails the affected failure contract. H8's safety portion is supported
only if model/reference/score shift changes quality without producing a false-safe output across the
frozen supported paths; this remains an empirical corpus claim.

Expected `ORACLE_UNKNOWN`, `UNSAFE_CORE`, or `SAFE_FALLBACK` results are correct fail-closed
behavior, not algorithm failures to be excluded. Operational go/no-go thresholds may still reject a
system with excessive fallbacks even if logically safe.

## Non-claims

- Injected failure coverage is not proof against all implementation bugs.
- Relative policy intentionally permits recorded baseline-unsatisfiable classes; this is not full
  coherence.
- Robustness to incomplete references concerns learning/evaluation quality, not oracle safety.
- Alternative reasoner disagreement is investigated, not resolved by voting.
