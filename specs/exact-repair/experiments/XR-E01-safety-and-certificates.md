# XR-E01 — Safety, optimality conformance, and certificate replay

**Status:** planned<br>
**Depends on:** XR-E00 frozen harness; XR-WP2<br>
**Question/hypothesis:** RQ1/H1<br>
**Primary outcome:** false-safe count and rate

## Authorized claim

Whether, on the frozen corpus and declared complete profile/policy paths, Exact-Repair labels no
assignment safe that violates an enabled policy under final replay/independent verification.

This study may also establish candidate-relative optimizer conformance on enumerable instances. It
does not establish semantic correctness, richer-state benefit, or unrestricted OWL safety.

## Systems

- exhaustive assignment enumeration plus the reference oracle on tiny cases;
- XR exact engine with deterministic hand-coded utility;
- XR exact engine under utility perturbations/random utilities;
- configured independent final verifier where semantic/profile overlap is complete;
- rebuild-mode same-adapter replay, always reported separately from independent verification.

## Corpus strata

1. every hand-authored logical mechanism fixture within declared complete profiles;
2. small mechanism-aware corruptions with fully enumerable state spaces;
3. medium public/captured matcher outputs using base states;
4. unsafe-core, baseline-unsatisfiable, pinned/trusted, multiple-explanation, and multiple-optimum
   cases;
5. unsupported-profile cases used only to assess non-authorizing status;
6. fault-injected oracle/solver/explanation/certificate failures.

## Procedure

For enumerable cases:

1. freeze core, policy, state inventory, governance, and integer utility;
2. classify every assignment;
3. compute all safe optima;
4. run lazy exact repair;
5. replay every conflict against all safe assignments;
6. compare returned objective/assignment set/status with enumeration;
7. reconstruct and reverify the certificate without solver/model.

For all cases, repeat final verification, audit state/axiom provenance, and inject failures at each
orchestrator boundary. Repeat with adversarial learned/random utility orderings to demonstrate that
acceptance does not depend on score quality.

## Outcomes

Primary:

- count/rate of results labelled `OPTIMAL_SAFE`, `SAFE_WITH_GAP`, or `SAFE_FALLBACK` whose final
  alignment violates any enabled query.

Required secondary:

- exact-objective mismatch versus enumeration;
- invalid no-good count (a cut excluding any enumerated safe assignment);
- final replay mismatch;
- incorrect status under unsafe core, timeout, unsupported feature, missing explanation, or solver
  timeout;
- provenance reconstruction failures;
- repeated rejected assignments and termination failures;
- verifier disagreements, stratified by whether both paths declare completeness.

## Decision rule

- H1 is ineligible/safety gate fails after any confirmed false-safe output.
- Exact conformance fails after any objective mismatch or invalid no-good on an enumerable case.
- Adapter disagreement blocks the affected profile from supported claims until resolved; majority
  vote is not used.
- Unknown/unsupported results are correct only if they do not authorize the candidate and the
  status/certificate identifies the reason.

Report a one-sided zero-failure interval for the observed corpus, but do not convert it into a proof
of universal absence. The formal claim remains conditional on reviewed protocol capabilities.

## Artifacts

- assignment truth tables for enumerable cases;
- conflict sufficiency/replay reports;
- full certificates and independent replay reports;
- fault schedule and observed status matrix;
- per-profile/query-family eligibility flow;
- discrepancies and root-cause resolution records.

## Non-claims

- Passing says nothing about whether utility matches expert truth.
- It does not authorize incomplete reasoner paths.
- Zero failures on the frozen corpus is not unrestricted OWL 2 DL verification.
- It does not show that Exact-Repair preserves more content than deletion.
