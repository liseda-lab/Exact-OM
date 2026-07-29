# XR module-restricted reasoning: soundness contract

**Status:** proposed; freeze at G0 together with `00`–`02`<br>
**Applies to:** every module-restricted policy check in
`01-architecture-and-contracts.md` §Counterexample-guided orchestration, XR-WP2, and XR-WP5

Module extraction is a performance device, never a semantics change. This document fixes, before
any implementation exists, exactly what a module-restricted check may and may not conclude. The
distinctions are load-bearing: one lemma below is an impossibility result, and ignoring it would
produce precisely the false-safe outputs XR-E01 is designed to catch.

## Definitions

- **Extraction signature Σ:** the union of (a) every symbol occurring in any state of the entire
  frozen candidate inventory — endpoints, replacement endpoints, guard and qualification
  expressions, for every revision object, not only selected states; (b) every symbol of the
  trusted integration axioms; (c) every symbol of every configured forbidden-entailment query.
  Σ is computed after inventory freeze and hashed into the certificate.
- **Module M:** a syntactic locality-based module of the immutable core `O` for Σ, using ⊥ or
  ⊤⊥\* locality. These module notions are self-contained and depleting and therefore robust under
  replacement. Any other module notion requires its own reviewed proof before use.
- **S:** the generated axioms of a candidate assignment. By construction of Σ,
  `sig(S) ∩ sig(O) ⊆ Σ`.

When locality of a construct (datatypes, nominals, ABox assertion forms) is not certain, the
axiom is treated as non-local and included. Over-inclusion costs performance, never correctness.

## Lemmas

All positive lemmas follow from robustness under replacement of self-contained depleting
locality-based modules (Cuenca Grau, Horrocks, Kazakov, Sattler, JAIR 2008; Sattler, Schneider,
Zakharyaschev, IJCAI 2009): for any `O'` with `sig(O') ∩ sig(O) ⊆ Σ` and any axiom `α` with
`sig(α) ⊆ Σ ∪ sig(O')`,

```text
O ∪ O' ⊨ α   iff   M ∪ O' ⊨ α
```

- **L1 — violation soundness.** Any violation of any enabled monotone policy family found on
  `M ∪ S` holds on `O ∪ S`, by monotonicity (`M ⊆ O`). Every module-derived conflict and no-good
  is therefore globally valid.
- **L2 — consistency completeness.** `O ∪ S` is inconsistent iff `M ∪ S` is inconsistent
  (`α = ⊤ ⊑ ⊥` has empty signature). A module consistency check may verify incumbents.
- **L3 — Σ-class coherence completeness.** For any named class `A ∈ Σ ∪ sig(S)`:
  `O ∪ S ⊨ A ⊑ ⊥` iff `M ∪ S ⊨ A ⊑ ⊥`. Unsatisfiability of inventory-signature classes is
  decided exactly on the module.
- **L4 — full-signature coherence incompleteness.** For monitored classes outside `Σ ∪ sig(S)`
  the module check is **not** complete. Witness (mandatory fixture):

  ```text
  O = { C ⊑ ∃r.A,   ∃r.B ⊑ ⊥ }     Σ = {A, B}     S = { A ⊑ B }
  ```

  Both core axioms are ⊥-local w.r.t. `{A, B}`, so `M = ∅`; `M ∪ S` is coherent, yet `C` is
  unsatisfiable in `O ∪ S` while `A` and `B` remain satisfiable. This holds already in EL⊥, so no
  profile restriction inside XR-1 escapes it. A module-level `SAFE` therefore never verifies an
  incumbent under the default full-signature relative-coherence policy.
- **L5 — forbidden-entailment completeness.** Σ includes every configured query signature, so
  each forbidden-entailment query is decided exactly on the module.
- **L6 — guard non-vacuity completeness.** Guard antecedent signatures are in Σ by construction,
  so antecedent-satisfiability checks are decided exactly on the module.
- **L7 — justification containment.** Every justification over `O ∪ S` of an entailment with
  signature in `Σ ∪ sig(S)` is contained in `M ∪ S`. For a violation witnessed only by
  full-integration classification (an out-of-Σ class, per L4), extract a fresh module for
  `Σ ∪ {that class}` before explanation; that extraction is sound by the same robustness
  property.

## Normative consequences

| Policy family | Module check may find violations? | Module `SAFE` verifies incumbent? |
|---|---|---|
| consistency | yes (L1) | yes (L2) |
| relative coherence, full monitored signature (default) | yes (L1) | **no** (L4); full-integration classification required |
| relative coherence, explicitly module-restricted signature | yes (L1) | yes (L3) — but this is a different named policy per `01` §SafetyPolicyV1, never reported as full coherence |
| finite forbidden entailments | yes (L1) | yes (L5) |
| guard non-vacuity | yes (L1) | yes (L6) |

- Under the default policy, **every** verified incumbent — not only the final one — requires the
  full-integration coherence classification. Modules accelerate the `UNSAFE` path (violation
  finding, explanation, shrinking), not coherence incumbent verification.
- Module extraction runs once per (core fingerprint, Σ, module type, extractor version) and is
  content-addressed. Any inventory change — including XR-WP5 adaptive expansion — changes Σ and
  invalidates the module.
- Certificates record module type, extractor version, Σ hash, module hash, and which checks ran
  module-restricted. Replay reproduces module-restricted results with the recorded module.
- ABox-heavy cores can defeat locality and yield near-total modules; that is a measured
  performance outcome for the XR-WP2 audit, not a correctness concern.

## Required tests

1. the L4 witness fixture: the engine must not report a coherence-verified incumbent from a
   module-level `SAFE`, and the full check on the same fixture must find `C` unsatisfiable;
2. property test: module and full-integration checks agree on consistency, Σ-class
   satisfiability, forbidden entailments, and guard non-vacuity for generated small instances;
3. differential test: every module-derived conflict replays on the full integration;
4. invalidation test: changing any inventory state changes the Σ and module hashes and forces
   re-extraction;
5. uncertain-locality fixture: datatype/nominal/ABox axioms of unproven locality are included in
   the module.

Weakening any lemma's use, adding a new module notion, or restricting the monitored signature is
a policy or suite revision, not an implementation choice.
