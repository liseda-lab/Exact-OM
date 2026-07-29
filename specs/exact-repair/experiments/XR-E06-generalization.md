# XR-E06 — Cross-pair, cross-domain, and cross-matcher transfer

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP4; XR-WP5; selected model/candidate variant frozen from training/pilot
only<br>
**Question/hypothesis:** RQ7/H8 (semantic-quality portion)<br>
**Primary outcome:** transfer degradation in repair regret and semantic preservation

## Authorized claim

How a frozen Exact-Repair utility/proposal configuration transfers to ontology pairs, domains, and
matcher score distributions excluded from all fitting and tuning. Safety results are reported, but
the broad fail-closed claim remains grounded in XR-E01/XR-E08.

## Transfer regimes

1. **Leave-one-pair-out:** same broad domain, unseen ontology pair.
2. **Leave-one-domain-out:** biomedical, anatomy/conference, security/built-environment or the final
   frozen domain families.
3. **Leave-one-matcher-out:** train/calibrate without outputs from the evaluated matcher.
4. **Combined shift:** unseen domain and matcher where sample availability permits.
5. **Pair adaptation sensitivity:** a separately labelled bounded transductive calibration arm with
   interaction/label budget reported.

The offline primary arm receives no target-pair labels, pair-specific reference queries, or fitted
calibration. Ontology-native structural inference is permitted because it is available at deployment;
the split auditor records it separately from learned information.

## Systems

- frozen hand-coded utility and deterministic candidates;
- pointwise learned utility;
- selected decision-focused utility;
- structural/conflict-feature versus lexical-only ablation;
- frozen proposal model where XR-E05 justified it;
- optional bounded pair calibration/adaptation sensitivity.

All systems share policy, candidate budget, final oracle, and solver. Matcher scores are calibrated
only according to the arm definition.

## Outcomes

Primary:

- within-to-held-out change in external repair regret;
- within-to-held-out change in weighted semantic preservation.

Secondary:

- relation-aware F1, candidate recall, calibration/error, fallback/unknown rate;
- eligible output safety/replay count;
- degradation by ontology profile, size, lexical shift, and matcher score distribution;
- deterministic baseline gap and benefit/cost of bounded adaptation.

## Decision rule

RQ7 is reported as a transfer profile, not a binary universal claim. The structural/conflict model
supports the proposal's transfer hypothesis only if its held-out degradation is smaller than the
lexical-only model by the G4 margin in the preregistered regimes. H8's quality portion expects
measurable degradation under material shift; its safety portion is evaluated jointly with XR-E08.

If safety eligibility changes under model shift, stop and treat it as an RQ1/RQ9 defect rather than
a normal quality tradeoff.

## Leakage audit

- no ontology pair/entity text/embedding cache from held-out groups in training;
- no test matcher calibration or threshold selection;
- no synthetic corruption siblings across groups;
- no expert/test conflict labels reused;
- no test-specific query-basis fitting;
- adaptation interactions and pseudo-labels counted and unavailable to the offline arm.

## Non-claims

- Successful transfer across the frozen families does not establish universal domain robustness.
- Safety invariance is conditional on the complete oracle; model quality cannot guarantee it.
- Transductive adaptation is not reported as zero-label generalization.
- Matcher-output diversity does not replace comparison with external repair systems.
