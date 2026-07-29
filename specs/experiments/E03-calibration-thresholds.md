# E03 — Score Calibration, Threshold Transfer & Tuning-Objective Ablation

**Motivation** (audit obs. 8 + findings F4): three intertwined choices are currently
unmeasured: (a) `S_final` is not a calibrated probability, yet a fixed global threshold (0.7)
is shared across tasks — and silently replaced by the median learned acceptance threshold when
the calibrated selector runs; (b) the acceptance-threshold tuner double-counts wrong-winner
reference sources as FP **and** FN — an implicit precision bias; (c) with no training
reference, the heuristic selector's constants (support weight 0.60, no-match 0.55) are folklore.

## Research questions

- **RQ03.1**: Does Platt or isotonic calibration improve probability calibration and make a
  shared threshold more stable across tasks without reducing F1?
- **RQ03.2**: How much F1 is lost when a threshold selected on one task transfers to another,
  compared with a target-pair threshold, across the full donor×recipient matrix?
- **RQ03.3**: What precision/recall effect is caused by counting a wrong winner as FP+FN versus
  FP only?
- **RQ03.4**: Which single score-distribution threshold is the strongest no-target-label
  primitive to carry into the complete label-free selectors in E15?

**Hypothesis**: per-task calibration (Platt/isotonic on the train split) makes one universal
threshold transfer across tasks within 0.3 F1 pts of per-task tuned thresholds; removing the
FP+FN double-count shifts the P/R balance measurably (direction: recall up) — whether F1
improves is the open question.

## Change

1. `matching.calibration: none|platt|isotonic` — fit on train-split OOF decisions (never
   full reference; harness guards this), applied to `S_final` before threshold/extraction.
2. `selector.tuning.count_reference_miss_as: fp_fn|fp` — expose the double-count rule.
3. Unsupervised threshold arm: replace fixed 0.7 with Otsu/knee-point selection over the
   per-task score distribution (for the no-training-reference regime).
4. Report reliability diagrams + ECE per task (extend `exact/analysis/` with a calibration
   plot) — a deliverable regardless of promotion.

Touched: selector tuning path, small calibration module, analysis plot.

## Arms & validation

{none, platt, isotonic} × {fp_fn, fp} on Bio-ML (train splits exist). Compute threshold transfer
from saved score frames—no additional matcher runs:

1. **Deployable matrix**: every task with a legitimate validation reference is a donor; tune on
   that donor validation split and apply unchanged to every recipient. No-label tasks are
   recipients only. Reporting recipients are evaluated together in the one frozen final pass.
2. **Symmetric oracle diagnostic**: after all predictions are frozen, let every task act as a
   donor using its labelled reference and compute the full task×task arithmetic matrix. Mark
   every such threshold `oracle_diagnostic`; it cannot select a method or promote, but exposes
   compatibility for E15/E16.

For each cell report donor threshold, recipient F1/regret to its own oracle threshold,
prevalence/score-quantile shift, and whether the loss exceeds E00's recipient MDE. Summarize the
median, p90, worst-case, and within-/cross-domain transfer loss rather than basing RQ03.2 on one
arbitrary donor. Unsupervised arms remain on Conference/Anatomy. Three seeds. Primary: macro F1;
secondary: ECE and threshold variance.

The unsupervised arm is a threshold-component experiment only; E15 owns the claim that a full
acceptance strategy works without target-pair training labels. Platt/isotonic fitting is the
`supervised` resolution of the calibration component and the distribution rule is its
`label_free` resolution; E22 fits how many usable labelled source groups the former needs before it is worth
selecting.

**Promotion**: standard; the declared robustness arm may promote on lower cross-task threshold
variance when its 95% CI excludes zero and macro F1 is non-inferior (quality-delta CI lower
bound above −0.5 points). This pre-registers threshold variance as that arm's primary endpoint
rather than treating a non-significant F1 difference as evidence of equality.

**Effort**: S–M. **Risks**: isotonic overfits small train splits — cap bins; Conference refs
are tiny, treat unsupervised arm as exploratory.
