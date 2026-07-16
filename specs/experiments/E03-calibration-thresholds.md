# E03 — Score Calibration, Threshold Transfer & Tuning-Objective Ablation

**Motivation** (audit obs. 8 + findings F4): three intertwined choices are currently
unmeasured: (a) `S_final` is not a calibrated probability, yet a fixed global threshold (0.7)
is shared across tasks — and silently replaced by the median learned acceptance threshold when
the calibrated selector runs; (b) the acceptance-threshold tuner double-counts wrong-winner
reference sources as FP **and** FN — an implicit precision bias; (c) with no training
reference, the heuristic selector's constants (support weight 0.60, no-match 0.55) are folklore.

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

{none, platt, isotonic} × {fp_fn, fp} on Bio-ML (train splits exist); threshold-transfer test:
tune a single global threshold on one task's val split, apply to all others, compare vs
per-task. Unsupervised arm on Conference/Anatomy (no train refs). 3 seeds. Primary: macro F1;
secondary: ECE, per-task threshold variance.

**Promotion**: standard; calibration also promotes if it merely *matches* F1 with materially
lower cross-task threshold variance (robustness win), CI-supported.

**Effort**: S–M. **Risks**: isotonic overfits small train splits — cap bins; Conference refs
are tiny, treat unsupervised arm as exploratory.
