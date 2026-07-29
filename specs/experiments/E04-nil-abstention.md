# E04 — NIL / Abstention as a First-Class Output (DISO 2026)

**Motivation** (audit obs. 9): DISO-2026 ranking pools contain an explicit NIL option (WP-I
drops it with an `extras` flag today); the full-alignment task already *implicitly* abstains
via the acceptance classifier, but nothing emits or evaluates "no match" as a prediction. Any
track with NIL-aware metrics will penalize a system that must always pick a real candidate.

## Research questions

- **RQ04.1**: Does explicit NIL scoring improve NIL-aware ranking and global metrics over never
  predicting NIL and over a fixed threshold?
- **RQ04.2**: Is `p_nil` from a supervised accept model better calibrated than the heuristic or
  target-label-free E15 selector, especially under changing NIL prevalence?
- **RQ04.3**: Can NIL recall improve without materially harming ranking among sources that do
  have a real match?
- **RQ04.4**: Are synthetic pool-ablation NIL examples representative of natural NIL cases?

**Hypothesis**: deriving calibrated real-candidate and NIL logits from the existing acceptance
machinery and ranking their joint probabilities beats both "never NIL" and a naive threshold
rule on DISO's NIL-aware metrics, without hurting non-NIL MRR by more than 0.005.

## Change

1. Ranking task: when the pool declares NIL (WP-I `extras.nil_candidates`), compute a
   per-candidate `p_match(s,t)` for **every** real candidate from the accept head after any LLM
   fusion/calibration; heuristic mode uses its bounded support score. Compute source-level
   `p_nil(s)` from the trained no-match head (with `1−max_t p_match` as the naive baseline) or
   `_no_match_risk` in heuristic mode. Never compare `p_nil` with raw `S_final`.
2. Put real candidates and NIL on one ranking scale:
   `q = softmax({logit(p_match(s,t))}_t ∪ {logit(p_nil(s))})`, then rank real rows and the
   synthetic NIL row by `q`. E07's `P(A..E,Z)` enters through its declared `p_llm` feature and
   the refitted accept/calibration path, so overflow and non-listwise candidates receive the
   same final treatment. Export both pre-normalization probabilities and `q` for audit. Config
   `matching.nil: off|accept_model|heuristic` and
   `matching.nil_ranking_scale: joint_accept_probability`.
3. Global task: `p_nil` also exported per abstained source in explanations/outputs (typed-tsv
   gains optional NIL rows where the track's format allows).
4. Calibrate candidate/NIL logits on train-split sources whose gold target is absent from their
   pool versus those whose gold is present; calibration is fit on the mutually exclusive
   real-target/NIL outcome, not on raw ranking scores. Such NIL sources
   exist in Bio-ML candidate files — verify count first; if scarce, construct by pool ablation
   on the train split only).

Touched: selector output path, ranking writer, small calibration step; WP-I descriptor flag
consumed instead of dropped.

## Arms & validation

off / heuristic / accept_model on DISO (primary; NIL-aware metrics per the track's eval —
via the OAEI-Bio-ML-eval backend when it lands, else the kit's own scorer) + Bio-ML local
(guard: MRR/Hits unchanged within CI). Pool-ablation stress test: remove gold from x% of val
pools, measure NIL detection AUROC. 3 seeds.

Report categorical NLL/Brier/ECE for the joint distribution, NIL ECE, and rank changes caused
by replacing raw `S_final` order with calibrated `p_match`/`q`. Non-NIL MRR guards whether the
common scale damages real-candidate ordering.

Scheduling dependency: the deadline-critical `off`/`heuristic`/binary-`accept_model` arms have
no E07 dependency. The additional `accept_model` arm that consumes listwise `p_none` runs after
or jointly with E07 and answers RQ07.4; if it misses the DISO schedule, record that RQ as
`inconclusive` and run the joint arm as a frozen follow-up rather than imputing its result.

**Promotion**: subject to the standard criteria, the winning NIL arm ships as the default
**for tracks whose descriptor declares NIL**; non-NIL tracks keep current behavior regardless
of outcome.

**Effort**: M. **Risks**: few natural NIL training examples — the pool-ablation construction
must stay train-split-only (harness leakage guard applies); DISO metric definitions may shift
before the campaign — pin the eval version in the results note.
