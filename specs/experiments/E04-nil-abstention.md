# E04 — NIL / Abstention as a First-Class Output (DISO 2026)

**Motivation** (audit obs. 9): DISO-2026 ranking pools contain an explicit NIL option (WP-I
drops it with an `extras` flag today); the full-alignment task already *implicitly* abstains
via the acceptance classifier, but nothing emits or evaluates "no match" as a prediction. Any
track with NIL-aware metrics will penalize a system that must always pick a real candidate.

**Hypothesis**: deriving a NIL score from the existing acceptance machinery
(`p_nil = 1 − p_match`, calibrated) and inserting it into rankings beats both "never NIL" and a
naive threshold rule on DISO's NIL-aware metrics, without hurting non-NIL MRR by more than
0.005.

## Change

1. Ranking task: when the candidate pool declares NIL (WP-I `extras.nil_candidates`), insert a
   synthetic NIL row ranked by `p_nil` — from the accept model when trained, else from the
   heuristic no-match risk (already computed: `_no_match_risk`). Config
   `matching.nil: off|accept_model|heuristic`.
2. Global task: `p_nil` also exported per abstained source in explanations/outputs (typed-tsv
   gains optional NIL rows where the track's format allows).
3. Calibrate `p_nil` on train-split sources whose gold target is absent from their pool (these
   exist in Bio-ML candidate files — verify count first; if scarce, construct by pool ablation
   on the train split only).

Touched: selector output path, ranking writer, small calibration step; WP-I descriptor flag
consumed instead of dropped.

## Arms & validation

off / heuristic / accept_model on DISO (primary; NIL-aware metrics per the track's eval —
via the OAEI-Bio-ML-eval backend when it lands, else the kit's own scorer) + Bio-ML local
(guard: MRR/Hits unchanged within CI). Pool-ablation stress test: remove gold from x% of val
pools, measure NIL detection AUROC. 3 seeds.

**Promotion**: NIL arms ship as the default **for tracks whose descriptor declares NIL**;
non-NIL tracks keep current behavior regardless of outcome.

**Effort**: M. **Risks**: few natural NIL training examples — the pool-ablation construction
must stay train-split-only (harness leakage guard applies); DISO metric definitions may shift
before the campaign — pin the eval version in the results note.
