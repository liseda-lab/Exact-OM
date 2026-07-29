# 04 — Methodology Audit (sanity check of the model & system methodology)

Audit date: 2026-07-16, baseline `9e72ecf`. Sources: full code extraction of the scoring math
(`pair_adaptive_scorer.py`, `semantic_scorer.py`), the selection/calibration pipeline
(`candidate_set_selector.py`, `trainer.py`, `evaluator.py`), and the ESWC-2026 paper's Methods
section (`Paper/main.tex` §3). This file records **findings and non-result-changing fixes**
(assigned to existing WPs). Result-changing improvement proposals live in the separate
**`specs/experiments/`** plan and must be empirically validated there — nothing from that plan
may be folded into the engineering overhaul.

## Verdict

The methodology is sound and internally consistent with the paper:

- **Fusion is what the paper claims**: σ-weighted quality×deviation mixing
  (`σ_k = q_k·|s_k−τ|^γ`), convex composition, exact importance decomposition
  (`I_lex + ΣI_k + I_LLM = 1` reproduces `S_pair` algebraically) — the interpretability story
  checks out in code (`pair_adaptive_scorer.py:1749-1916`).
- **No test-label leakage in the default path.** The full/test reference is merged into the
  candidate frame as a `Label` column (`dataset.py:1060-1065`), but scoring, gating, selection,
  thresholding, and cardinality never read it; it feeds only `ground_truth` reporting and
  evaluation. Verified exhaustively.
- **Selector calibration is done properly**: source-disjoint K-fold OOF validation, seeded
  splits, hard negatives from wrong-winner reference sources, threshold tuned on pooled OOF
  decisions, final refit reported separately (`candidate_set_selector.py:1194-1514`).
- **Evaluation follows the OAEI Bio-ML protocol**: train refs subtracted from both preds and
  refs; `use_in_alignment=false` filtering; standard Hits@k/MRR.

## Findings — fixes that do NOT change results (fold into the main suite)

| # | Finding | Fix | Owner |
|---|---------|-----|-------|
| F1 | **Latent train-on-test path**: with `use_llm_calibration: True` (default False), LLM-decision calibration coefficients are fit on `(p_llm, gold)` pairs whose gold comes from the merged full/test reference (`semantic_scorer.py:1225-1284`, sample collection at `pair_adaptive_scorer.py:1887`). Enabling the flag = calibrating on test labels. | Restrict calibration samples to training-reference pairs (or an explicit validation slice); hard-error if only full-reference labels are available. Regression test. | WP-A (A4 bug list) |
| F2 | **No provenance guard on reference files**: nothing records/asserts which `-r`/`-f` files fed a run; eval's train-subtraction silently depends on the caller passing the same `-r` to eval. | Record sha256 + row counts of every reference/candidate input in `run_stats.json`; eval warns when the train file it subtracts differs from the one the selector calibrated on. | WP-E |
| F3 | **Stale documented constants**: `docs/semantic_scorer_defaults.md` says `gamma: 0.73`, `tau_LLM: 0.6`, `batch_size: 128`; shipped config is `2.0 / 0.5 / 512` (paper agrees with config). Class defaults differ again (`tau_LLM=0.35`). | Killed structurally by WP-J single-source-of-defaults + WP-H generated reference; until then, fix the doc numbers. | WP-H / WP-J |
| F4 | **Undocumented threshold override**: when the calibrated selector runs, the configured `alignment_params.threshold` is silently replaced by the median per-source `selection_accept_threshold` (`semantic_runner.py:2845-2857`). Correct behavior, surprising UX. | Document prominently; log a one-line notice at run time stating the effective threshold and its origin. | WP-H (doc) + WP-C (log line) |
| F5 | **Buried scoring/selection constants** prevent systematic study: channel blends fixed 0.5/0.5, per-relation caps 2/3, stability factor 2, lexical retrieval fusion `max(tok, 0.85·gram, 0.65·tok+0.35·gram)`, DF ceiling `max(10, 0.2N)`, attribute property weights 1.0/0.8/0.6/0.5, `U` constants, `s_attr` floor, alias caps/blocklists (full inventory in the two audit reports, kept with this file's git history). | Expose as config with current values as defaults (bit-identical by construction). This is a **prerequisite for `specs/experiments/`** sweeps. | WP-J (add task) |
| F6 | **Tie-breaking nondeterminism**: `heapq.nlargest` order among equal scores is implementation-defined (`entity.py:89-136`); AMP/cuBLAS reductions add run-to-run jitter on GPU. | Deterministic tie-break (secondary sort key = target IRI) — verify no metric change on fixtures + one real task before merging; document residual GPU nondeterminism; multi-seed protocol lives in experiments E00. | WP-D (with check) |
| F7 | **Local eval K default inconsistency**: `EvaluationData.K` defaults `[1]` while config `k: [1,5,10]`; harmless when config flows through, confusing otherwise. | Align defaults. | WP-A |

## Methodological observations that motivate the experiments plan

(Recorded here as findings; proposals + validation protocols in `specs/experiments/`.)

1. **Extraction is greedy per-source, then per-target** — never mutual-best or a global
   assignment; the selector zeroes all non-winner rows before cardinality runs. → E01.
2. **Every pair is scored independently** — no cross-pair structural consistency; confident
   matches (exacts, high scores) never reinforce neighboring candidates. → E02.
3. **Encoder mismatch between retrieval and scoring**: candidates are retrieved with
   `all-MiniLM-L6-v2` but scored with SapBERT; channels are fused by unweighted `max`. Retrieval
   is the recall ceiling for everything downstream. → E05.
4. **Lexical evidence is embedding-only**; no character/string similarity anywhere in scoring
   (only in retrieval). Trivial string evidence (shared rare tokens, near-identical spellings)
   is delegated entirely to SapBERT geometry. → E06.
5. **The attribute channel is support-only** (`s_attr = max(τ, r_attr)`) — attributes can push a
   score up but never down, and they are matched against a bank containing the *other side's*
   labels/hierarchy/similarity sentences, which risks double-counting evidence already used by
   other channels. → E08.
6. **LLM arbitration is per-pair binary** with a fixed `p ≥ 0.5` label cut and β·U mixing;
   candidates of the same source are never compared jointly by the LLM. → E07.
7. **Hierarchy evidence = depth-2 ancestors with 1/(d+1) specificity**; no ancestor-set overlap
   statistics (IC-weighted Jaccard), no sibling evidence. → E09.
8. **Threshold-tuning double-counts** a wrong-winner reference source as both FP and FN
   (`candidate_set_selector.py:2278-2293`) — an implicit precision bias that is a modeling
   choice, not a bug; its effect is unmeasured. → E03.
9. **No NIL/abstention concept** beyond selector abstention — DISO-2026 pools contain an
   explicit NIL option the system cannot emit. → E04.
10. **Accept model trains on one sample per source** (its winner only); runner-up information
    and pairwise structure are discarded. → E10.
11. **Property matching is implemented but not empirically validated as its own task**;
    class-dominated aggregate metrics could hide poor domain/range, sub-property, and usage
    evidence. → E11.
12. **Instance matching is implemented but not empirically validated at ABox scale**; ambiguous
    labels, types, literal values, neighborhoods, hubs, and reference incompleteness create a
    different regime from TBox class matching. → E12.
13. **OWL/RDF, CSV, and CSV+Datalog pass source conformance tests but lack representation-level
    parity and information ablations**; parser fidelity and gains from additional materialized
    facts are currently indistinguishable. → E13.
14. **Typed relation output relies on a transparent hierarchy heuristic**; equivalence versus
    either subsumption direction has not been evaluated separately from entity-pair detection,
    and semantic entailment has not been compared with learning. → E14.
15. **The calibrated global accept classifier requires target-pair training mappings**; the
    current heuristic fallback is not a validated, complete target-label-free method. → E15.
16. **Cross-pair transfer is unmeasured**; it is unknown whether selector/calibration/relation
    heads trained on one ontology pair can replace unavailable labels on another. → E16.
17. **Independent experiment wins do not establish that the promoted stack is beneficial**;
    retrieval changes candidate pools, channels interact through σ-mixing/selector features,
    and extraction interacts with calibration, NIL, and label-free acceptance. → E17.
18. **Supervised ranking exists but has not been experimentally isolated**: selector calibration
    already groups candidate rows by source and fits a linear listwise-softmax ranker over
    `RANK_FEATURE_NAMES` (`selector/calibration.py:_rank_training_groups,_fit_rank_model`) before
    fitting the top-1 acceptance head. Its objective/model family, incomplete-reference behavior,
    NIL handling, and explanation constraint have not been ablated against pointwise, pairwise,
    constrained, or more expressive alternatives. Training labels also optionally calibrate LLM
    probabilities, so “labels reach only acceptance” is not an accurate system description. → E18.
19. **The σ-mixing form is asserted, not fitted**: `σ_c = q_c·|s_c − τ|^γ`
    (`pair_adaptive_scorer.py:572`) applies one exponent to every channel, domain, and entity
    kind. Observation 8 questions the constants; the functional form itself is equally
    unmeasured, and it also fixes `U` and therefore the LLM gating rate. → E19.
20. **Retrieval is entirely zero-shot** even where training mappings and in-pool hard negatives
    are available; since retrieval is the recall ceiling (obs. 3), it is the one stage whose
    lost recall cannot be recovered downstream. → E20.
21. **LLM use is zero-shot and gated by a proxy**: prompts show no labelled exemplars, and
    `U ≥ τ_LLM` (`pair_adaptive_scorer.py:680`) gates on uncertainty rather than on whether a
    call will change the decision — so confirmatory calls are paid for and buy nothing. → E21.
22. **Supervision is an all-or-nothing switch**: a training reference either resolves or it does
    not (`core/actions/alignment.py:323`), and the sole quantity consulted anywhere is
    `min_positive_sources: 50` (`selector.py:64`). How many labels each component needs, where a
    fixed budget is best spent, and when a label-free method is simply good enough are all
    unknown. → E22.
23. **The evidence model is TBox-shaped and applied uniformly**: the hierarchy channel reads
    depth-2 ancestors with `1/(d+1)` specificity (obs. 7), so on a source with one shallow type
    level or no class hierarchy it degenerates toward its `τ` default and the matcher falls back
    to labels — precisely where labels are weakest. Those inputs are relation-dense rather than
    structurally poor, and the system has no supervised structural method for them: E02's anchor
    propagation is its only structural alternative and is unsupervised. → E23.

## Relationship between the two plans

- The engineering suite (WP-A…WP-L) is **behavior-preserving** and lands first.
- `specs/experiments/` (E00–E23) runs **after** the overhaul (it needs WP-J's exposed
  constants, WP-C's honest timing, WP-I's pinned datasets, and the WP-B parity baseline as its
  frozen reference).
- Promotion rule: an experiment graduates into the product default only per the criteria in
  `specs/experiments/README.md` — never by folding into an engineering WP.
