# Exact-OM Methodology Experiments Plan

**This plan is deliberately separate from the engineering overhaul (`specs/WP-*`).** Everything
here **changes results** and therefore requires empirical validation before it can touch the
product defaults. It runs **after** the overhaul completes, because it depends on: WP-J
(scoring constants exposed as config — audit finding F5), WP-C (honest cumulative timing),
WP-I (revision-pinned datasets), WP-B (the frozen parity baseline), and WP-E (extended eval).

Motivating findings are in `specs/04-methodology-audit.md` §"Methodological observations" —
each experiment cites its finding. The current methodology is *sound*; these are hypotheses for
improving it, not fixes.

## Experiments

| ID | Title | Finding | Cost | Expected value | Priority |
|----|-------|---------|------|----------------|----------|
| E00 | Experiment harness & frozen baselines | — | S | enables everything | **first, mandatory** |
| E01 | Global alignment extraction (mutual-best / assignment) | 1 | S | P↑ on 1-1 tracks, cheap | **1** |
| E03 | Score calibration, threshold transfer & tuning-objective ablation | 8 | S–M | robustness across tasks | **1** |
| E06 | String-similarity ensemble as a second lexical signal | 4 | S–M | R↑ on non-biomedical tracks | **1** |
| E05 | Candidate retrieval upgrades (encoder, fusion, adaptive k) | 3 | M | recall ceiling ↑ | **2** |
| E07 | Listwise LLM arbitration with abstain option | 6 | M | accuracy ↑ AND LLM cost ↓ | **2** |
| E04 | NIL / abstention as a first-class output (DISO 2026) | 9 | M | required for DISO track | **2** (deadline-driven) |
| E02 | Anchor-guided structural rescoring (second pass) | 2 | M–L | biggest headroom on sparse-lexical tracks | **3** |
| E08 | Attribute-channel polarity & evidence double-counting | 5 | M | precision ↑, cleaner explanations | **3** |
| E09 | Hierarchy semantics: IC-weighted ancestor overlap + siblings | 7 | M | structural channel strength ↑ | **3** |
| E10 | Fusion & selector ablations (γ/τ_LLM sweep, GBDT accept, pairwise accept) | 8, 10 | M–L | validates/updates core constants | **3** |

Priorities 1 → 3 = run order; within a priority band experiments are independent and can run on
parallel agents/machines. E00 blocks all.

## Validation protocol (binding for every experiment)

**Datasets & splits** (via `exact data`, revisions pinned in the lockfile, recorded in
`run_stats.json`):
- **Development**: Bio-ML tasks with provided train (30%) / val splits — tune ONLY here.
- **Reporting**: Bio-ML test splits, Anatomy, Conference (all pairs), DISO (E04/E07), mini-BioKG
  where relevant. Test splits are touched once per experiment, at the end.
- Never tune on anything the eval subtracts or scores. The F1/F2 provenance guards
  (audit findings) must be active.

**Runs & statistics**:
- ≥3 seeds per configuration (baseline and variant), same seeds across arms.
- Report per-task P/R/F1 (global) and MRR/Hits@1 (local), plus macro averages.
- Significance: paired bootstrap over per-source decisions (10k resamples) on the primary
  metric; report the CI, not just the point delta.
- Cost: LLM call counts + tokens, and wall-time from the WP-C ledger (`cumulative compute`),
  reported next to quality metrics. A quality win that doubles cost is a finding, not a win.

**Discipline**:
- One change per experiment arm; if two mechanisms interact (e.g. E01×E03), run the 2×2.
- Every experiment lands as: config-flagged code (default = current behavior), a runner config
  under `exp/experiments/EXX/`, and a results note appended to the experiment's spec file
  (tables + decision). No experiment code merges without its flag defaulting off.
- The pre-experiment baseline is the WP-B parity baseline rerun on the pinned data with the
  E00 harness — every delta in every table is against that, not against paper numbers.

## Promotion criteria (experiment → product default)

An arm becomes the new default only if ALL hold:
1. Macro primary-metric improvement with 95% bootstrap CI excluding zero, on ≥3 seeds.
2. No single reporting task regresses by more than 0.5 F1 points (or 0.005 MRR).
3. Cost neutral or justified (wall time within 1.2×; LLM tokens within 1.2× — unless the
   experiment's stated goal is a cost reduction).
4. Explanations remain exact: the importance decomposition must still reconstruct the final
   score algebraically (this is a product invariant, not a metric).
5. The flag flip + updated generated config/docs land as their own PR citing the results note.

## Template for new experiment specs

`EXX-title.md`: Motivation (audit finding) · Hypothesis (falsifiable, with expected direction
and rough magnitude) · Change (implementation sketch, config flags, touched modules) ·
Arms & sweep · Validation (tasks, metrics, seeds, cost) · Promotion decision rule ·
Effort & risks · Results note (appended after running).
