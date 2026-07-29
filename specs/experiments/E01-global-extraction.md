# E01 — Global Alignment Extraction (mutual-best / assignment)

**Motivation** (audit obs. 1): extraction is greedy per-source top-1 (threshold → per-source
`nlargest` → exact re-injection → per-target `nlargest`), with the selector zeroing non-winner
rows first. Two sources claiming the same target are resolved by score order alone; the loser
gets nothing even when its runner-up was fine. Classic OM systems gain precision from global
1-1 extraction.

## Research questions

- **RQ01.1**: Does mutual-best, stable marriage, or optimal assignment improve global F1 over
  greedy extraction when the reference is predominantly one-to-one?
- **RQ01.2**: Are gains precision-led, and how often does global extraction recover a valid
  runner-up after resolving a target collision?
- **RQ01.3**: How does extraction interact with the selector, exact-match protection, and
  tracks whose true cardinality is not one-to-one?
- **RQ01.4**: Which observable cardinality statistic is safe for choosing an extraction mode
  without consulting test references?

**Hypothesis**: on near-1-1 tracks (Anatomy, Bio-ML equiv), mutual-best or optimal assignment
raises precision ≥0.5 pts at ≤0.3 pts recall cost vs. the greedy cascade; assignment ≥
mutual-best.

## Change

New post-selector extraction strategies behind `matching.extraction` (default `greedy` =
current), operating on the pre-zeroed score frame (selector emits per-candidate scores instead
of zeroing when a global strategy is active — flag-gated in the selector):

- `mutual_best`: keep (s,t) iff t = argmax over T_s and s = argmax over sources proposing t.
- `assignment`: Jonker-Volgenant (scipy `linear_sum_assignment`) on the sparse candidate score
  matrix per connected component, threshold applied after assignment.
- `stable_marriage`: Gale-Shapley on mutual candidate lists (cheap ordering-robust middle
  ground).
Exact matches stay protected (pre-assigned before optimization). Local/ranking task unaffected.

Touched: `impl/models/selector/acceptance.py` (score emission mode),
`core/entities/mappings/entity.py` extraction functions, new `impl/extraction.py`.

## Arms & validation

Arms: greedy (baseline) / mutual_best / assignment / stable_marriage; ×{selector on, selector
off} on one task to check interaction. Tasks: full reporting matrix (global metric only).
3 seeds. Primary metric: macro F1; secondary: per-task P/R, count of sources changed.

For RQ01.4, compute only reference-free task statistics: source/target signature-size ratio,
candidate-graph component density, target-collision rate among source top-1s, reciprocal-top-1
rate, and accepted-score margin distribution. Choose any extraction-mode rule on development
tasks, validate it leave-one-development-task-out, then freeze and apply it to every reporting
task. Reporting references assess the frozen rule only; reference cardinality is examined post
hoc and cannot select the mode.

On Anatomy, Conference, and KG tasks whose inventory does not declare complete gold, invoke
the shared blinded disagreement-adjudication protocol. Report adjusted precision beside the
official raw precision so a precision-led extraction arm is not rejected solely for proposing
valid novel mappings absent from the reference.

**Promotion**: standard criteria; expect `assignment` default for equivalence tracks if it
holds; keep `greedy` for n-m tracks (cardinality >1) — decision may be per-cardinality-config.

**Effort**: S. **Risks**: assignment on huge components (SNOMED) — cap component size, fall
back to mutual-best above it; measure runtime via ledger.
