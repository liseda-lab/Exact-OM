# E02 — Anchor-Guided Structural Rescoring (second pass)

**Motivation** (audit obs. 2): every pair is scored independently; the system never exploits
the alignment's own coherence. Exact matches and high-confidence pairs are perfect anchors:
if s≡t, then neighbors of s should preferentially match neighbors of t (the intuition behind
LogMap's anchor expansion and similarity flooding — both repeatedly effective in OAEI).

**Hypothesis**: a single anchor-agreement rescoring pass improves F1 most where lexical signal
is weak (DISO, OAEI-KG, SNOMED-FMA), ≥1 pt on at least one such track, ≈neutral on
lexically-easy tracks; a second iteration adds little (diminishing returns expected — verify).

## Change

New optional stage between scorer and selector (`matching.anchor_rescoring: off|one_pass|
iterate`), implemented as `impl/models/anchor_rescorer.py`:

1. Anchor set A = exact matches ∪ pairs with S_final ≥ a (config, default 0.95, swept).
2. For each candidate (s,t): `anchor_agreement(s,t)` = weighted fraction of s's neighbors
   (hierarchy + object edges, from the existing per-entity bundles) whose anchored image lands
   in t's neighborhood; symmetrized; weights = edge IC (already computed). Also
   `anchor_conflict`: anchored neighbors landing far from t.
3. Fold in as a sixth channel through the existing σ-mixing (it gets a confidence
   `s_anchor = f(agreement, conflict)` and quality `q_anchor = anchor coverage`), so the
   importance decomposition stays exact — **no ad-hoc score addition**. `iterate` mode: recompute
   anchors from pass-1 output, max 2 passes.

Touched: new module + one registration in the scorer's channel list (flag-gated); selector
features gain `s_anchor` when active.

## Arms & validation

off (baseline) / one_pass / iterate; anchor threshold a ∈ {0.9, 0.95, exact-only}. Tasks: full
matrix with emphasis on DISO + SNOMED pairs; both global and local metrics (channel helps
ranking too). 3 seeds. Report channel-importance distributions (how often anchor channel
dominates) for the explanation story.

**Promotion**: standard criteria; also require explanation invariant (decomposition exact) and
runtime within 1.2× (neighbor joins are index lookups over existing bundles — should hold).

**Effort**: M–L. **Risks**: error propagation from wrong anchors — mitigate with exact-only
anchor arm; hub entities inflating agreement — IC weighting should damp, verify on OAEI-KG.
