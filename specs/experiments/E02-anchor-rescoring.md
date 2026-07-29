# E02 — Anchor-Guided Structural Rescoring (second pass)

**Motivation** (audit obs. 2): every pair is scored independently; the system never exploits
the alignment's own coherence. Exact matches and high-confidence pairs are perfect anchors:
if s≡t, then neighbors of s should preferentially match neighbors of t (the intuition behind
LogMap's anchor expansion and similarity flooding — both repeatedly effective in OAEI).

## Research questions

- **RQ02.1**: Does predicted-anchor neighborhood agreement improve ranking and global
  alignment on lexically weak pairs?
- **RQ02.2**: Which anchor source—exact-only or high-confidence predicted pairs—provides the
  best gain/error-propagation trade-off?
- **RQ02.3**: Does a second iteration add useful evidence or mainly reinforce first-pass
  errors, hubs, and popular targets?
- **RQ02.4**: Are improvements explained by genuinely coherent graph neighborhoods and do they
  extend to property/instance bundles once E11/E12 freeze those configurations?

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
3. Fold in as an additional channel through the existing σ-mixing (it gets a confidence
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

For each pass, persist the exact/predicted anchor set, how many anchors were added/removed,
their source (exact, base score, or LLM-influenced score), coverage/degree distribution, and
post-hoc precision against the reference. Also report downstream decision errors connected to
newly wrong anchors. Gold is joined only after all passes finish, so these diagnostics measure
error propagation without steering iteration.

**Promotion**: standard criteria; also require explanation invariant (decomposition exact) and
runtime within 1.2× (neighbor joins are index lookups over existing bundles — should hold).

Anchor propagation is the system's only structural method that needs no labels, and it is
therefore the `label_free` resolution of the structural component. E23 supplies the supervised
counterpart for inputs whose class hierarchy is too shallow for this channel's neighbor evidence
to carry weight, and is measured against whatever this experiment promotes.

**Effort**: M–L. **Risks**: error propagation from wrong anchors — mitigate with exact-only
anchor arm; hub entities inflating agreement — IC weighting should damp, verify on OAEI-KG.
