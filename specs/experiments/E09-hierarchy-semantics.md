# E09 — Hierarchy Semantics: IC-Weighted Ancestor Overlap + Sibling Evidence

**Motivation** (audit obs. 7): the hierarchy channel compares *labels of depth≤2 ancestors*
(embedding support matrix over ≤6 triples/side with specificity 1/(d+1)) — it never uses the
ontologies' own set-theoretic structure. Classical semantic-similarity results (Resnik/Lin,
and their use across OM systems) show IC-weighted ancestor-set overlap is a strong equivalence
signal, and it becomes *cross-ontology computable* once anchors exist (shared upper structures,
exact-matched ancestors). Siblings are similarly untapped: two candidates whose parents match
but which have many exactly-matching siblings pointing elsewhere are likely non-matches.

**Hypothesis**: adding (a) an IC-weighted ancestor-overlap sub-signal (via anchor-mapped
ancestor sets) and (b) a sibling-consistency penalty improves hierarchy-channel discrimination
on deep taxonomies (SNOMED pairs, Anatomy) ≥0.5 F1 pts; label-embedding-only hierarchy is
retained as a fallback where anchors are sparse.

## Change

Inside `_score_hierarchy_family` (flag `matching.channels.hier.mode: labels|labels+overlap`):
- Ancestor sets to configurable depth (full closure now cheap post-WP-B) mapped across
  ontologies through the exact-match anchor table; overlap score = IC-weighted Jaccard
  (Lin-style normalization); blended into `s_f` with an exposed weight (audit F5 exposure).
- `q_f` gains an anchor-coverage term (overlap is only trustworthy when enough ancestors are
  anchor-mapped) — low coverage degrades gracefully to the current label-embedding behavior.
- Sibling term (separate flag): penalty proportional to the fraction of s's exact-matched
  siblings whose images are NOT siblings of t.
- Depends on cheap `ancestors()` from the WP-B hierarchy index; anchor table shared with E02
  (implementation coordination — build the anchor-mapping utility once).

## Arms & validation

labels (baseline) / +overlap / +overlap+siblings; depth ∈ {2, full}. Emphasis tracks: SNOMED-*,
Anatomy, OAEI-KG; full matrix for guard. 3 seeds. Secondary analysis: hierarchy-channel
quality/importance distributions; wins/losses vs baseline where lexical was ambiguous
(U ≥ τ_LLM slice — does better hierarchy reduce LLM invocations?).

**Promotion**: standard; an LLM-call reduction at neutral F1 also promotes (cost clause).

**Effort**: M. **Risks**: anchor sparsity on Conference (few exacts) — coverage term must keep
it neutral there (guard metric); IC estimates differ per ontology — use per-ontology
normalized IC (already available as `edge_ic_norm`).
