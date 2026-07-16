# E06 — String-Similarity Ensemble as a Second Lexical Signal

**Motivation** (audit obs. 4): scoring-time lexical evidence is *pure SapBERT cosine* — no
edit/character similarity anywhere in the scorer. Embedding geometry can miss trivially strong
string evidence (near-identical spellings, shared rare tokens, acronym expansions) and is
domain-biased (SapBERT is biomedical; Conference/OAEI-KG labels are out-of-domain). PyLogMap's
parity work showed classic ISUB remains a strong, near-free signal.

**Hypothesis**: an ISUB/Jaro-Winkler/token-set ensemble folded in as its own channel improves
F1 on non-biomedical tracks (Conference, OAEI-KG) ≥1 pt and is ≈neutral on Bio-ML (where
SapBERT already covers strings); it also improves `q_lex` reliability (margin quality) by
disagreeing with the encoder exactly where the encoder hallucinates similarity.

## Change

New channel `strsim` through the standard σ-mixing (exact importance decomposition preserved):
- `s_strsim` = max over label-pair ensemble score: `max(isub, jaro_winkler, token_set_ratio)`
  (each on the normalized label forms already computed for exact matching);
- `q_strsim` = top-1/top-2 margin (same rule as the lexical channel).
- Config `matching.channels.strsim: off|on` (+ ensemble weights, exposed per audit F5).
- Implementation: pure-Python first; ISUB is a named hot-kernel candidate under the
  performance policy (`03-performance.md`) if profiling demands — with the bit-identical
  fallback contract.

Touched: one new channel module, scorer channel registry, config.

## Arms & validation

off (baseline) / strsim-as-channel / strsim-folded-into-lex (replace `s_label` with
`max(s_label, s_strsim)` — the blunt variant, to test whether channel-level treatment matters).
Full matrix, 3 seeds, global + local metrics. Report per-track channel-importance shift and
cases where strsim vetoes/rescues the encoder (qualitative table for the explanations story).

**Promotion**: standard; if gains are non-biomedical-only, promote as default-on with the
biomedical profile free to disable (per-profile default, like E05).

**Effort**: S–M. **Risks**: token-set ratio inflates scores on long compositional labels —
length-normalize; double-counting with retrieval's lexical score — they live at different
stages, but check the selector feature correlation matrix before/after.
