# E06 — String-Similarity Ensemble as a Second Lexical Signal

**Motivation** (audit obs. 4): scoring-time lexical evidence is *pure SapBERT cosine* — no
edit/character similarity anywhere in the scorer. Embedding geometry can miss trivially strong
string evidence (near-identical spellings, shared rare tokens, abbreviations) and is
domain-biased (SapBERT is biomedical; Conference/OAEI-KG labels are out-of-domain). PyLogMap's
parity work showed classic ISUB remains a strong, near-free signal.

## Research questions

- **RQ06.1**: Does a character/token string ensemble improve equivalence ranking and global F1
  beyond embedding-only lexical scoring?
- **RQ06.2**: Is the ensemble most useful out of domain and on property/instance names, or does
  it duplicate the retrieval/scoring evidence already present?
- **RQ06.3**: Is treating string similarity as an independent confidence-weighted channel
  better calibrated and more interpretable than taking a blunt maximum with the lexical score?
- **RQ06.4**: Which label-length, abbreviation/expansion, spelling-variation, and
  compositional-label slices explain its wins and false positives?

**Hypothesis**: an ISUB/Jaro-Winkler/token-set ensemble folded in as its own channel improves
F1 on non-biomedical tracks (Conference, OAEI-KG) ≥1 pt and is ≈neutral on Bio-ML (where
SapBERT already covers strings); it also improves `q_lex` reliability (margin quality) by
disagreeing with the encoder exactly where the encoder hallucinates similarity. A separate
abbreviation feature is expected to improve the abbreviation/expansion slice; no acronym claim
is attributed to ISUB/Jaro-Winkler/token-set alone.

## Change

New channel `strsim` through the standard σ-mixing (exact importance decomposition preserved):
- `s_strsim` = max over label-pair ensemble score: `max(isub, jaro_winkler, token_set_ratio)`
  (each on the normalized label forms already computed for exact matching);
- Optional `s_abbr`: detect a 2–10-character short form against the other label's token
  initials/camel humps, plus a conservative ordered-subsequence score for single compound
  expansions; require the first character to agree and apply length/coverage penalties. Keep
  this as a separately ablated sub-signal so common subsequences do not silently inflate the
  base ensemble;
- when enabled, `s_strsim_effective=max(s_strsim,s_abbr)`; report which sub-signal won for each
  pair and compute the channel margin from the effective score;
- `q_strsim` = top-1/top-2 margin (same rule as the lexical channel).
- Config `matching.channels.strsim: off|on` and
  `matching.channels.strsim.abbreviation: off|initialism` (+ ensemble weights, exposed per
  audit F5).
- Implementation: pure-Python first; ISUB is a named hot-kernel candidate under the
  performance policy (`03-performance.md`) if profiling demands — with the bit-identical
  fallback contract.

Touched: one new channel module, scorer channel registry, config.

## Arms & validation

off (baseline) / strsim-as-channel / strsim+abbreviation-as-channel /
strsim-folded-into-lex (replace `s_label` with `max(s_label, s_strsim)` — the blunt variant, to
test whether channel-level treatment matters). Full matrix, 3 seeds, global + local metrics.
Report per-track channel-importance shift and cases where strsim vetoes/rescues the encoder.
For RQ06.4, pre-register abbreviation pairs from label forms without using mapping correctness;
report the base ensemble and abbreviation sub-signal separately on that slice.

**Promotion**: standard; if gains are non-biomedical-only, promote as default-on with the
biomedical profile free to disable (per-profile default, like E05).

**Effort**: S–M. **Risks**: token-set ratio inflates scores on long compositional labels —
length-normalize; double-counting with retrieval's lexical score — they live at different
stages, but check the selector feature correlation matrix before/after. Ordered-subsequence
abbreviation matches can reward accidental short patterns; the first-character/length guards
and separate ablation prevent those errors being hidden in the base string result.
