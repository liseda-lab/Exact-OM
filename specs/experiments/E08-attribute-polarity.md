# E08 — Attribute-Channel Polarity & Evidence Double-Counting

**Motivation** (audit obs. 5): two structural quirks in the attribute channel:
(a) `s_attr = max(τ, r_attr)` — attributes can only ever *support* a match, never oppose it,
even when, e.g., cross-references or codes actively contradict; (b) each side's attributes are
matched against a bank containing the **other side's labels + selected hierarchy/similarity
sentences**, so evidence already consumed by the lexical/hierarchy/sim channels re-enters
through the attribute channel — correlated channels inflate σ-mixing confidence and muddy the
"independent channels" explanation story.

**Hypothesis**: (a) allowing signed attribute evidence (identifier/xref mismatch pushes below
τ) raises precision on identifier-rich tracks (OMIM-ORDO, NCIT-DOID) with small recall cost;
(b) restricting the bank to attributes-vs-attributes (+labels only, no channel sentences)
reduces cross-channel correlation without hurting F1 — cleaner and at worst neutral.

## Change

Config under `matching.channels.attr`:
- `polarity: support_only|signed` — signed mode: for high-weight *identifier-class* properties
  (xref/id/code, weight ≥0.8 in the existing property-weight map), a strong best-match to a
  *different* entity's identifier bank contributes negative deviation (s_attr can drop below
  τ); free-text attributes stay support-only (definitions legitimately differ).
- `bank: full|attrs_labels|attrs_only` — controls the cross-channel bank composition.

Touched: `_score_attribute_channel` + bank builder only; mixing untouched (a below-τ `s_attr`
already flows correctly through σ-weighting).

## Arms & validation

2×3 factorial (polarity × bank), full matrix, 3 seeds. Primary: macro F1; secondary: per-track
P/R, channel-correlation matrix (Pearson over per-pair channel scores — report shrinkage),
and count of pairs where signed attributes flipped the decision (manual inspection sample of
30 flips for error analysis — are the vetoes right?).

**Promotion**: standard; the bank restriction additionally promotes on neutrality + correlation
reduction (explanation-quality win, same logic as E03's robustness clause).

**Effort**: M. **Risks**: xref conventions differ per ontology pair (same-as vs related-to
xrefs) — gate signed mode on property IRI allowlists per profile; sparse attributes on
Conference make (a) untestable there — scope claims to attribute-rich tracks.
