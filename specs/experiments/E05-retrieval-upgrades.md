# E05 — Candidate Retrieval Upgrades (encoder, fusion, adaptive k)

**Motivation** (audit obs. 3): retrieval is the recall ceiling — a gold target outside the
top-20 pool is unrecoverable. Today: dense channel uses `all-MiniLM-L6-v2` while scoring uses
SapBERT (encoder mismatch); channels merge by plain `max(semantic, lexical)`; k is a fixed 20
per source regardless of how concentrated the retrieval distribution is; the tooling to
measure recall already exists (`analysis/candidate_recall.py`).

**Hypothesis**: (a) SapBERT-based retrieval raises candidate recall ≥1 pt on biomedical tracks
at equal k (at some encoding-cost increase, it's the same model already used downstream);
(b) reciprocal-rank fusion (RRF) beats max-fusion at equal k; (c) adaptive k (grow the pool
until a score-gap/entropy criterion is met, capped) recovers most tail recall at a fraction of
the pool-size cost of raising k globally.

## Change

Config under `candidates.*` (all current values remain defaults): `encoder` (add SapBERT arm —
reuse the scorer's cached embedder), `fusion: max|rrf|weighted`, `adaptive_k: {enabled, gap
criterion, k_min, k_max}`. Implementation confined to `utils/candidate_generation.py` +
`impl/datasets/base.py` pool assembly.

## Arms & validation

Two-stage evaluation, cheap first:
1. **Retrieval-only** (no full runs): candidate recall + gold-rank median/p90 via the existing
   recall tool, all arms × all tasks, 1 seed (retrieval is deterministic on CPU). Kill arms
   that don't move recall.
2. **End-to-end** for surviving arms: full matrix, 3 seeds — recall gains must survive the
   selector (more candidates can add noise downstream; measure precision impact explicitly).

Primary: end-to-end macro F1 + local MRR; secondary: candidate recall, mean pool size,
dataset-build wall time (ledger).

**Promotion**: standard criteria; encoder swap additionally requires dataset-build time within
1.5× (SapBERT is heavier — batch-encode once, embeddings are already cached across runs).

**Effort**: M. **Risks**: MiniLM may actually win on non-biomedical tracks (Conference,
OAEI-KG) — if arms split by domain, promote per-track-profile encoder selection (config
already supports it via profiles) rather than a global swap.
