# E05 — Candidate Retrieval Upgrades (encoder, fusion, adaptive k)

**Motivation** (audit obs. 3): retrieval is the recall ceiling — a gold target outside the
top-20 pool is unrecoverable. Today: dense channel uses `all-MiniLM-L6-v2` while scoring uses
SapBERT (encoder mismatch); channels merge by plain `max(semantic, lexical)`; k is a fixed 20
per source regardless of how concentrated the retrieval distribution is; the tooling to
measure recall already exists (`analysis/candidate_recall.py`).

## Research questions

- **RQ05.1**: Which encoder and lexical/dense fusion maximizes candidate recall at a fixed mean
  pool size in biomedical and non-biomedical domains?
- **RQ05.2**: Does adaptive k recover tail gold candidates more efficiently than raising k for
  every source?
- **RQ05.3**: Do retrieval-recall gains survive end-to-end scoring/selection, or do larger and
  harder pools reduce precision or calibration?
- **RQ05.4**: Are retrieval choices transferable across entity kinds and input formats, or are
  kind/domain-specific profiles required?

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

Every arm emits a candidate-pool manifest containing retrieval config/model fingerprint,
source/target/data-lock hashes, per-kind pool hashes, mean/quantile pool size, and gold-free
pool summary. Downstream experiment manifests bind to this fingerprint.

## Arms & validation

Two-stage evaluation, cheap first:
1. **Development retrieval screening** (no full runs): candidate recall + gold-rank median/p90
   via the existing recall tool, all arms × development tasks, 1 seed under the deterministic-
   screening carve-out (retrieval is deterministic on CPU). Freeze the surviving arm(s) here.
2. **Reporting confirmation**: evaluate retrieval diagnostics and end-to-end results for the
   frozen arms on the reporting matrix in the same final pass, with 3 end-to-end seeds. No arm
   is killed or selected from reporting candidate recall. Recall gains must survive the selector
   because additional candidates can add downstream noise; measure precision explicitly.

Primary: end-to-end macro F1 + local MRR; secondary: candidate recall, mean pool size,
dataset-build wall time (ledger).

**Promotion**: standard criteria; encoder swap additionally requires dataset-build time within
1.5× (SapBERT is heavier — batch-encode once, embeddings are already cached across runs).

E05's promotion decision precedes all downstream candidate-consuming experiments. If an E05 arm
promotes, append the new rolling baseline and regenerate E00 capability/power artifacts before
those experiments freeze. A later retrieval change makes downstream product-promotion evidence
stale until E17 reconfirms it on the new pool.

**Pre-registered criterion-3 override**: the encoder-swap arm may use 1.5× dataset-build time
because SapBERT is inherently heavier and the embedding cache amortizes this cost across runs.
End-to-end scoring retains the default 1.2× bound, and the results note also reports whether
dataset construction passed that default bound.

Every arm here is zero-shot. E20 asks the supervised version of the same question — fine-tuning
the encoder on training mappings with hard negatives mined from these pools — and shares this
experiment's slot discipline, because it also changes candidate pools. The winning arm here is
the `label_free` resolution of the retrieval component and the baseline E20 must beat.

**Effort**: M. **Risks**: MiniLM may actually win on non-biomedical tracks (Conference,
OAEI-KG) — if arms split by domain, promote per-track-profile encoder selection (config
already supports it via profiles) rather than a global swap.
