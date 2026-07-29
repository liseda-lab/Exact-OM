# E20 — Supervised Retrieval: Contrastive Fine-Tuning and Cross-Encoder Reranking

**Motivation** (audit obs. 20; extends obs. 3): E05 asks which *off-the-shelf* encoder to use and
how to fuse retrieval channels — every one of its arms is zero-shot. But retrieval is the stated
recall ceiling: a gold target outside the pool is unrecoverable no matter how good scoring,
fusion, or acceptance become, so it is the one stage where a label-driven gain cannot be
recovered elsewhere. The labels needed are already on disk. Training references supply positive
pairs, and the existing candidate pools supply hard negatives for free: the non-gold entries a
zero-shot encoder ranked above or beside the gold target are precisely the confusions a
fine-tuned encoder should learn to separate.

Supervised retrieval also carries a methodological trap that must be handled explicitly, and is
the reason this experiment is specified separately rather than folded into E05: once the encoder
is fitted, candidate recall measured on training-split sources is optimistic by construction.

## Research questions

- **RQ20.1**: Does contrastive fine-tuning of the retrieval encoder on training mappings raise
  candidate recall at fixed mean pool size over the best zero-shot E05 configuration?
- **RQ20.2**: Does a supervised cross-encoder over the top-k improve ranking beyond E18's
  feature-based reranker, and at what latency multiple?
- **RQ20.3**: When the same labelled source pool is reusable by every compatible stage, what is
  the marginal value of enabling supervised retrieval relative to fusion, reranking, and
  acceptance? This experiment contributes the retrieval arm of E22's shared-label comparison.
- **RQ20.4**: Do fine-tuned encoders transfer to other ontology pairs and domains, or do they
  overfit the training pair's vocabulary?
- **RQ20.5**: Do supervised recall gains survive end-to-end, or do the additional near-miss
  candidates they surface cost precision — E05's RQ05.3, re-asked with a supervised pool?

## Hypotheses

Contrastive fine-tuning raises recall@k by 2–4 points at equal pool size on biomedical tracks,
and by more on out-of-domain tracks where the base encoder is weakest and therefore has the most
headroom. The cross-encoder is expected to improve ranking but to cost at least 5× candidate-
scoring latency, most likely failing the cost gate as a shipped default while remaining valuable
as a quality ceiling that bounds what any feature-based reranker could achieve. Transfer across
pairs that share an ontology (SNOMED appears in two Bio-ML pairs) is expected to look
substantially better than transfer to disjoint pairs; reporting only the pooled number would
overstate generalization.

## Change

Under `candidates.*`, defaulting to current behavior:

1. `encoder_finetune: off|contrastive` with mining parameters. Positives are training-split
   mappings. A non-reference candidate is a hard negative only when the inventory declares that
   training reference complete for the task×kind×relation, or the dataset provides an explicit
   negative. In-batch negatives are filtered against every known positive and are used as
   confirmed negatives only under the same completeness rule. For `known_incomplete` references,
   use a pre-registered positive-unlabelled contrastive objective or semantically certified
   incompatible negatives; `unknown`-completeness negative-mining arms are descriptive and cannot
   establish a promotion claim. The fine-tuned encoder artifact records base model, training
   pairs, completeness/negative policy, mining configuration, dataset lock, epochs, and seed.
2. `candidates.cross_encoder: off|on` with `cross_encoder_top_k`. A supervised pair classifier
   re-scores the top-k pool entries before the pool is handed downstream.
3. The candidate-pool manifest defined in E05 gains the encoder-artifact hash. A pool built with
   a fitted encoder is never fingerprint-compatible with a zero-shot pool, and the harness must
   refuse to reuse a downstream fitted head across that boundary.

**Leakage discipline** (this experiment's central risk): encoder training consumes training-split
mappings only. Validation-split mappings may select hyperparameters and early stopping; test
mappings are never touched. The harness asserts this at startup, as it does for the selector.
Reference completeness controls the meaning of negatives, not merely reporting: absence from an
incomplete reference is never silently converted into a negative label.

**Optimistic-recall rule**: candidate recall is reported on **test-split sources** as the
headline number. Training-split recall is reported separately and labelled optimistic; it
measures memorization, not retrieval quality, and may never be used for an arm decision or a
promotion claim.

Implementation boundary: fine-tuning and cross-encoder inference live in
`exact/utils/candidate_generation.py` and the dataset pool assembly in
`exact/impl/datasets/base.py`, alongside E05's changes. Training itself is an experiment-side
tool, not runtime code; the runtime only loads a fitted artifact.

## Arms & validation

Following E05's two-stage shape, because retrieval screening is cheap and end-to-end runs are
not:

Stage 1 (development retrieval screening, 1 seed under the deterministic-screening carve-out):
{best zero-shot E05 arm, contrastive fine-tuned} × pool sizes, measured by candidate recall and
gold-rank median/p90 on development tasks. Freeze survivors here.

Stage 2 (reporting, 3 seeds end-to-end): frozen survivors × {cross_encoder off, on}, on Bio-ML
test and every eligible track with a training split and a declared negative-label policy. Report
the label-free comparator (the promoted zero-shot E05 configuration) in the same table, plus
`cross_pair_transfer` for the
fine-tuned encoder — separated into shared-ontology and disjoint-ontology slices, per RQ20.4.

Primary: end-to-end macro F1. Co-primary declared in advance: test-split candidate recall at
fixed mean pool size, since a retrieval experiment whose recall claim is only secondary cannot
answer RQ20.1. Secondary: local MRR/Hits@1, precision (RQ20.5), mean pool size, dataset-build and
per-candidate scoring wall time, peak memory, and encoder training cost reported once per
artifact.

## Promotion

Standard criteria, plus:

- E05's ordering discipline applies in full. A promotion here changes candidate pools, so every
  downstream experiment's product-promotion evidence becomes stale until refit and reconfirmed
  under E17. Schedule E20 in the same slot discipline as E05, never after downstream pools have
  been frozen.
- Fine-tuning ships as the **`supervised` resolution** of the retrieval component under
  `supervision.mode`; the promoted zero-shot configuration remains the `label_free` resolution.
- **Pre-registered criterion-3 override**: dataset-build time may reach 2× for the fine-tuned
  encoder arm, because encoding is amortized by the existing embedding cache across runs and the
  fitted artifact is trained once, not per run. Per-candidate scoring time keeps the default 1.2×
  bound. The results note reports both gates. No override is offered for the cross-encoder arm;
  it is a per-run inference cost and must pass the default bound to ship as a default.

**Paper contribution**: fine-tuned retrieval is standard practice in entity linking and largely
assumed in ontology matching, but the size of its contribution *relative to supervising later
stages of the same pipeline* is not established. Because E18, E19, and E20 share a harness,
tracks, and the same reusable labelled-source pool, this programme can report the marginal value
of enabling supervision at each stage without pretending that one mapping label is consumed by
only one component — a question usually answered one component at a time, on different data.

**Effort**: L. **Risks**: GPU training cost and reproducibility of the fitted artifact — pin
seeds, commit the artifact hash, document GPU jitter as E00 does; false hard negatives from
incomplete or one-to-many references — enforce the completeness/positive-unlabelled rule above;
overfitting to the training pair's vocabulary, which the disjoint-ontology transfer slice is
designed to expose; and the optimistic-recall trap above, which would silently invalidate RQ20.1
if training-split sources entered the headline number.
