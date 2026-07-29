# E21 — Supervised LLM Use: Exemplars, Learned Gating, and Distillation

**Motivation** (audit obs. 21; extends obs. 6): E07 makes the LLM's *frame* comparative, but
every one of its arms remains zero-shot, and the decision to invoke the LLM at all is still the
fixed uncertainty rule `U ≥ τ_LLM` with mixing weight `w_i = β·U`
(`pair_adaptive_scorer.py:678-683`). The only supervised LLM machinery that exists today
calibrates the returned probability after the fact
(`_apply_llm_calibration` / `_collect_calibration_samples`, `pair_adaptive_scorer.py:718-736`).

Three levers remain unused, and one of them is the sharpest cost result available in the
programme. The current gate fires on *uncertainty*, which is only a proxy for the thing that
matters: whether the call will change the decision. Most gated pairs are presumably confirmations
of what `S_base` already concluded, and every one of those is a paid call that buys nothing. With
training labels, a gate can be trained on the real target instead of the proxy.

## Research questions

- **RQ21.1**: Do k-NN-retrieved labelled exemplars in the prompt improve LLM decision accuracy
  over zero-shot at an equal number of calls?
- **RQ21.2**: Can a learned gate — trained to predict whether an LLM call will change the
  decision, and change it correctly — reduce calls at equal or better quality compared with
  `U ≥ τ_LLM`?
- **RQ21.3**: Can a student distilled from LLM judgements recover most of the LLM's contribution
  at a fraction of its cost, and on which slices does distillation fail?
- **RQ21.4**: Does a learned fusion weight beat `w_i = β·U`, and does refitting calibration on
  training labels beat the currently fitted calibration?
- **RQ21.5**: Do these gains hold across models, or are they artifacts of one model's behavior —
  and what happens to a fitted gate or student when the provider's model changes?

## Hypotheses

Exemplars help most on out-of-domain tracks, where the model has the weakest domain priors and
the most to gain from being shown what a correct alignment looks like in that vocabulary; on
Bio-ML the effect is expected to be small. Learned gating cuts calls by at least 50% at
non-inferior quality, because the majority of currently gated pairs are confirmations.
Distillation is expected to recover a majority but not all of the LLM's contribution, degrading
specifically where the contribution is genuine world knowledge rather than a recoverable pattern
over the evidence — which slice that is, is itself the interesting finding. A learned fusion
weight is expected to beat `β·U` modestly, since proportionality to uncertainty is an assumption
that has never been tested.

## Change

Under `llm.*`, defaulting to current behavior:

1. `llm.exemplars: off|knn` with `exemplar_count` and a retrieval configuration. Exemplars are
   retrieved from **training-split mappings only**, by embedding similarity to the query source
   entity. The harness asserts no exemplar originates in a validation or test reference, and that
   a query pair can never retrieve itself.
2. `llm.gate: uncertainty|learned`. The learned gate is a small classifier over the same
   pre-LLM features available at gating time — `S_base`, channel scores and qualities, `U`, pool
   statistics — trained on **counterfactual training data**: for a sample of training pairs, run
   the pipeline both with and without the LLM call and label each pair by whether the call
   changed the decision and whether the change was correct. This one-time training-split LLM
   sweep is a real cost and is reported as such. “Correct” is assigned only where the training
   reference is complete for that source/kind/relation or the outcome is explicitly adjudicated;
   an unlisted mapping in an incomplete reference is unknown and is excluded from the supervised
   gate target rather than labelled incorrect.
3. `llm.distill: off|student` — a student trained on LLM judgements over **training-split**
   sources plus gold labels, used in place of the call at inference. Training a student on LLM
   outputs over reporting sources would be transductive; if run at all, it is a labelled
   diagnostic arm that cannot promote.
4. `llm.fusion_weight: beta_u|learned` and a refit of the existing probability calibration on
   training labels.

Implementation boundary: prompt assembly and exemplar retrieval in `exact/impl/models/
semantic_llm.py`; the gate and student are heads under `exact/impl/models/`, consulted at the
existing gating and decision points in `pair_adaptive_scorer.py`. The router is untouched.

## Arms & validation

Development: exemplar count sweep and gate/student model selection on Bio-ML train/validation,
1 seed, pruned to one frozen configuration per lever.

Reporting (3 seeds): {zero-shot E07 winner, +exemplars} × {uncertainty gate, learned gate}, plus
the distilled student as its own arm, on eligible tracks with training splits and a declared
counterfactual-label/completeness policy. The label-free
comparator column is the promoted E07 configuration with its uncertainty gate.

Primary: macro F1. **Co-primary, declared before results open**: LLM calls and tokens. For the
gate and distillation arms the cost endpoint is primary and quality is held to a non-inferiority
margin, following E07's pattern — the CI lower bound of the quality delta must sit above −0.5
macro F1 points, and the call reduction must have a CI excluding zero.

Secondary: local MRR, decision-flip counts and their correctness, gate precision/recall against
the counterfactual label, per-slice distillation gap, ECE/Brier after fusion, and the one-time
counterfactual-collection and student-training costs reported separately from per-run cost.

All arms pin E00's complete LLM fingerprint. Because a learned gate and a distilled student are
fitted to one model's behavior, this experiment carries an additional artifact rule: the fitted
gate and student record the resolved model identity, and the harness refuses to apply them under
a different fingerprint. RQ21.5's cross-model replication runs on one development task with a
second pinned model; if the effect reverses, the claim is scoped to the primary model and
cross-model generalization is recorded as inconclusive rather than assumed.

## Promotion

Standard criteria for the exemplar and fusion-weight arms. For the gate and distillation arms,
the pre-registered cost-primary endpoint above governs.

- The learned gate and the distilled student ship as the **`supervised` resolution** of the LLM
  component under `supervision.mode`. The uncertainty gate remains the `label_free` resolution
  and stays the default wherever no training reference resolves.
- A fitted gate or student may only ship bound to an immutable model deployment. A provider
  exposing only a mutable alias cannot carry one of these artifacts into a product default, by
  the same rule E00 applies to confirmatory LLM comparisons.

**Paper contribution**: the literature reports LLM-assisted matching largely as an accuracy
story, with cost mentioned as an afterthought. RQ21.2 inverts that: it asks how much of the LLM
budget a matcher actually needs, and answers it with a gate trained on the counterfactual rather
than on a hand-set uncertainty threshold. The distillation slice analysis (RQ21.3) is the
complementary scientific question — it localizes what the LLM contributes that the evidence
model cannot reconstruct.

**Effort**: M–L. **Risks**: provider model drift invalidates fitted gates and students, which is
both an experimental hazard and a genuine product limitation that the results note must state
plainly; counterfactual data collection costs a full training-split LLM sweep, so budget it
before scheduling; exemplar retrieval is a leakage surface and is asserted, not assumed; a gate
that learns to suppress calls on a slice where the LLM was quietly helping would show up as a
slice regression, so per-slice reporting is mandatory rather than optional here.
